"""M3 with groups of neurons (exploratory; plan in research/NOTE_neurons_vs_cells.md, "M3g").

Single-neuron ablations sat on a bfloat16 noise floor (macro_micro.py run 2), so here each ablation replaces a group of
128 neurons at one layer with their means. Groups: neurons sorted by mean |a| on the ranking texts (tok_meta eval
texts 100-129) and cut into contiguous bins of 128, so groups span small to large; 48 bins per layer (every other bin
at the 12,288-wide layer 22) plus 8 random groups. Predictors summed over a group, on the same 50 evaluation texts
as the KL: act (mean |a_i|), write (|a_i| |w_i| / rms(MLP output), after the post-norm) and write_rel (write / norm of
the residual entering the layer). Writes results/m3_groups.json.

python m3_groups.py
"""
import json
import re

import numpy as np
import torch
from scipy.stats import spearmanr

from common import CACHE_DIRS, RESULTS, set_threads
from extract_hf import load_causal_lm
from lens import log

MODEL, MAX_LEN = "google/gemma-4-E2B", 64
LAYERS, G, N_BIN, N_RAND, N_TEXTS, RANK, B = (8, 13, 14, 22), 128, 48, 8, 50, (100, 130), 4
PRED = ("act", "write", "write_rel")


@torch.no_grad()
def main():
    set_threads()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = load_causal_lm(MODEL, torch.bfloat16, "cuda")
    layers = {int(r.group(1)): m for n, m in model.named_modules() if (r := re.search(r"language_model\.layers\.(\d+)$", n))}
    cur, sel = {}, {}

    def hook(L):
        def f(mod, inp):
            if L in sel:                                              # row b of the batch ablates group sel[L][b]
                x = inp[0].clone()
                for b, (ix, val) in enumerate(sel[L]):
                    x[b, :, ix] = val.to(x.dtype)
                return (x,)
            cur[L] = inp[0]
        return f
    hooks = [layers[L].mlp.down_proj.register_forward_pre_hook(hook(L)) for L in LAYERS]

    meta = np.load(next(d / "tok_meta.npz" for d in CACHE_DIRS if (d / "tok_meta.npz").exists()), allow_pickle=True)
    enc_t = lambda ts: [tok(str(t), truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda") for t in ts]
    evals, rank_enc = enc_t(meta["eval_texts"][:N_TEXTS]), enc_t(meta["eval_texts"][slice(*RANK)])
    mean_a, abs_a, ntok = {L: 0 for L in LAYERS}, {L: 0 for L in LAYERS}, 0
    for e in rank_enc:
        model(**e, use_cache=False)
        for L in LAYERS:
            mean_a[L] = mean_a[L] + cur[L][0].float().sum(0)
            abs_a[L] = abs_a[L] + cur[L][0].float().abs().sum(0)
        ntok += e["input_ids"].shape[1]
    mean_a = {L: v / ntok for L, v in mean_a.items()}
    rng = np.random.default_rng(0)
    groups = {}
    for L in LAYERS:
        order = torch.argsort(abs_a[L]).cpu().numpy()
        bins = [order[i * G:(i + 1) * G] for i in range(len(order) // G)]
        bins = bins[::len(bins) // N_BIN][:N_BIN]
        groups[L] = bins + [rng.choice(len(order), G, replace=False) for _ in range(N_RAND)]
    nG = N_BIN + N_RAND

    Wd, gw, eps = {}, {}, {}
    for L in LAYERS:
        gw[L] = layers[L].post_feedforward_layernorm(torch.ones(1, 1536, device="cuda", dtype=torch.bfloat16)).float()[0]
        Wd[L] = layers[L].mlp.down_proj.weight.float()
        eps[L] = getattr(layers[L].post_feedforward_layernorm, "eps", 1e-6)
    wn = {L: (Wd[L] * gw[L][:, None]).norm(dim=0) for L in LAYERS}
    pred = {L: {k: np.zeros(nG) for k in PRED} for L in LAYERS}
    kl = {L: np.zeros(nG) for L in LAYERS}
    npos, null = 0, 0.0
    for t, e in enumerate(evals):
        o = model(**e, output_hidden_states=True, use_cache=False)
        batch = {k: v.expand(B, -1) for k, v in e.items()}
        clean = torch.log_softmax(model(**batch, use_cache=False).logits[:, 1:].float(), -1)   # same batch shape
        pc = clean.exp()
        null += float((pc[0] * (clean[0] - clean[1])).sum())
        for L in LAYERS:
            a = cur[L][0, 1:].float()
            rms = torch.sqrt(((a @ Wd[L].T) ** 2).mean(-1) + eps[L])
            hn = o.hidden_states[L][0, 1:].float().norm(dim=-1)
            size = a.abs() * wn[L] / rms[:, None]
            per = {"act": a.abs().sum(0), "write": size.sum(0), "write_rel": (size / hn[:, None]).sum(0)}
            for k in PRED:
                v = per[k].cpu().numpy()
                pred[L][k] += np.array([v[g].sum() for g in groups[L]])
        npos += clean.shape[1]
        for L in LAYERS:
            for s in range(0, nG, B):
                gs = groups[L][s:s + B]
                sel.clear()
                sel[L] = [(torch.as_tensor(g, device="cuda"), mean_a[L][torch.as_tensor(g, device="cuda")]) for g in gs]
                lp = torch.log_softmax(model(**{k: v[:len(gs)] for k, v in batch.items()}, use_cache=False).logits[:, 1:].float(), -1)
                kl[L][s:s + B] += (pc[:len(gs)] * (clean[:len(gs)] - lp)).sum(-1).sum(1).cpu().numpy()
        sel.clear()
        if t % 10 == 0:
            log("text", t)
    for h in hooks:
        h.remove()

    kl = {L: v / npos for L, v in kl.items()}
    pred = {L: {k: v / npos for k, v in d.items()} for L, d in pred.items()}
    res = {"group_size": G, "n_texts": N_TEXTS, "n_pos": npos, "kl_null_identical_rows": null / npos,
           "per_layer": {}, "groups": {L: [g.tolist() for g in gs] for L, gs in groups.items()},
           "kl": {L: v.tolist() for L, v in kl.items()}, "pred": {L: {k: v.tolist() for k, v in d.items()} for L, d in pred.items()}}
    pooled = {k: [] for k in PRED}
    klp = []
    for L in LAYERS:
        res["per_layer"][L] = {k: float(spearmanr(pred[L][k], kl[L])[0]) for k in PRED}
        res["per_layer"][L].update(kl_min=float(kl[L].min()), kl_median=float(np.median(kl[L])), kl_max=float(kl[L].max()),
                                   kl_random_median=float(np.median(kl[L][N_BIN:])))
        for k in PRED:
            pooled[k] += list(pred[L][k])
        klp += list(kl[L])
    res["pooled"] = {k: float(spearmanr(v, klp)[0]) for k, v in pooled.items()}
    res["M3g_pass"] = res["pooled"]["write_rel"] >= 0.5 and res["pooled"]["write_rel"] > res["pooled"]["write"]
    log("M3g", {k: res[k] for k in ("kl_null_identical_rows", "per_layer", "pooled", "M3g_pass")})
    (RESULTS / "m3_groups.json").write_text(json.dumps(res, indent=1, default=str))
    log("saved")


if __name__ == "__main__":
    main()
