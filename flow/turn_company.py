"""Size vs content at the turn: does the company (10 nearest neighbours) survive the swaps that keep the turn?
(exploratory, review follow-up; plan in research/NOTE_neurons_vs_cells.md, "S"). Writes results/turn_company_<tag>.json.

Same positions as neuron_cells.py test C (2,000 tokens of tok_meta eval texts 0-99). At both turn layers together:
  all       every neuron set to its mean (a constant write: invisible to a centered cosine)
  resample  every neuron taken from another text's token (a real write, wrong content)
  noise     the MLP output (before the post-norm) replaced by Gaussian noise of the same norm per token
  top1024 / rand1024   mean ablation of the top-1,024 neurons by write energy / 1,024 random neurons
Measured per hidden state: centered cosine with the clean state at the same depth, 10-NN overlap with the clean state
(cosine on centered tokens), and within the condition the centered cosine and 10-NN overlap across the turn.

python turn_company.py --model google/gemma-4-E2B --layers 13 14 --tag e2b
"""
import argparse
import json
import re

import numpy as np
import torch

from common import CACHE_DIRS, RESULTS, set_threads
from extract_hf import load_causal_lm
from lens import log
from neuron_cells import unit

N_EVAL, RANK_TEXTS, N_POS, MAX_LEN, K = 100, (100, 130), 2000, 64, 10


def knn(U, k=K):
    S = U @ U.T
    np.fill_diagonal(S, -np.inf)
    return np.argpartition(-S, k, axis=1)[:, :k]


def overlap(A, B):
    return float(np.mean([len(np.intersect1d(a, b)) / K for a, b in zip(A, B)]))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B")
    ap.add_argument("--layers", nargs=2, type=int, default=[13, 14])
    ap.add_argument("--tag", default="e2b")
    a = ap.parse_args()
    LAYERS = tuple(a.layers)
    L0, L1 = LAYERS
    set_threads()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_causal_lm(a.model, torch.bfloat16, "cuda")
    layers = {int(r.group(1)): m for n, m in model.named_modules() if (r := re.search(r"language_model\.layers\.(\d+)$", n))}
    width = {L: layers[L].mlp.down_proj.in_features for L in LAYERS}
    cur, sel, mode, gen = {}, {}, ["mean"], torch.Generator(device="cuda")

    def pre(L):
        def f(mod, inp):
            if sel.get(L) is None or mode[0] == "noise":
                cur[L] = inp[0]
                return None
            x = inp[0].clone()
            ix = sel[L].to(x.device)
            if mode[0] == "mean":
                x[..., ix] = mean_a[L][ix].to(x.dtype)
            else:
                rows = torch.randint(0, len(pool_a[L]), x.shape[:-1], generator=gen, device="cuda")
                x[..., ix] = pool_a[L][rows][..., ix].to(x.dtype)
            return (x,)
        return f

    def post(L):
        def f(mod, inp, out):
            if sel.get(L) is None or mode[0] != "noise":
                return None
            z = torch.randn(out.shape, generator=gen, device=out.device, dtype=torch.float32)
            return (z * out.float().norm(dim=-1, keepdim=True) / z.norm(dim=-1, keepdim=True)).to(out.dtype)
        return f
    for L in LAYERS:
        layers[L].mlp.down_proj.register_forward_pre_hook(pre(L))
        layers[L].mlp.down_proj.register_forward_hook(post(L))

    meta = np.load(next(d / "tok_meta.npz" for d in CACHE_DIRS if (d / "tok_meta.npz").exists()), allow_pickle=True)
    enc_t = lambda ts: [tok(str(t), truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda") for t in ts]
    evals, rank_enc = enc_t(meta["eval_texts"][:N_EVAL]), enc_t(meta["eval_texts"][slice(*RANK_TEXTS)])
    hidden = layers[L0].mlp.down_proj.out_features
    gw = {L: layers[L].post_feedforward_layernorm(torch.ones(1, hidden, device="cuda", dtype=torch.bfloat16)).float()[0]
          for L in LAYERS}
    Wd = {L: layers[L].mlp.down_proj.weight.float().T * gw[L] for L in LAYERS}   # as neuron_cells.py ranks them
    energy, mean_a, pool_a, ntok = {L: 0 for L in LAYERS}, {L: 0 for L in LAYERS}, {L: [] for L in LAYERS}, 0
    for e in rank_enc:
        model(**e, use_cache=False)
        for L in LAYERS:
            x = cur[L][0].float()
            energy[L] = energy[L] + ((x ** 2) * (Wd[L] ** 2).sum(1)).sum(0)
            mean_a[L] = mean_a[L] + x.sum(0)
            pool_a[L].append(x[1:])
        ntok += e["input_ids"].shape[1]
    mean_a = {L: v / ntok for L, v in mean_a.items()}
    pool_a = {L: torch.cat(v) for L, v in pool_a.items()}
    g = torch.Generator().manual_seed(0)
    conds = {"clean": ("mean", None),
             "all": ("mean", {L: torch.arange(width[L]) for L in LAYERS}),
             "resample": ("resample", {L: torch.arange(width[L]) for L in LAYERS}),
             "noise": ("noise", {L: torch.arange(width[L]) for L in LAYERS}),
             "top1024": ("mean", {L: torch.argsort(energy[L], descending=True)[:1024] for L in LAYERS}),
             "rand1024": ("mean", {L: torch.randperm(width[L], generator=g)[:1024] for L in LAYERS})}

    rng = np.random.default_rng(0)                                     # same positions as neuron_cells.py test C
    allpos = [(i, p) for i, e in enumerate(evals) for p in range(1, e["input_ids"].shape[1])]
    by_text = {}
    for k in sorted(rng.choice(len(allpos), min(N_POS, len(allpos)), replace=False).tolist()):
        i, p = allpos[k]
        by_text.setdefault(i, []).append(p)
    states, kl, t1 = {c: [] for c in conds}, {c: [] for c in conds}, {c: [] for c in conds}
    for i, e in enumerate(evals):
        clean_lp = None
        for c, (md, s) in conds.items():
            sel.clear()
            sel.update(s or {})
            mode[0] = md
            gen.manual_seed(i)
            out = model(**e, output_hidden_states=True, use_cache=False)
            lp = torch.log_softmax(out.logits[0].float(), -1)
            if clean_lp is None:
                clean_lp = lp
            kl[c].append((clean_lp.exp() * (clean_lp - lp)).sum(-1).cpu().numpy())
            t1[c].append((clean_lp.argmax(-1) == lp.argmax(-1)).cpu().numpy())
            if i in by_text:
                states[c].append(torch.stack([h[0, by_text[i]] for h in out.hidden_states], 0).float().cpu().numpy())
        if i % 25 == 0:
            log("text", i)
    sel.clear()

    depths = list(range(L0 - 1, L1 + 3)) + [L1 + 6, len(layers)]
    U = {c: {d: unit(st[d] - st[d].mean(0)) for d in depths} for c, st in ((c, np.concatenate(v, 1)) for c, v in states.items())}
    NN = {c: {d: knn(U[c][d]) for d in depths} for c in conds}
    res = {"model": a.model, "layers": LAYERS, "depths": depths, "k": K, "n_pos": int(len(U["clean"][L0])), "conds": {}}
    for c in conds:
        r = {"kl": float(np.concatenate(kl[c]).mean()), "top1": float(np.concatenate(t1[c]).mean()),
             "turn": float((U[c][L0] * U[c][L1 + 1]).sum(1).mean()),
             "company_across_turn": overlap(NN[c][L0], NN[c][L1 + 1]),
             "cos_vs_clean": {d: float((U[c][d] * U["clean"][d]).sum(1).mean()) for d in depths},
             "company_vs_clean": {d: overlap(NN[c][d], NN["clean"][d]) for d in depths}}
        res["conds"][c] = r
        log(c, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)},
            "company vs clean", {d: round(v, 2) for d, v in r["company_vs_clean"].items()})
    res["chance_company"] = K / (res["n_pos"] - 1)
    (RESULTS / f"turn_company_{a.tag}.json").write_text(json.dumps(res, indent=1))
    log("saved")


if __name__ == "__main__":
    main()
