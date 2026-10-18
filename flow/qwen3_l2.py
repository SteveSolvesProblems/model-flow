"""Qwen3-1.7B's apparent turn at layer 2: is the few-coordinate event the model's own, and does it matter?
(exploratory; plan in research/NOTE_neurons_vs_cells.md, "Q2"). Writes results/qwen3_l2.json.

Massive dimensions: mean |x| > 20x the median over dimensions, on tokens of tok_meta eval texts 100-129, at the output
of decoder layer 2 (hidden 3). Conditions at that output: clean; massive dims set to 0; set to their mean; held at their
value entering layer 2 (the switch-on blocked). Measured: KL(clean || condition) and top-1 agreement on 50 eval texts;
centered cosine between adjacent hidden states (mean over tokens) on 1,000 DBpedia test articles, on all coordinates
and with the massive ones left out.

python qwen3_l2.py [--from-pos 1]   (--from-pos 1: leave the first token alone and take means over the others;
                                     added after the first run, whose all-position mean put the first token's values
                                     on every token)
"""
import argparse
import json
import re

import numpy as np
import torch

from common import CACHE_DIRS, RESULTS, set_threads
from extract_hf import dbpedia, load_causal_lm
from lens import log

MODEL, LAYER, THR, MAX_LEN, N_ART, N_TEXTS, RANK, BATCH = "Qwen/Qwen3-1.7B", 2, 20, 64, 1000, 50, (100, 130), 16
CONDS = ("clean", "zero", "mean", "hold")


def ccos(A, B):
    A, B = A - A.mean(0), B - B.mean(0)
    return float(((A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))).mean())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-pos", type=int, default=0)
    P0 = ap.parse_args().from_pos
    set_threads()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = load_causal_lm(MODEL, torch.bfloat16, "cuda")
    layer = next(m for n, m in model.named_modules() if re.search(rf"layers\.{LAYER}$", n))
    st = {"cond": "clean"}

    def pre(mod, args, kwargs):
        st["x_in"] = (args[0] if args else kwargs["hidden_states"]).detach()

    def post(mod, inp, out):
        c = st["cond"]
        if c == "clean":
            return None
        h = (out[0] if isinstance(out, tuple) else out).clone()
        ix = st["ix"]
        if c == "zero":
            h[:, P0:, ix] = 0
        elif c == "mean":
            h[:, P0:, ix] = st["mu"].to(h.dtype)
        else:
            h[:, P0:, ix] = st["x_in"][:, P0:, ix]
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    layer.register_forward_pre_hook(pre, with_kwargs=True)
    layer.register_forward_hook(post)

    meta = np.load(next(d / "tok_meta.npz" for d in CACHE_DIRS if (d / "tok_meta.npz").exists()), allow_pickle=True)
    enc_t = lambda ts: [tok(str(t), truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda") for t in ts]
    evals, rank_enc = enc_t(meta["eval_texts"][:N_TEXTS]), enc_t(meta["eval_texts"][slice(*RANK)])
    H = torch.cat([model(**e, output_hidden_states=True, use_cache=False).hidden_states[LAYER + 1][0].float() for e in rank_enc])
    H0 = torch.stack([model(**e, output_hidden_states=True, use_cache=False).hidden_states[LAYER + 1][0, 0].float() for e in rank_enc])
    m = H.abs().mean(0)                                                # dims found on all tokens, as in the first run
    ix = torch.nonzero(m > THR * m.median()).flatten()
    Hm = torch.cat([model(**e, output_hidden_states=True, use_cache=False).hidden_states[LAYER + 1][0, P0:].float() for e in rank_enc])
    st["ix"], st["mu"] = ix, Hm[:, ix].mean(0)
    Hin = torch.cat([model(**e, output_hidden_states=True, use_cache=False).hidden_states[LAYER][0].float() for e in rank_enc])
    res = {"model": MODEL, "layer": LAYER, "massive_dims": ix.tolist(),
           "massive_mean_abs_out": m[ix].tolist(), "massive_mean_abs_in": Hin[:, ix].abs().mean(0).tolist(),
           "median_abs_out": float(m.median()), "first_token_abs": H0[:, ix].abs().mean(0).tolist(), "from_pos": P0, "conds": {}}
    log("massive dims", res["massive_dims"], [round(v) for v in res["massive_mean_abs_out"]], "in", [round(v) for v in res["massive_mean_abs_in"]])

    texts = dbpedia(10, 6000)[0]["test"][0][:N_ART]
    keep = np.setdiff1d(np.arange(H.shape[1]), ix.cpu().numpy())
    for c in CONDS:
        st["cond"] = c
        kl = top1 = n = 0.0
        for e in evals:
            st["cond"] = "clean"
            ref = torch.log_softmax(model(**e, use_cache=False).logits[0, 1:].float(), -1)
            st["cond"] = c
            lp = torch.log_softmax(model(**e, use_cache=False).logits[0, 1:].float(), -1)
            kl += float((ref.exp() * (ref - lp)).sum())
            top1 += float((ref.argmax(-1) == lp.argmax(-1)).sum())
            n += ref.shape[0]
        hs = None
        for s in range(0, N_ART, BATCH):
            enc = tok(texts[s:s + BATCH], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
            out = model(**enc, output_hidden_states=True, use_cache=False).hidden_states[:LAYER + 4]
            w = enc["attention_mask"].unsqueeze(-1).float()
            pooled = [((h.float() * w).sum(1) / w.sum(1)).cpu().numpy() for h in out]
            hs = pooled if hs is None else [np.concatenate([a, b]) for a, b in zip(hs, pooled)]
        r = {"kl": kl / n, "top1": top1 / n,
             "adj_cos": [ccos(hs[k], hs[k + 1]) for k in range(len(hs) - 1)],
             "adj_cos_without_massive": [ccos(hs[k][:, keep], hs[k + 1][:, keep]) for k in range(len(hs) - 1)]}
        res["conds"][c] = r
        log(c, {k: (round(v, 4) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in r.items()})
    (RESULTS / f"qwen3_l2{'_from' + str(P0) if P0 else ''}.json").write_text(json.dumps(res, indent=1))
    log("saved")


if __name__ == "__main__":
    main()
