"""research/PREREG_bins_w3.md: W2-b's patching test over more blocks (W3-a) and more K (W3-b).

Same pairs, matching, patching and scoring as bins_w2b.py (imported). Writes results/bins_w3.json after each
configuration (DIAG_RESULTS can point elsewhere).

python bins_w3.py
"""
import json

import numpy as np
import torch

from bins_w2b import MODELS, MAX_LEN, choose_pairs, decoder_layers, score
from common import CACHE_DIRS, RESULTS, cache_meta, load_cache, set_threads
from extract_hf import load_causal_lm, log
from partition import assign, kmeans
from quant_lens import fit_frame

PLAN = {"gemma4_e2b": [(b, 512) for b in (9, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30)] + [(26, 32), (26, 4096)],
        "qwen3_17b": [(b, 512) for b in (4, 7, 10, 14, 17, 21, 24)] + [(14, 32), (14, 4096)]}
OUT = RESULTS / "bins_w3.json"


@torch.no_grad()
def run_model(key, spec, res):
    from transformers import AutoTokenizer
    meta = np.load(next(c / spec["meta"] for c in CACHE_DIRS if (c / spec["meta"]).exists()), allow_pickle=True)
    rows, texts = meta["eval"], meta["eval_texts"]
    nm_of = lambda m: f'{spec["cache"]}_L{m + 1:02d}'
    dtype = getattr(torch, cache_meta(nm_of(spec["L"] // 2)).get("dtype", "float16"))
    tok = AutoTokenizer.from_pretrained(spec["model"])
    model = load_causal_lm(spec["model"], dtype, "cuda")
    layers = decoder_layers(model, spec["L"])
    state = {"m": None, "vecs": None, "p": None}

    def hook(mod, inp, out):
        if state["vecs"] is None or mod is not layers[state["m"]]:
            return out
        h = (out[0] if isinstance(out, tuple) else out).clone()
        for b, v in enumerate(state["vecs"]):
            if v is not None:
                h[b, state["p"]] = v.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    hooks = [l.register_forward_hook(hook) for l in layers]
    cache = {}
    for m, K in PLAN[key]:
        tag = f"{key}_m{m}_K{K}"
        if tag in res:
            continue
        if m not in cache:
            cache.clear()
            Xf, Xe = load_cache(nm_of(m), "fit"), load_cache(nm_of(m), "eval")
            mu, W = fit_frame(Xf)
            cache[m] = (Xe, (Xf - mu) @ W, (Xe - mu) @ W)
        Xe, Zf, Ze = cache[m]
        bins = assign(Ze, kmeans(Zf, K, seed=0, nredo=1))[0]
        pairs, dw, dr = choose_pairs(Ze, Xe, bins, rows, np.random.default_rng(0))
        log(tag, "pairs", len(pairs), "no whiten partner", dw, "no raw partner", dr)
        state["m"] = m
        kl = {k: [] for k in ("self", "same", "diff_w", "diff_raw")}
        cos_check = []
        for i, pr in enumerate(pairs):
            r = pr["r"]
            d_, p = int(rows[r, 0]), int(rows[r, 1])
            ids = tok(str(texts[d_]), truncation=True, max_length=MAX_LEN, return_tensors="pt")["input_ids"]
            assert int(ids[0, p]) == int(rows[r, 2]), "token mismatch"
            variants = [("base", None), ("self", r), ("same", pr["same"]), ("diff_w", pr["diff_w"]), ("diff_raw", pr["diff_raw"])]
            variants = [(k, v) for k, v in variants if k == "base" or v is not None]
            state["vecs"] = [None if v is None else torch.from_numpy(Xe[v]).cuda() for _, v in variants]
            state["p"] = p
            out = model(input_ids=ids.repeat(len(variants), 1).cuda(), use_cache=False, output_hidden_states=(i < 20))
            lp = torch.log_softmax(out.logits[:, p].float(), -1)
            k_ = (lp[0].exp() * (lp[0] - lp)).sum(-1).cpu().numpy()
            for j, (k, _) in enumerate(variants[1:], start=1):
                kl[k].append((i, float(k_[j])))
            if i < 20:
                h = out.hidden_states[m + 1][0, p].float().cpu().numpy()
                cos_check.append(float(h @ Xe[r] / (np.linalg.norm(h) * np.linalg.norm(Xe[r]) + 1e-12)))
            state["vecs"] = None
        res[tag] = {"block": m, "K": K, "n_pairs": len(pairs), "no_partner_whiten": dw, "no_partner_raw": dr,
                    "cos_unpatched_vs_cache": cos_check, "kl": kl}
        res[tag]["score"] = score(res[tag])
        OUT.write_text(json.dumps(res))
        log(tag, json.dumps(res[tag]["score"]))
    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()


def eff(res, tag):
    s = res[tag]["score"]
    return s["diff_w"]["paired"], s["void"]


def summarize(res):
    out = {}
    q = [eff(res, f"qwen3_17b_m{b}_K512") for b, _ in PLAN["qwen3_17b"][:7]]
    out["W3a1"] = {"supported_blocks": sum(p["mean"] > 0 and p["excludes_0"] and not v for p, v in q), "of": 7}
    out["W3a1"]["supported"] = out["W3a1"]["supported_blocks"] >= 6
    gb = [b for b, _ in PLAN["gemma4_e2b"][:11]]
    pos = [(lambda p, v: p["mean"] > 0 and p["excludes_0"] and not v)(*eff(res, f"gemma4_e2b_m{b}_K512")) for b in gb]
    best = None
    for i, b in enumerate(gb):
        errors = sum(not x for x in pos[i:]) + sum(x for x in pos[:i])
        if errors <= 1 and any(pos[i:]):
            best = b
            break
    out["W3a2"] = {"blocks": gb, "positive": pos, "b_star": best, "switch_exists": best is not None,
                   "in_20_26": best is not None and 20 <= best <= 26}
    rb = {}
    for key, m in (("gemma4_e2b", 26), ("qwen3_17b", 14)):
        e = {K: res[f"{key}_m{m}_K{K}"] for K in (32, 512, 4096)}

        def paired(t):
            same = dict(t["kl"]["same"])
            return np.array([v - same[i] for i, v in t["kl"]["diff_w"]])
        a, b = paired(e[32]), paired(e[4096])
        rng = np.random.default_rng(0)
        bs = np.array([a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
                       for _ in range(10000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p4096 = e[4096]["score"]["diff_w"]["paired"]
        rb[key] = {"effect": {K: e[K]["score"]["diff_w"]["paired"]["mean"] for K in e},
                   "ci": {K: e[K]["score"]["diff_w"]["paired"]["ci95"] for K in e},
                   "diff_32_minus_4096": float(a.mean() - b.mean()), "diff_ci95": [float(lo), float(hi)],
                   "coarse_only": bool(lo > 0 and not p4096["excludes_0"])}
    out["W3b1"] = {"per_model": rb, "supported": all(v["coarse_only"] for v in rb.values())}
    return out


def main():
    set_threads()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for key in PLAN:
        run_model(key, MODELS[key], res)
    res["summary"] = summarize(res)
    OUT.write_text(json.dumps(res))
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
