"""research/PREREG_bins_w2.md W2-b: do bin boundaries matter to the model beyond distance?

Per model and layer m (residual after decoder block m = cache layer m+1; m = round(L/4), round(L/2), round(3L/4)):
bins = K = 512 k-means (seed 0) in the whiten-64 frame fit on the fit tokens (as token_lens.py), eval tokens assigned.
1,000 same-bin pairs (recipient, donor at the same position 8-63, different texts); for each, a different-bin donor
for the same recipient at a distance within +-2% of the same-bin donor's, matched in the whiten-64 frame (primary) and
in raw coordinates (secondary). Patch: replace the recipient's state after block m at that position with the donor's
cached state; outcome KL(unpatched || patched) of the next-token distribution there. Noise floor: patch with the
recipient's own cached state. Writes results/bins_w2b.json (after each layer).

DIAG_LOW_VRAM=1 DIAG_ATTN=eager python bins_w2b.py
"""
import json

import numpy as np
import torch

from common import CACHE_DIRS, RESULTS, cache_meta, load_cache, set_threads
from extract_hf import load_causal_lm, log
from partition import assign, kmeans
from quant_lens import fit_frame

MODELS = {"gemma4_e2b": {"model": "google/gemma-4-E2B", "cache": "tok_base_bos", "meta": "tok_meta_bos.npz", "L": 35},
          "qwen3_17b": {"model": "Qwen/Qwen3-1.7B", "cache": "tok_qwen3", "meta": "tok_meta_qwen3.npz", "L": 28}}
N_PAIRS, TOL, MAX_LEN, K = 1000, 0.02, 64, 512
OUT = RESULTS / "bins_w2b.json"


def decoder_layers(m, n):
    cands = [(name, mod) for name, mod in m.named_modules() if isinstance(mod, torch.nn.ModuleList)
             and name.endswith("layers") and len(mod) == n and "vision" not in name and "audio" not in name]
    assert len(cands) == 1, [c[0] for c in cands]
    return cands[0][1]


def choose_pairs(Ze, Xe, bins, rows, rng):
    pos, doc = rows[:, 1], rows[:, 0]
    elig = np.flatnonzero((pos >= 8) & (pos <= 63))
    by_pos = {p: np.flatnonzero(pos == p) for p in np.unique(pos[elig])}
    order = rng.permutation(elig)
    pairs, dropped_w, dropped_r = [], 0, 0
    for r in order:
        if len(pairs) >= N_PAIRS:
            break
        c = by_pos[pos[r]]
        c = c[doc[c] != doc[r]]
        same = c[bins[c] == bins[r]]
        if len(same) == 0:
            continue
        d = int(rng.choice(same))
        diff = c[bins[c] != bins[r]]
        out = {"r": int(r), "same": d, "diff_w": None, "diff_raw": None}
        for key, Z in (("diff_w", Ze), ("diff_raw", Xe)):
            ds = np.linalg.norm(Z[d] - Z[r])
            dd = np.linalg.norm(Z[diff] - Z[r], axis=1)
            ok = np.flatnonzero(np.abs(dd - ds) <= TOL * ds)
            if len(ok):
                out[key] = int(diff[ok[np.argmin(np.abs(dd[ok] - ds))]])
        dropped_w += out["diff_w"] is None
        dropped_r += out["diff_raw"] is None
        pairs.append(out)
    return pairs, dropped_w, dropped_r


@torch.no_grad()
def run_model(key, spec, res):
    from transformers import AutoTokenizer
    meta = np.load(next(c / spec["meta"] for c in CACHE_DIRS if (c / spec["meta"]).exists()), allow_pickle=True)
    rows, texts = meta["eval"], meta["eval_texts"]
    L = spec["L"]
    nm_of = lambda m: f'{spec["cache"]}_L{m + 1:02d}'             # only the three layers used need to be local
    dtype = getattr(torch, cache_meta(nm_of(round(L / 2))).get("dtype", "float16"))
    tok = AutoTokenizer.from_pretrained(spec["model"])
    model = load_causal_lm(spec["model"], dtype, "cuda")
    layers = decoder_layers(model, spec["L"])
    state = {"m": None, "vecs": None, "p": None}

    def hook(mod, inp, out):
        if state["vecs"] is None or mod is not layers[state["m"]]:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        for b, v in enumerate(state["vecs"]):
            if v is not None:
                h[b, state["p"]] = v.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    hooks = [l.register_forward_hook(hook) for l in layers]
    for m in (round(L / 4), round(L / 2), round(3 * L / 4)):
        tag = f"{key}_m{m}"
        if tag in res:
            continue
        nm = nm_of(m)
        Xf, Xe = load_cache(nm, "fit"), load_cache(nm, "eval")
        mu, W = fit_frame(Xf)
        Zf, Ze = (Xf - mu) @ W, (Xe - mu) @ W
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
            vecs = [None if v is None else torch.from_numpy(Xe[v]).cuda() for _, v in variants]
            state["vecs"], state["p"] = vecs, p
            batch = ids.repeat(len(variants), 1).cuda()
            out = model(input_ids=batch, use_cache=False, output_hidden_states=(i < 20))
            lp = torch.log_softmax(out.logits[:, p].float(), -1)
            k_ = (lp[0].exp() * (lp[0] - lp)).sum(-1).cpu().numpy()
            for j, (k, _) in enumerate(variants[1:], start=1):
                kl[k].append((i, float(k_[j])))
            if i < 20:            # the hook output is the cached layer: unpatched state vs cache
                h = out.hidden_states[m + 1][0, p].float().cpu().numpy()
                cos_check.append(float(h @ Xe[r] / (np.linalg.norm(h) * np.linalg.norm(Xe[r]) + 1e-12)))
            state["vecs"] = None
        res[tag] = {"layer_cache_index": m + 1, "n_pairs": len(pairs), "no_partner_whiten": dw, "no_partner_raw": dr,
                    "cos_unpatched_vs_cache": cos_check, "pairs": pairs, "kl": kl}
        res[tag]["score"] = score(res[tag])
        OUT.write_text(json.dumps(res))
        log(tag, json.dumps(res[tag]["score"]))
    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()


def boot(x, n=10000, seed=0):
    x = np.asarray(x)
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    lo, hi = np.percentile(x[idx].mean(1), [2.5, 97.5])
    return {"mean": float(x.mean()), "ci95": [float(lo), float(hi)], "n": int(len(x)), "excludes_0": bool(lo > 0 or hi < 0)}


def score(t):
    same = dict(t["kl"]["same"])
    out = {"noise_floor": float(np.mean([v for _, v in t["kl"]["self"]])), "same_mean": float(np.mean(list(same.values())))}
    out["void"] = out["noise_floor"] > out["same_mean"]
    for k in ("diff_w", "diff_raw"):
        d = [v - same[i] for i, v in t["kl"][k]]
        out[k] = {"diff_mean": float(np.mean([v for _, v in t["kl"][k]])), "paired": boot(d)}
        out[k]["supported"] = bool(out[k]["paired"]["mean"] > 0 and out[k]["paired"]["excludes_0"])
    return out


def main():
    set_threads()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for key, spec in MODELS.items():
        run_model(key, spec, res)
    summary = {}
    for key in MODELS:
        tags = [t for t in res if t.startswith(key + "_m")]
        ok = [t for t in tags if not res[t]["score"]["void"] and res[t]["score"]["diff_w"]["supported"]]
        ok2 = [t for t in tags if not res[t]["score"]["void"] and res[t]["score"]["diff_raw"]["supported"]]
        summary[key] = {"W2b1_layers_supported": len(ok), "W2b2_layers_supported": len(ok2), "of": len(tags),
                        "void": [t for t in tags if res[t]["score"]["void"]]}
    W2b1 = all(v["W2b1_layers_supported"] >= 2 for v in summary.values())
    killed = all(v["W2b1_layers_supported"] < 2 for v in summary.values())
    res["summary"] = {"per_model": summary, "W2b1_supported": W2b1, "killed": killed}
    OUT.write_text(json.dumps(res))
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
