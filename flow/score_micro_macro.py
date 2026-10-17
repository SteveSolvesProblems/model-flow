"""Micro to macro (exploratory): do weight-level signals (weight_structure.py) line up with residual geometry blocks
(geometry_probe.py)? Index convention: weight layer k makes the change from cached state k to k+1, and its writes
land in cached state k+1.
  boundary   Spearman across layers between the geometric change made by layer k (1 - centered cosine between cached
             states k and k+1) and weight-level change at layer k (1 - similarity between layers k-1 and k), or a
             per-layer weight statistic
  block      Spearman between off-diagonal entries of the geometry matrix over layer outputs (cached 1..L) and a weight
             similarity matrix; p from a null that shuffles values within each diagonal (keeps the decay with layer
             distance, breaks block structure), 2,000 draws
Writes results/micro_macro.json.

python score_micro_macro.py
"""
import json

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS

PAIRS = [("gemma4_e2b", ["gemma4_e2b", "tok_base"]), ("qwen3_17b", ["qwen3_17b", "tok_qwen3"]),
         ("modernbert_large", ["mbert_large"])]
WMATS = ["write_overlap", "read_overlap", "read_overlap_nogain", "gain_cos", "compose_excess"]


def diag_shuffle(M, rng):
    L, out = len(M), M.copy()
    for o in range(1, L):
        i = np.arange(L - o)
        v = M[i, i + o][rng.permutation(L - o)]
        out[i, i + o] = v
        out[i + o, i] = v
    return out


def offdiag(M):
    iu = np.triu_indices(len(M), 1)
    return M[iu]


def main():
    rng = np.random.default_rng(0)
    out = {}
    for wname, gnames in PAIRS:
        w = json.loads((RESULTS / f"weight_structure_{wname}.json").read_text())
        L = w["n_layers"]
        W = {k: np.array(w[k]) for k in ("write_overlap", "read_overlap", "read_overlap_nogain", "gain_cos")}
        C, C0 = np.array(w["compose"]), np.array(w["compose_null"])
        W["compose_excess"] = (C - C0 + (C - C0).T) / 2                  # symmetrised, null-subtracted composition
        per = w["per_layer"]
        stat = {"write_norm": [p["write_norm"] for p in per], "write_pr": [p["write_pr"] for p in per],
                "attn_share": [p["attn_write_norm"] / (p["attn_write_norm"] + p["mlp_write_norm"]) for p in per]}
        if w.get("layer_scalar"):
            stat["layer_scalar"] = w["layer_scalar"]
        if w.get("layer_types"):
            stat["full_attention"] = [float(t == "full_attention") for t in w["layer_types"]]
        elif w.get("global_attn_every_n_layers"):
            stat["full_attention"] = [float(k % w["global_attn_every_n_layers"] == 0) for k in range(L)]
        for g in gnames:
            p = json.loads((RESULTS / f"geometry_probe_{g}.json").read_text())
            assert len(p["layers"]) == L + 1, (g, len(p["layers"]), L)
            Cc, T = np.array(p["cos_centered"]), np.array(p["self_top1"])
            geo_change = np.array([1 - Cc[k, k + 1] for k in range(L)])        # change made by layer k
            r = {"boundary": {}, "block": {}}
            ks = np.arange(1, L)
            for m in WMATS:
                wc = np.array([1 - W[m][k - 1, k] for k in ks])
                r["boundary"][f"change_{m}"] = float(spearmanr(geo_change[ks], wc).statistic)
            for s, v in stat.items():
                r["boundary"][s] = float(spearmanr(geo_change, v).statistic)
            for gm, G in (("self_top1", T[1:, 1:]), ("cos_centered", Cc[1:, 1:])):
                for m in WMATS:
                    obs = spearmanr(offdiag(G), offdiag(W[m])).statistic
                    null = np.array([spearmanr(offdiag(G), offdiag(diag_shuffle(W[m], rng))).statistic
                                     for _ in range(2000)])
                    r["block"][f"{gm}~{m}"] = {"rho": float(obs), "null_mean": float(null.mean()),
                                               "p": float((1 + (null >= obs).sum()) / (1 + len(null)))}
            out[g] = r
            print(f"== {g} (weights {wname})")
            print("  boundary:", {k: round(v, 2) for k, v in r["boundary"].items()})
            print("  block:", {k: (round(v["rho"], 2), round(v["null_mean"], 2), round(v["p"], 3)) for k, v in r["block"].items()})
    (RESULTS / "micro_macro.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
