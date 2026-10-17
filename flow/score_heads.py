"""Head behaviour vs geometry blocks (exploratory). Inputs: attention_heads_<w>.json, geometry_probe_<g>.json,
weight_structure_<w>.json. Decoder layer k writes cached state k+1.
  blocks    contiguous segmentation of the layer outputs (cached 1..L) maximizing the summed within-block
            (self-match - mean off-diagonal self-match): no block count fixed in advance
  profile   per block: mean of each head statistic, and counts of head types (thresholds fixed before looking:
            sink head sink > 0.5; previous-token head prev > 0.4; induction head induction > 0.3; local head
            dist < 0.15; diffuse head entropy > 0.8)
  block     Spearman between the geometry matrix and a layer-by-layer similarity of head-statistic profiles
            (negative Euclidean distance between layers' sorted, standardized head-statistic vectors), against the
            within-diagonal shuffle null (2,000 draws)
  boundary  Spearman across layers between the geometric change made by layer k and the change in head profile
            between layers k-1 and k
Writes results/heads_vs_blocks.json.
"""
import json

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS
from score_micro_macro import diag_shuffle, offdiag

RUNS = [("gemma4_e2b", ["gemma4_e2b", "tok_base"]), ("qwen3_17b", ["qwen3_17b", "tok_qwen3"]),
        ("modernbert_large", ["mbert_large"])]
STATS = ["entropy", "sink", "prev", "self", "dist", "induction"]
TYPES = {"sink": ("sink", ">", 0.5), "prev_token": ("prev", ">", 0.4), "induction": ("induction", ">", 0.3),
         "local": ("dist", "<", 0.15), "diffuse": ("entropy", ">", 0.8)}


def segment(S):
    """Contiguous blocks maximizing sum over blocks of sum_{i,j in block, i != j} (S_ij - m)."""
    L = len(S)
    m = offdiag(S).mean()
    D = S - m
    np.fill_diagonal(D, 0)
    P = D.cumsum(0).cumsum(1)
    blk = lambda a, b: P[b, b] - (P[a - 1, b] if a else 0) - (P[b, a - 1] if a else 0) + (P[a - 1, a - 1] if a else 0)
    best, prev = np.full(L + 1, -np.inf), np.zeros(L + 1, int)
    best[0] = 0
    for b in range(1, L + 1):
        for a in range(b):
            v = best[a] + blk(a, b - 1)
            if v > best[b]:
                best[b], prev[b] = v, a
    cuts, b = [], L
    while b > 0:
        cuts.append((prev[b], b - 1))
        b = prev[b]
    return cuts[::-1]


def main():
    rng = np.random.default_rng(0)
    out = {}
    for wname, gnames in RUNS:
        h = json.loads((RESULTS / f"attention_heads_{wname}.json").read_text())
        L = h["n_layers"]
        X = {s: np.array([lay[s] for lay in h["layers"]]) for s in STATS}          # L x H
        F = np.concatenate([np.sort((X[s] - X[s].mean()) / (X[s].std() + 1e-9), 1) for s in STATS], 1)
        Dh = -np.sqrt(((F[:, None, :] - F[None, :, :]) ** 2).sum(-1))              # layer similarity of head profiles
        types = {t: (X[s] > v) if op == ">" else (X[s] < v) for t, (s, op, v) in TYPES.items()}
        for g in gnames:
            p = json.loads((RESULTS / f"geometry_probe_{g}.json").read_text())
            T, C = np.array(p["self_top1"])[1:, 1:], np.array(p["cos_centered"])
            blocks = segment(T)                                                      # indices = decoder layers
            prof = []
            for a, b in blocks:
                r = {"layers": [int(a), int(b)]}
                r.update({s: round(float(X[s][a:b + 1].mean()), 3) for s in STATS})
                r.update({f"n_{t}": int(m[a:b + 1].sum()) for t, m in types.items()})
                r["n_heads"] = int((b - a + 1) * X["entropy"].shape[1])
                prof.append(r)
            obs = spearmanr(offdiag(T), offdiag(Dh)).statistic
            null = np.array([spearmanr(offdiag(T), offdiag(diag_shuffle(Dh, rng))).statistic for _ in range(2000)])
            change = np.array([1 - C[k, k + 1] for k in range(1, L)])
            dprof = np.array([-Dh[k - 1, k] for k in range(1, L)])
            bnd = {s: float(spearmanr(change, np.abs(np.diff(X[s].mean(1)))).statistic) for s in STATS}
            bnd["profile"] = float(spearmanr(change, dprof).statistic)
            out[g] = {"blocks": prof, "block_test": {"rho": float(obs), "null_mean": float(null.mean()),
                                                     "p": float((1 + (null >= obs).sum()) / (1 + len(null)))},
                      "boundary": bnd}
            print(f"== {g}: blocks {[tuple(r['layers']) for r in prof]}")
            for r in prof:
                print("   ", r)
            print("   block test", out[g]["block_test"], "| boundary", {k: round(v, 2) for k, v in bnd.items()})
    (RESULTS / "heads_vs_blocks.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
