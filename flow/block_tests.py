"""Stricter block tests (exploratory; replaces the null in score_micro_macro.py after the 2026-10-04 audit).

1. Do the geometry matrices have blocks at all? segment() always returns blocks, so test its gain against surrogates:
   gain = sum over blocks of within-block (S - offdiag mean), as maximized by segment(). Surrogates shuffle values
   within each diagonal (keeps the decay with layer distance) and, in a second null, within each diagonal after
   removing a smooth depth trend (keeps decay and depth trend). p = share of surrogates with gain >= observed.
   Also checks segment() on a block-free matrix built from the observed per-distance means (calibration).
2. Weights vs geometry (block test): partial Spearman of off-diagonal entries after regressing both matrices (ranks)
   on a smooth trend in layer distance d and depth s = i + j (d, d^2, s, s^2, d*s). Null: diagonal shuffle of the
   weight matrix's residuals, 10,000 draws. Calibrated with a pure trend matrix (i + j), which must not pass.
   Holm correction over all block tests run here.
Writes results/block_tests.json.
"""
import json

import numpy as np
from scipy.stats import rankdata

from common import RESULTS
from score_heads import segment
from score_micro_macro import PAIRS, WMATS, diag_shuffle, offdiag

N_NULL = 10000


def gain(S):
    m = offdiag(S).mean()
    return sum(((S[a:b + 1, a:b + 1] - m).sum() - np.trace(S[a:b + 1, a:b + 1] - m)) for a, b in segment(S))


def design(L):
    i, j = np.triu_indices(L, 1)
    d, s = (j - i).astype(float), (i + j).astype(float)
    d, s = (d - d.mean()) / d.std(), (s - s.mean()) / s.std()
    return np.column_stack([np.ones_like(d), d, d ** 2, s, s ** 2, d * s])


def resid(v, X):
    r = rankdata(v)
    return r - X @ np.linalg.lstsq(X, r, rcond=None)[0]


def to_mat(v, L):
    M = np.zeros((L, L))
    M[np.triu_indices(L, 1)] = v
    return M + M.T


def detrended_shuffle(S, X, rng):
    """Diagonal shuffle of the trend residuals, trend added back (keeps decay and depth trend, breaks blocks)."""
    L = len(S)
    v = offdiag(S)
    fit = X @ np.linalg.lstsq(X, v, rcond=None)[0]
    R = diag_shuffle(to_mat(v - fit, L), rng)
    out = to_mat(fit + offdiag(R), L)
    np.fill_diagonal(out, np.diag(S))
    return out


def has_blocks(S, rng, n=1000):
    L, X = len(S), design(len(S))
    g = gain(S)
    n1 = np.array([gain(diag_shuffle(S, rng)) for _ in range(n)])
    n2 = np.array([gain(detrended_shuffle(S, X, rng)) for _ in range(n)])
    toe = np.array([[np.mean(np.diag(S, abs(i - j))) for j in range(L)] for i in range(L)])
    return {"gain": float(g), "p_diag": float((1 + (n1 >= g).sum()) / (1 + n)),
            "p_trend": float((1 + (n2 >= g).sum()) / (1 + n)),
            "gain_null_median": float(np.median(n2)),
            "blocks": [[int(a), int(b)] for a, b in segment(S)],
            "blocks_on_blockfree": [[int(a), int(b)] for a, b in segment(toe)]}


def partial_test(G, W, rng):
    L = len(G)
    X = design(L)
    rg, rw = resid(offdiag(G), X), resid(offdiag(W), X)
    obs = float(np.corrcoef(rg, rw)[0, 1])
    Rw = to_mat(rw, L)
    null = np.array([np.corrcoef(rg, offdiag(diag_shuffle(Rw, rng)))[0, 1] for _ in range(N_NULL)])
    return {"rho_partial": obs, "p": float((1 + (null >= obs).sum()) / (1 + N_NULL))}


def holm(ps):
    keys = sorted(ps, key=ps.get)
    m, out, run = len(keys), {}, 0.0
    for r, k in enumerate(keys):
        run = max(run, min(1.0, (m - r) * ps[k]))
        out[k] = run
    return out


def main():
    rng = np.random.default_rng(0)
    out = {"geometry_blocks": {}, "weights_vs_geometry": {}, "calibration": {}}
    for wname, gnames in PAIRS:
        w = json.loads((RESULTS / f"weight_structure_{wname}.json").read_text())
        W = {k: np.array(w[k]) for k in ("write_overlap", "read_overlap", "read_overlap_nogain", "gain_cos")}
        C, C0 = np.array(w["compose"]), np.array(w["compose_null"])
        W["compose_excess"] = (C - C0 + (C - C0).T) / 2
        L = w["n_layers"]
        ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
        out["calibration"][wname] = partial_test(np.array(json.loads((RESULTS / f"geometry_probe_{gnames[0]}.json")
                                                 .read_text())["self_top1"])[1:, 1:], -(ii + jj).astype(float), rng)
        for g in gnames:
            p = json.loads((RESULTS / f"geometry_probe_{g}.json").read_text())
            mats = {"self_top1": np.array(p["self_top1"])[1:, 1:], "cos_centered": np.array(p["cos_centered"])[1:, 1:],
                    "knn10": np.array(p["knn10_overlap"])[1:, 1:]}
            out["geometry_blocks"][g] = {k: has_blocks(M, rng) for k, M in mats.items()}
            out["weights_vs_geometry"][g] = {f"{gm}~{m}": partial_test(G, W[m], rng)
                                             for gm, G in mats.items() for m in WMATS}
            print(g, {k: (v["p_diag"], v["p_trend"], v["blocks"]) for k, v in out["geometry_blocks"][g].items()})
            print("  ", {k: (round(v["rho_partial"], 2), v["p"]) for k, v in out["weights_vs_geometry"][g].items()})
    ps = {f"{g}|{k}": v["p"] for g, d in out["weights_vs_geometry"].items() for k, v in d.items()}
    ps.update({f"{g}|blocks|{k}": v["p_trend"] for g, d in out["geometry_blocks"].items() for k, v in d.items()})
    out["holm"] = holm(ps)
    out["n_tests"] = len(ps)
    print("calibration (pure depth trend, should not pass):", out["calibration"])
    print("survive Holm at 0.05:", [k for k, v in out["holm"].items() if v < 0.05])
    (RESULTS / "block_tests.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
