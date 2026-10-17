"""research/PREREG_overnight.md E3: cross-model layer correspondence (partition NMI vs linear CKA) and stage boundaries
(dips in adjacent-layer agreement), on the same DBpedia test documents. Needs results/codes/<model>.npz.
Writes results/correspondence.json (+ matrices in results/correspondence.npz)."""
import itertools
import json

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS, load_cache, set_threads
from lens import layers_of, log
from token_lens import fast_nmi

MODELS = ["qwen3_17b", "qwen35_08b", "qwen35_2b", "qwen35_4b", "gemma4_e2b", "gemma4_e4b", "smollm2_17b"]
KV_LAST_OWN = {"gemma4_e2b": 15, "gemma4_e4b": 24}   # lens layer = last decoder layer with its own KV + 1 (configs)
N_CKA = 1000


def adj_nmi(a, b, rng, n=3):
    return fast_nmi(a, b) - np.mean([fast_nmi(rng.permutation(a), b) for _ in range(n)])


def gram_vec(X):
    """Centered linear Gram, flattened and unit-normalized: CKA(X, Y) = <gx, gy>."""
    X = X - X.mean(0)
    K = X @ X.T
    H = K - K.mean(0) - K.mean(1)[:, None] + K.mean()
    v = H.ravel()
    return (v / np.linalg.norm(v)).astype(np.float32)


def local_minima(a):
    """Indices ℓ (into the per-layer array, NaN at ℓ=0) where a(ℓ) < both neighbours."""
    return [i for i in range(2, len(a) - 1) if a[i] < a[i - 1] and a[i] < a[i + 1]]


def main():
    set_threads()
    rng = np.random.default_rng(0)
    codes = {m: np.load(RESULTS / "codes" / f"{m}.npz") for m in MODELS}
    n = codes[MODELS[0]]["c0"].shape[1]
    sub = np.random.default_rng(0).choice(n, N_CKA, replace=False)
    grams = {}
    for m in MODELS:
        grams[m] = np.stack([gram_vec(load_cache(nm, "test")[sub].astype(np.float64)) for nm in layers_of(m)])
        log("gram", m, grams[m].shape)

    out, mats = {"models": MODELS, "pairs": {}, "boundaries": {}}, {}
    for m in MODELS:
        c0, c1 = codes[m]["c0"], codes[m]["c1"]
        L = len(c0)
        ceil = np.array([fast_nmi(c0[i], c1[i]) for i in range(L)])
        a = np.full(L, np.nan)
        a_adj = np.full(L, np.nan)
        for i in range(1, L):
            a[i] = fast_nmi(c0[i - 1], c0[i]) / np.sqrt(ceil[i - 1] * ceil[i])
            a_adj[i] = adj_nmi(c0[i - 1], c0[i], rng)
        mins = local_minima(a)
        mins_adj = local_minima(a_adj)
        out["boundaries"][m] = {"adjacent": a.tolist(), "adjacent_adjusted": a_adj.tolist(), "ceiling": ceil.tolist(),
                                "local_minima": mins, "local_minima_adjusted": mins_adj}
        if m in KV_LAST_OWN:
            k = KV_LAST_OWN[m]
            out["boundaries"][m]["kv_layer"] = k
            out["boundaries"][m]["X3_hit"] = any(abs(x - k) <= 1 for x in mins)
            out["boundaries"][m]["X3_hit_adjusted"] = any(abs(x - k) <= 1 for x in mins_adj)
        log("boundaries", m, mins)

    x1 = x2 = 0
    for A, B in itertools.combinations(MODELS, 2):
        ca, cb = codes[A]["c0"], codes[B]["c0"]
        La, Lb = len(ca), len(cb)
        N = np.array([[fast_nmi(ca[i], cb[j]) for j in range(Lb)] for i in range(La)])
        Nadj = np.array([[adj_nmi(ca[i], cb[j], rng, n=1) for j in range(Lb)] for i in range(La)])
        C = grams[A] @ grams[B].T
        mats[f"{A}|{B}|nmi"], mats[f"{A}|{B}|nmi_adj"], mats[f"{A}|{B}|cka"] = N, Nadj, C
        da, db = np.arange(La) / (La - 1), np.arange(Lb) / (Lb - 1)
        r1 = spearmanr(da, db[N.argmax(1)]).statistic
        r1_rev = spearmanr(db, da[N.argmax(0)]).statistic
        r1_adj = spearmanr(da, db[Nadj.argmax(1)]).statistic
        r1_cka = spearmanr(da, db[C.argmax(1)]).statistic
        r2 = spearmanr(N.ravel(), C.ravel()).statistic
        x1 += r1 >= 0.9
        x2 += r2 >= 0.7
        out["pairs"][f"{A}|{B}"] = {"X1_spearman": float(r1), "X1_reverse": float(r1_rev), "X1_adjusted": float(r1_adj),
                                   "X1_cka": float(r1_cka), "X2_nmi_vs_cka": float(r2),
                                   "best_match_A_to_B": N.argmax(1).tolist(), "best_match_cka": C.argmax(1).tolist()}
        log(A, B, f"X1={r1:.3f} X2={r2:.3f}")
    out["X1_pairs_ge_0.9"] = int(x1)
    out["X1_supported"] = bool(x1 >= 18)
    out["X2_pairs_ge_0.7"] = int(x2)
    out["X2_supported"] = bool(x2 >= 18)
    out["X3_supported"] = all(out["boundaries"][m]["X3_hit"] for m in KV_LAST_OWN)
    (RESULTS / "correspondence.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(RESULTS / "correspondence.npz", **mats)
    log("X1", x1, "X2", x2, "X3", out["X3_supported"])


if __name__ == "__main__":
    main()
