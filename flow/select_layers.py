"""Pre-registered test (research/PREREG_layer_selection.md): label-free layer selection.

Per layer: rule-S stability (cross-seed NMI of K=512 partitions in the whiten-64 frame), RankMe,
cosine kNN precision@10 (labels: evaluation only), and K=32 cross-seed churn for the architecture test.

python select_layers.py --model qwen35_08b_20ng --primary fine --secondary coarse
"""
import argparse

import faiss
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi

from common import cache_meta, load_cache, save_result, set_threads
from lens import frame, layers_of, log
from partition import assign, kmeans

K_RULE = 512
K_CHURN = 32


def rankme(X, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    X = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    p = s / s.sum() + 1e-12
    return float(np.exp(-(p * np.log(p)).sum()))


def p_at_k(Xtr, ytr, Xte, yte, k=10):
    mu = Xtr.mean(0)
    A = Xtr - mu
    B = Xte - mu
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    idx = faiss.IndexFlatIP(A.shape[1])
    idx.add(np.ascontiguousarray(A, dtype=np.float32))
    _, I = idx.search(np.ascontiguousarray(B, dtype=np.float32), k)
    return {lv: float((ytr[lv][I] == yte[lv][:, None]).mean()) for lv in ytr}


def churn_bootstrap(c0, c1, types, n_boot=200, seed=0):
    """Normalized cross-seed adjacent churn by block type, with a document bootstrap of (linear − full)."""
    L, N = c0.shape
    rng = np.random.default_rng(seed)

    def stats(idx):
        ceil = np.array([nmi(c0[i, idx], c1[i, idx]) for i in range(L)])
        ch = {}
        for i in range(1, L):
            a = 0.5 * (nmi(c0[i - 1, idx], c1[i, idx]) + nmi(c1[i - 1, idx], c0[i, idx]))
            ch.setdefault(types[i], []).append(1 - a / np.sqrt(ceil[i - 1] * ceil[i]))
        return {t: float(np.mean(v)) for t, v in ch.items()}

    point = stats(np.arange(N))
    out = {"point": point}
    if {"linear_attention", "full_attention"} <= set(point):
        diffs = []
        for _ in range(n_boot):
            s = stats(rng.integers(0, N, N))
            diffs.append(s["linear_attention"] - s["full_attention"])
        out["linear_minus_full"] = {"point": point["linear_attention"] - point["full_attention"],
                                    "ci95": np.quantile(diffs, [0.025, 0.975]).tolist()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--secondary", required=True)
    a = ap.parse_args()
    set_threads()
    names = layers_of(a.model)
    levels = [a.primary, a.secondary]
    ytr = {lv: load_cache(names[0], "train", lv)[1] for lv in levels}
    yte = {lv: load_cache(names[0], "test", lv)[1] for lv in levels}
    types = [cache_meta(n).get("layer_type") for n in names]
    rows = []
    churn_codes = [[], []]
    for li, name in enumerate(names):
        Xtr, Xte = load_cache(name, "train"), load_cache(name, "test")
        Ftr, Fte = frame(Xtr, Xte, preproc="whiten")
        c = [assign(Fte, kmeans(Ftr, K_RULE, seed=s, nredo=1))[0] for s in (0, 1)]
        k32 = [assign(Fte, kmeans(Ftr, K_CHURN, seed=s, nredo=1))[0] for s in (0, 1)]
        churn_codes[0].append(k32[0])
        churn_codes[1].append(k32[1])
        r = {"layer": li, "layer_type": types[li], "stability_K512": float(nmi(c[0], c[1])),
             "rankme": rankme(Xtr), **{f"p@10_{lv}": v for lv, v in p_at_k(Xtr, ytr, Xte, yte).items()}}
        rows.append(r)
        log(f"{a.model} L{li:02d} stab={r['stability_K512']:.4f} rankme={r['rankme']:.1f} "
            f"p@10 {a.primary}={r[f'p@10_{a.primary}']:.3f}")
    L = len(rows)
    stab = np.array([r["stability_K512"] for r in rows])
    rk = np.array([r["rankme"] for r in rows])
    sel = {"S": int(np.flatnonzero(stab == stab.max()).max()), "last": L - 1, "middle": L // 2,
           "rankme_max": int(rk.argmax()), "rankme_min": int(rk.argmin())}
    res = {}
    for lv in levels:
        p = np.array([r[f"p@10_{lv}"] for r in rows])
        res[lv] = {"oracle_layer": int(p.argmax()), "oracle": float(p.max()),
                   **{k: {"layer": v, "p@10": float(p[v]), "regret": float(p.max() - p[v])} for k, v in sel.items()}}
    churn = churn_bootstrap(np.array(churn_codes[0]), np.array(churn_codes[1]), types)
    save_result(a.model, "select", {"args": vars(a), "rows": rows, "selection": res, "churn": churn})
    log("selection", {lv: {k: (v["layer"], round(v["regret"], 4)) for k, v in res[lv].items() if isinstance(v, dict)}
                      for lv in levels})
    log("churn", churn)


if __name__ == "__main__":
    main()
