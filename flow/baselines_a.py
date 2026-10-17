"""research/PREREG_baselines.md phase A: intrinsic dimension, dataset entropy, few-label selection, and per-document
p@10 for rule S / last / oracle (for bootstrap CIs). One pair per call; results/<pair>__baselines_a.json.

python baselines_a.py --model qwen35_4b --primary l2 --secondary l3
"""
import argparse

import faiss
import numpy as np

from common import RESULTS, load_cache, save_result, set_threads
from lens import layers_of, log

import json

MS = [1, 2, 5, 10]
DRAWS = 20


def unit(X, mu):
    A = X - mu
    return np.ascontiguousarray(A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12), dtype=np.float32)


def twonn(X, seed=0, n=5000, discard=0.1):
    """Facco et al. 2017: ratio of 2nd to 1st neighbour distance; linear fit through the origin of
    -log(1 - F(mu)) against log(mu), after discarding the largest 10% of mu."""
    rng = np.random.default_rng(seed)
    X = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    X = np.unique(X - X.mean(0), axis=0).astype(np.float32)
    idx = faiss.IndexFlatL2(X.shape[1])
    idx.add(np.ascontiguousarray(X))
    D, _ = idx.search(np.ascontiguousarray(X), 3)
    r1, r2 = np.sqrt(np.maximum(D[:, 1], 0)), np.sqrt(np.maximum(D[:, 2], 0))
    ok = r1 > 0
    mu = np.sort(r2[ok] / r1[ok])
    keep = int(len(mu) * (1 - discard))
    F = np.arange(1, len(mu) + 1) / len(mu)
    x, y = np.log(mu[:keep]), -np.log(1 - F[:keep])
    return float((x @ y) / (x @ x))


def dataset_entropy(X, seed=0, n=2000):
    """Matrix-based α=1 entropy (Giraldo et al. 2014) of the trace-normalized Gram of centered, L2-normalized docs."""
    rng = np.random.default_rng(seed)
    Z = unit(X[rng.choice(len(X), min(n, len(X)), replace=False)], X.mean(0)).astype(np.float64)
    lam = np.linalg.eigvalsh(Z @ Z.T)
    lam = np.clip(lam, 0, None)
    lam = lam / lam.sum()
    lam = lam[lam > 0]
    return float(-(lam * np.log(lam)).sum())


def hits(A, ya, B, yb, k=10, drop_self=None):
    """Per-query share of the k nearest database items (cosine) sharing the query's label."""
    idx = faiss.IndexFlatIP(A.shape[1])
    idx.add(A)
    _, I = idx.search(B, k + (1 if drop_self is not None else 0))
    if drop_self is not None:        # queries are train rows: drop each query's own index
        I = np.array([[j for j in row if j != q][:k] for row, q in zip(I, drop_self)])
    return (ya[I] == yb[:, None]).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--secondary", required=True)
    a = ap.parse_args()
    set_threads()
    sel = json.loads((RESULTS / f"{a.model}__select.json").read_text())
    s1 = sel["selection"][a.primary]
    S, last, orc = s1["S"]["layer"], len(sel["rows"]) - 1, s1["oracle_layer"]
    p10 = np.array([r[f"p@10_{a.primary}"] for r in sel["rows"]])
    names = layers_of(a.model)
    assert len(names) == len(sel["rows"])
    ytr = load_cache(names[0], "train", a.primary)[1]
    yte = load_cache(names[0], "test", a.primary)[1]
    classes = np.unique(ytr)
    draws = {m: [np.concatenate([np.random.default_rng(s).choice(np.flatnonzero(ytr == c), min(m, (ytr == c).sum()),
                                                                 replace=False) for c in classes])
                 for s in range(DRAWS)] for m in MS}
    rows, perdoc = [], {}
    for li, name in enumerate(names):
        Xtr, Xte = load_cache(name, "train"), load_cache(name, "test")
        mu = Xtr.mean(0)
        A = unit(Xtr, mu)
        r = {"layer": li, "twonn": twonn(Xte), "dataset_entropy": dataset_entropy(Xte)}
        for m in MS:
            r[f"fewlabel_m{m}"] = [float(hits(A, ytr, A[q], ytr[q], drop_self=q).mean()) for q in draws[m]]
        if li in (S, last, orc):
            perdoc[li] = hits(A, ytr, unit(Xte, mu), yte)
        rows.append(r)
        log(f"{a.model} L{li:02d} id={r['twonn']:.1f} H={r['dataset_entropy']:.3f} "
            f"few1={np.mean(r['fewlabel_m1']):.3f} few10={np.mean(r['fewlabel_m10']):.3f}")
    # sanity: per-document p@10 at the S layer reproduces the frozen select.json value. 20 Newsgroups has duplicate
    # and empty posts, so nearest-neighbour ties can resolve differently (seen: up to 3e-4); both values are kept.
    assert abs(perdoc[S].mean() - p10[S]) < 1e-3, (perdoc[S].mean(), p10[S])
    few = {}
    for m in MS:
        picks = [int(np.argmax([rows[l][f"fewlabel_m{m}"][d] for l in range(len(rows))])) for d in range(DRAWS)]
        few[m] = {"layers": picks, "regrets": [float(p10.max() - p10[l]) for l in picks]}
    save_result(a.model, "baselines_a", {
        "args": vars(a), "rows": rows, "S": S, "last": last, "oracle": orc, "p10": p10.tolist(), "fewlabel": few,
        "perdoc_check": {"S_perdoc_mean": float(perdoc[S].mean()), "S_select_json": float(p10[S])},
        "perdoc_hits": {str(k): v.astype(np.float32).tolist() for k, v in perdoc.items()}})


if __name__ == "__main__":
    main()
