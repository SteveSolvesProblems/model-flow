"""research/PREREG_stability_v2.md: per-layer measures for one model-collection pair.

Per layer 0..L: S10 stability (10 seed pairs x 20,000-row train subsamples, as robustness.py b1), TwoNN on 5,000 test
vectors (baselines_a.twonn), p@10 (T1) and linear-probe accuracy (T2) on the pair's label, and one-label-per-class
selection hits (20 draws, as baselines_a). Writes results/<pair>__stabv2.json after every layer, and resumes from it.

python stab_v2.py --model llama32_1b --label l2
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from baselines_a import hits, twonn, unit
from common import RESULTS, cache_meta, load_cache, set_threads
from lens import frame, layers_of, log
from partition import assign, kmeans
from select_layers import p_at_k
from token_lens import fast_nmi

NPAIRS, SUB, K, DRAWS, PROBE_N, PROBE_PCS = 10, 20000, 512, 20, 12000, 256


def stability(Xtr, Xte):
    out = []
    for j in range(NPAIRS):
        idx = np.random.default_rng(j).choice(len(Xtr), min(SUB, len(Xtr)), replace=False)
        Ftr, Fte = frame(Xtr[idx], Xte)
        c = [assign(Fte, kmeans(Ftr, K, seed=s, nredo=1))[0] for s in (2 * j, 2 * j + 1)]
        out.append(float(fast_nmi(c[0], c[1])))
    return out


def probe(Xtr, ytr, Xte, yte):
    """Multinomial L2 logistic regression (C = 1) on the top 256 PCs of standardized features. PCA, scaling and the
    probe are all fit on the same <= 12,000 train rows (seed 0); scored on the full test split."""
    idx = np.random.default_rng(0).choice(len(Xtr), min(PROBE_N, len(Xtr)), replace=False)
    A = Xtr[idx].astype(np.float64)
    mu, sd = A.mean(0), A.std(0) + 1e-6
    A = (A - mu) / sd
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    W = Vt[:PROBE_PCS].T
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(A @ W, ytr[idx])
    return float((clf.predict(((Xte - mu) / sd) @ W) == yte).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    set_threads()
    names = layers_of(a.model)
    assert len(names) == cache_meta(names[0])["n_layers"] + 1, names
    out_path = RESULTS / f"{a.model}__stabv2.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {"model": a.model, "label": a.label, "rows": []}
    assert res["label"] == a.label
    ytr = load_cache(names[0], "train", a.label)[1]
    yte = load_cache(names[0], "test", a.label)[1]
    classes = np.unique(ytr)
    draws = [np.concatenate([np.random.default_rng(s).choice(np.flatnonzero(ytr == c), 1, replace=False)
                             for c in classes]) for s in range(DRAWS)]
    for li in range(len(res["rows"]), len(names)):
        Xtr, Xte = load_cache(names[li], "train"), load_cache(names[li], "test")
        A = unit(Xtr, Xtr.mean(0))
        r = {"layer": li, "stability": stability(Xtr, Xte), "twonn": twonn(Xte),
             "p10": p_at_k(Xtr, {a.label: ytr}, Xte, {a.label: yte})[a.label],
             "probe": probe(Xtr, ytr, Xte, yte),
             "fewlabel_m1": [float(hits(A, ytr, A[q], ytr[q], drop_self=q).mean()) for q in draws]}
        res["rows"].append(r)
        out_path.write_text(json.dumps(res))
        log(a.model, f"L{li:02d}", f"S={np.mean(r['stability']):.4f} id={r['twonn']:.1f} p10={r['p10']:.3f} "
            f"probe={r['probe']:.3f}")
    res["cache_meta"] = cache_meta(names[0])
    out_path.write_text(json.dumps(res, indent=1))
    log(a.model, "done")


if __name__ == "__main__":
    main()
