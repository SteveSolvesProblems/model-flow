"""research/PREREG_large_k.md: the attractor lens at K up to 4096.

Per layer, per train size N (first N rows of the cache) and K:
  stability (cross-seed test NMI, rule S), cell-majority classifier accuracy, NMI with each label level,
  nestedness of the K partition in the K/2 partition (train members), cosine p@10.
At the rule-S_512 layer and the last layer: 10-NN majority vote and a linear probe for comparison.

python large_k.py --model qwen35_2b --levels l2 l3 --ks 512 1024 2048
python large_k.py --model qwen35_2b_big --levels l2 l3 --ks 512 1024 2048 4096 --ns 24000 120000
"""
import argparse
import json

import faiss
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score as nmi

from common import RESULTS, load_cache, save_result, set_threads
from lens import frame, layers_of, log
from partition import assign, kmeans


def cell_majority(ctr, ytr, K):
    """Majority train label per cell (-1 for empty cells)."""
    C = np.zeros((K, ytr.max() + 1), np.int64)
    np.add.at(C, (ctr, ytr), 1)
    lab = C.argmax(1)
    lab[C.sum(1) == 0] = -1
    return lab


def nested(fine, coarse, thr=0.9):
    """Share of fine cells whose members fall ≥ thr into one coarse cell."""
    Kf, Kc = fine.max() + 1, coarse.max() + 1
    C = np.zeros((Kf, Kc), np.int64)
    np.add.at(C, (fine, coarse), 1)
    n = C.sum(1)
    ok = n > 0
    return float(np.mean(C[ok].max(1) / n[ok] >= thr))


def knn(Xtr, Xte, k=10):
    mu = Xtr.mean(0)
    A, B = Xtr - mu, Xte - mu
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    idx = faiss.IndexFlatIP(A.shape[1])
    idx.add(np.ascontiguousarray(A, dtype=np.float32))
    return idx.search(np.ascontiguousarray(B, dtype=np.float32), k)[1]


def vote(I, ytr):
    lab = ytr[I]
    return np.array([np.bincount(r).argmax() for r in lab])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--levels", nargs="+", required=True, help="primary first")
    ap.add_argument("--ks", nargs="+", type=int, default=[512, 1024, 2048])
    ap.add_argument("--ns", nargs="+", type=int, default=[0], help="train sizes; 0 = all rows")
    ap.add_argument("--probe-n", type=int, default=20000)
    ap.add_argument("--slim", action="store_true",
                    help="set A (amended): stability only; p@10 and K=512 selection taken from <model>__select.json")
    a = ap.parse_args()
    if a.slim:
        return slim(a)
    set_threads()
    names = layers_of(a.model)
    L = len(names)
    Ytr_all = {lv: load_cache(names[0], "train", lv)[1] for lv in a.levels}
    yte = {lv: load_cache(names[0], "test", lv)[1] for lv in a.levels}
    ns = [n if n else len(Ytr_all[a.levels[0]]) for n in a.ns]
    rows = []
    for li, name in enumerate(names):
        Xall, Xte = load_cache(name, "train"), load_cache(name, "test")
        for N in ns:
            Xtr = Xall[:N]
            ytr = {lv: y[:N] for lv, y in Ytr_all.items()}
            Ftr, Fte = frame(Xtr, Xte, preproc="whiten")
            I = knn(Xtr, Xte)
            r = {"layer": li, "N": N, **{f"p@10_{lv}": float((ytr[lv][I] == yte[lv][:, None]).mean()) for lv in a.levels}}
            prev_tr = None
            for K in a.ks:
                C = [kmeans(Ftr, K, seed=s, nredo=1) for s in (0, 1)]
                te = [assign(Fte, c)[0] for c in C]
                tr0 = assign(Ftr, C[0])[0]
                r[f"stab_K{K}"] = float(nmi(te[0], te[1]))
                for lv in a.levels:
                    maj = cell_majority(tr0, ytr[lv], K)
                    r[f"cellacc_{lv}_K{K}"] = float(np.mean(maj[te[0]] == yte[lv]))
                    r[f"nmi_{lv}_K{K}"] = float(nmi(yte[lv], te[0]))
                if prev_tr is not None and K == 2 * prev_tr[0]:
                    r[f"nest_K{K}"] = nested(tr0, prev_tr[1])
                prev_tr = (K, tr0)
            rows.append(r)
            log(f"{a.model} L{li:02d} N={N} " + " ".join(f"stab{K}={r[f'stab_K{K}']:.3f}" for K in a.ks)
                + f" p@10={r[f'p@10_{a.levels[0]}']:.3f}")
    # rule S per (N, K), regret, and reference classifiers at S_512 and last layer
    sel, ref = {}, {}
    for N in ns:
        R = [r for r in rows if r["N"] == N]
        p = {lv: np.array([r[f"p@10_{lv}"] for r in R]) for lv in a.levels}
        for K in a.ks:
            st = np.array([r[f"stab_K{K}"] for r in R])
            S = int(np.flatnonzero(st == st.max()).max())
            sel[f"N{N}_K{K}"] = {"layer": S, **{f"regret_{lv}": float(p[lv].max() - p[lv][S]) for lv in a.levels}}
        sel[f"N{N}_oracle"] = {lv: int(p[lv].argmax()) for lv in a.levels}
        sel[f"N{N}_last"] = {f"regret_{lv}": float(p[lv].max() - p[lv][-1]) for lv in a.levels}
        k0 = 512 if 512 in a.ks else a.ks[0]
        for tag, li in (("S", sel[f"N{N}_K{k0}"]["layer"]), ("last", L - 1)):
            Xall, Xte = load_cache(names[li], "train"), load_cache(names[li], "test")
            Xtr = Xall[:N]
            Ftr, Fte = frame(Xtr, Xte, preproc="whiten")
            I = knn(Xtr, Xte)
            sub = np.random.default_rng(0).choice(N, min(a.probe_n, N), replace=False)
            d = {"layer": li}
            for lv in a.levels:
                y = Ytr_all[lv][:N]
                d[f"knn10_{lv}"] = float(np.mean(vote(I, y) == yte[lv]))
                clf = LogisticRegression(max_iter=2000).fit(Ftr[sub], y[sub])
                d[f"probe_{lv}"] = float(clf.score(Fte, yte[lv]))
                d.update({f"cellacc_{lv}_K{K}": next(r for r in rows if r["N"] == N and r["layer"] == li)[f"cellacc_{lv}_K{K}"]
                          for K in a.ks})
            ref[f"N{N}_{tag}"] = d
            log(f"N={N} {tag} L{li}", {k: round(v, 3) for k, v in d.items() if k != "layer"})
    log("selection", sel)
    save_result(a.model, "largek", {"args": vars(a), "rows": rows, "selection": sel, "classifiers": ref})


def slim(a):
    """PREREG_large_k.md Amendment 1: stability at the given K per layer; regret against the frozen select results."""
    set_threads()
    names = layers_of(a.model)
    ref = json.loads((RESULTS / f"{a.model}__select.json").read_text())
    rows = []
    for li, name in enumerate(names):
        Ftr, Fte = frame(load_cache(name, "train"), load_cache(name, "test"), preproc="whiten")
        r = {"layer": li, "stab_K512": ref["rows"][li]["stability_K512"]}
        for K in a.ks:
            te = [assign(Fte, kmeans(Ftr, K, seed=s, nredo=1))[0] for s in (0, 1)]
            r[f"stab_K{K}"] = float(nmi(te[0], te[1]))
        rows.append(r)
        log(f"{a.model} L{li:02d} " + " ".join(f"stab{K}={r[f'stab_K{K}']:.3f}" for K in [512] + a.ks))
    sel = {}
    for K in [512] + a.ks:
        st = np.array([r[f"stab_K{K}"] for r in rows])
        S = int(np.flatnonzero(st == st.max()).max())
        sel[f"K{K}"] = {"layer": S}
        for lv in a.levels:
            p = np.array([r[f"p@10_{lv}"] for r in ref["rows"]])
            sel[f"K{K}"][f"regret_{lv}"] = float(p.max() - p[S])
    log("selection", sel)
    save_result(a.model, "largek_slim", {"args": vars(a), "rows": rows, "selection": sel})


if __name__ == "__main__":
    main()
