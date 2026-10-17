"""Follow-up to geometry_probe.py (exploratory, not pre-registered): is the layer-to-layer change of coordinates a
linear change that can be undone? For every layer pair (i, j), fit on the fit split and score on the same 3,000
held-out items as geometry_probe.py (each layer centered by its fit-split mean, rows unit length):
  identity     no map (reproduces geometry_probe's self_top1)
  rotation     orthogonal Procrustes map (rotation/reflection only, keeps all distances)
  linear       ridge regression map (any linear map)
For each: self_top1 (the mapped item's nearest neighbour at layer j is itself, among the 3,000; chance 1/3000) and,
for the linear map, held-out R^2. If the linear map recovers the items where identity fails, the information is all
still there in a linearly re-expressed form; if it does not, the layers in between rewrote it.
Writes results/geometry_linmap_<name>.json.

python geometry_linmap.py --name tok_base --fit fit --eval eval
"""
import argparse
import json
import time

import numpy as np

from common import RESULTS, sample_idx, cache_dir, set_threads
from lens import layers_of, log


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def rows(name, split, idx):
    return np.load(cache_dir(name) / f"{split}_X.npy", mmap_mode="r")[idx].astype(np.float32)


def top1(P, T):
    return float(((unit(P) @ T.T).argmax(1) == np.arange(len(P))).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--fit", default="train")
    ap.add_argument("--eval", default="test")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--n-fit", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0, help="eval sample seed (0 = geometry_probe's sample); fit sample uses seed+2")
    ap.add_argument("--prep", default="center", choices=["center", "robust"],
                    help="robust: per-dimension median/MAD scaling (fit split) before centering, as geometry_probe --prep robust")
    ap.add_argument("--lam", type=float, default=1e-2, help="ridge strength, relative to the mean eigenvalue of X'X")
    a = ap.parse_args()
    set_threads()
    names = layers_of(a.name)
    ne = np.load(cache_dir(names[0]) / f"{a.eval}_X.npy", mmap_mode="r").shape[0]
    nf = np.load(cache_dir(names[0]) / f"{a.fit}_X.npy", mmap_mode="r").shape[0]
    eidx = sample_idx(ne, a.n, a.seed)   # same items as geometry_probe
    fidx = np.sort(np.random.default_rng(a.seed + 2).choice(nf, min(a.n_fit, nf), replace=False))
    F, E = [], []
    for nm in names:
        Xf, Xe = rows(nm, a.fit, fidx), rows(nm, a.eval, eidx)
        if a.prep == "robust":
            med = np.median(Xf, 0)
            mad = np.median(np.abs(Xf - med), 0) + 1e-6
            Xf, Xe = (Xf - med) / mad, (Xe - med) / mad
        mu = Xf.mean(0)
        F.append(unit(Xf - mu))
        E.append(unit(Xe - mu))
    L, d = len(names), F[0].shape[1]
    ident, rot, lin, r2 = (np.zeros((L, L)) for _ in range(4))
    t0 = time.time()
    for i in range(L):
        G = F[i].T @ F[i]
        G[np.diag_indices(d)] += a.lam * np.trace(G) / d
        Gi = np.linalg.inv(G)
        for j in range(L):
            ident[i, j] = top1(E[i], E[j])
            C = F[i].T @ F[j]
            P = E[i] @ (Gi @ C)
            lin[i, j] = top1(P, E[j])
            r2[i, j] = 1 - ((E[j] - P) ** 2).sum() / ((E[j] - E[j].mean(0)) ** 2).sum()
            if j > i:                                         # Procrustes for (j, i) is the transpose of (i, j)
                U, _, Vt = np.linalg.svd(C)
                R = U @ Vt
                rot[i, j] = top1(E[i] @ R, E[j])
                rot[j, i] = top1(E[j] @ R.T, E[i])
            elif j == i:
                rot[i, j] = ident[i, j]
        k = min(i + 9, L - 1)
        log(a.name, i, f"+{k - i} layers: identity {ident[i, k]:.3f} rotation {rot[i, k]:.3f} linear {lin[i, k]:.3f}"
            f" R2 {r2[i, k]:.2f}  ({time.time() - t0:.0f}s)")
    out = {"name": a.name, "fit": a.fit, "eval": a.eval, "n": int(len(eidx)), "n_fit": int(len(fidx)), "lam": a.lam, "prep": a.prep,
           "layers": names, "identity_top1": ident.tolist(), "rotation_top1": rot.tolist(),
           "linear_top1": lin.tolist(), "linear_r2": r2.tolist(), "chance_top1": 1.0 / len(eidx)}
    (RESULTS / f"geometry_linmap_{a.name}{'' if a.prep == 'center' else '_robust'}{'' if a.seed == 0 else f'_s{a.seed}'}.json").write_text(json.dumps(out))
    log("saved")


if __name__ == "__main__":
    main()
