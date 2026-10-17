"""Feasibility probe (not pre-registered): does a token's or a document's residual vector stay in a similar place
across layers? The residual stream shares one coordinate system across layers (each block adds to it), so raw
coordinates can be compared directly, unlike the per-layer bins.

For a sample of items, per layer pair (i, j):
  cos_raw      mean cosine between the same item's vectors at layers i and j
  cos_centered the same after subtracting each layer's mean vector (removes the shared offset / anisotropy)
  self_top1    share of items whose nearest neighbour at layer j (centered cosine, among all sampled items at j)
               is the item itself, queried from layer i: does the item keep its place relative to the others?
Writes results/geometry_probe_<name>.json.

python geometry_probe.py --name tok_base --split eval --n 3000
"""
import argparse
import json

import numpy as np

from common import RESULTS, sample_idx, load_cache, set_threads
from lens import layers_of, log


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0, help="sample seed (0 = the original sample)")
    ap.add_argument("--prep", default="center", choices=["center", "robust"],
                    help="center: subtract the layer mean; robust: per-dimension median / MAD scaling per layer")
    a = ap.parse_args()
    set_threads()
    names = layers_of(a.name)
    X0 = load_cache(names[0], a.split)
    idx = sample_idx(len(X0), a.n, a.seed)
    raw, cen, nrm = [], [], []
    for nm in names:
        X = load_cache(nm, a.split)[idx].astype(np.float32)
        raw.append(unit(X))
        if a.prep == "robust":
            med = np.median(X, 0)
            cen.append(unit((X - med) / (np.median(np.abs(X - med), 0) + 1e-6)))
        else:
            cen.append(unit(X - X.mean(0)))
        nrm.append(float(np.linalg.norm(X, axis=1).mean()))
    L = len(names)
    # relational view: each item's 10 nearest neighbours (centered cosine, within the sample) at every layer
    knn = [np.argsort(-(c @ c.T), axis=1)[:, 1:11] for c in cen]
    knn_overlap = np.zeros((L, L))
    for i in range(L):
        for j in range(i, L):
            ov = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(knn[i], knn[j])])
            knn_overlap[i, j] = knn_overlap[j, i] = ov
    cos_raw = np.zeros((L, L)); cos_cen = np.zeros((L, L)); top1 = np.zeros((L, L)); chance = np.zeros((L, L))
    perm = np.random.default_rng(1).permutation(len(idx))     # chance: an item vs a different random item
    for i in range(L):
        for j in range(L):
            cos_raw[i, j] = float((raw[i] * raw[j]).sum(1).mean())
            cos_cen[i, j] = float((cen[i] * cen[j]).sum(1).mean())
            chance[i, j] = float((cen[i] * cen[j][perm]).sum(1).mean())
            S = cen[i] @ cen[j].T
            top1[i, j] = float((S.argmax(1) == np.arange(len(idx))).mean())
        log(a.name, i, "next-layer cos raw", round(cos_raw[i, min(i + 1, L - 1)], 3), "centered",
            round(cos_cen[i, min(i + 1, L - 1)], 3), "self-top1 to last", round(top1[i, -1], 3))
    out = {"name": a.name, "split": a.split, "n": int(len(idx)), "layers": names, "mean_norm": nrm,
           "prep": a.prep, "seed": a.seed, "cos_raw": cos_raw.tolist(), "cos_centered": cos_cen.tolist(), "self_top1": top1.tolist(),
           "chance_cos": chance.tolist(), "chance_top1": 1.0 / len(idx),
           "knn10_overlap": knn_overlap.tolist(), "chance_knn10": 10.0 / (len(idx) - 1)}
    tag = ("" if a.prep == "center" else "_robust") + ("" if a.seed == 0 else f"_s{a.seed}") + ("" if a.n == 3000 else f"_n{a.n}")
    (RESULTS / f"geometry_probe_{a.name}{tag}.json").write_text(json.dumps(out))
    log("saved")


if __name__ == "__main__":
    main()
