"""Compression and expansion per layer (exploratory, not pre-registered): fluid-style statistics of the residual
stream for the geometry_probe sample (3,000 held-out items). Per layer l (centered, unit-length rows unless noted):
  part_ratio     participation ratio of the covariance spectrum, (sum lambda)^2 / sum lambda^2 (effective dimension)
  twonn          TwoNN intrinsic dimension (Facco et al. 2017)
  spread         median pairwise distance
Per adjacent pair (l, l+1):
  local_log_ratio  mean over items of log(d_{l+1} / d_l) for each item's 10 nearest neighbours at l, minus the same for
                   10 random items: < 0 means neighbourhoods tighten relative to the whole cloud (local compression)
  contract_share   share of items whose neighbourhood tightens relative to the cloud
  update_rel       mean ||h_{l+1} - h_l|| / ||h_l|| on the raw residual (size of the layer's write)
  update_cos       mean cosine between the update and h_l (writing along the current vector vs across it)
  update_in_top    share of the centered update's energy inside the top PCs that hold 90% of h_l's variance
                   (writing into the subspace already in use vs into new directions)
Writes results/geometry_flow_<name>.json.

python geometry_flow.py --name tok_base --split eval
"""
import argparse
import json

import numpy as np

from common import RESULTS, cache_dir, set_threads
from lens import layers_of, log


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def twonn(D):
    s = np.sort(D, 1)[:, :2]
    s = s[s[:, 0] > 1e-6]                       # repeated tokens share one input embedding: drop exact duplicates
    mu = s[:, 1] / s[:, 0]
    mu = np.sort(mu[np.isfinite(mu) & (mu > 1)])[: int(0.9 * len(mu))]   # standard 10% tail discard
    return float(len(mu) / np.log(mu).sum())


def dist(U):
    return np.sqrt(np.maximum(2 - 2 * (U @ U.T), 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    set_threads()
    names = layers_of(a.name)
    n0 = np.load(cache_dir(names[0]) / f"{a.split}_X.npy", mmap_mode="r").shape[0]
    idx = np.sort(np.random.default_rng(0).choice(n0, min(a.n, n0), replace=False))
    raw = [np.load(cache_dir(nm) / f"{a.split}_X.npy", mmap_mode="r")[idx].astype(np.float32) for nm in names]
    U = [unit(X - X.mean(0)) for X in raw]
    rnd = np.random.default_rng(3).integers(0, len(idx), (len(idx), 10))
    out = {k: [] for k in ("part_ratio", "twonn", "spread", "local_log_ratio", "contract_share",
                           "update_rel", "update_cos", "update_in_top", "top_k90")}
    D = dist(U[0])
    for l in range(len(names)):
        ev = np.clip(np.linalg.eigvalsh(np.cov(U[l].T)), 0, None)[::-1]
        out["part_ratio"].append(float(ev.sum() ** 2 / (ev ** 2).sum()))
        out["twonn"].append(twonn(D + np.diag(np.full(len(D), np.inf))))
        out["spread"].append(float(np.median(D[np.triu_indices(len(D), 1)])))
        if l == len(names) - 1:
            break
        Dn = dist(U[l + 1])
        nb = np.argsort(D + np.diag(np.full(len(D), np.inf)), 1)[:, :10]
        r = np.arange(len(D))[:, None]
        dl, dn = D[r, nb].mean(1), Dn[r, nb].mean(1)
        ok = dl > 1e-6                              # skip items whose 10 neighbours are all exact duplicates
        loc = np.log(dn[ok] / dl[ok])
        glob = np.log(Dn[r, rnd].mean(1)[ok] / D[r, rnd].mean(1)[ok])
        out["local_log_ratio"].append(float((loc - glob).mean()))
        out["contract_share"].append(float((loc < glob).mean()))
        X, Y = raw[l], raw[l + 1]
        dlt = Y - X
        out["update_rel"].append(float((np.linalg.norm(dlt, axis=1) / np.linalg.norm(X, axis=1)).mean()))
        out["update_cos"].append(float((unit(dlt) * unit(X)).sum(1).mean()))
        Xc = X - X.mean(0)
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = int(np.searchsorted(np.cumsum(s ** 2) / (s ** 2).sum(), 0.9) + 1)
        dc = dlt - dlt.mean(0)
        out["update_in_top"].append(float(((dc @ Vt[:k].T) ** 2).sum() / (dc ** 2).sum()))
        out["top_k90"].append(k)
        log(a.name, l, f"PR {out['part_ratio'][-1]:.1f} twonn {out['twonn'][-1]:.1f} local {out['local_log_ratio'][-1]:+.3f} "
            f"update {out['update_rel'][-1]:.2f} cos {out['update_cos'][-1]:+.2f} in-top{k} {out['update_in_top'][-1]:.2f}")
        D = Dn
    (RESULTS / f"geometry_flow_{a.name}.json").write_text(json.dumps({"name": a.name, "split": a.split, "n": int(len(idx)),
                                                                     "layers": names, **out}))
    log("saved")


if __name__ == "__main__":
    main()
