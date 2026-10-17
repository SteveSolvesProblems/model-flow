"""Gemma's layer scalar out of the update statistics (exploratory, after the 2026-10-04 audit). Gemma 4 multiplies the
whole residual by a learned scalar s_k after layer k: h_{k+1} = s_k (h_k + delta_k). geometry_flow's update
h_{k+1} - h_k = (s_k - 1) h_k + s_k delta_k is dominated by the rescale when s_k is small (cosine with h_k -> -1).
Here the layer's own write delta_k = h_{k+1} / s_k - h_k (same 3,000 items as geometry_flow):
  update_rel   mean |delta_k| / |h_k|        update_cos   mean cosine(delta_k, h_k)
The last cached state is after the final norm, so the last layer is left out.
--drop-massive T: zero, in every state, the coordinates that are massive at any layer (mean |x| > T x the median
coordinate's, as in outlier_dims.py), then recompute; also reports the turn (centered cosine across each layer).
Writes results/geometry_flow_noscale_<name>.json.

python flow_noscale.py --name gemma4_e2b --split test --weights gemma4_e2b
"""
import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS, cache_dir, sample_idx, set_threads
from lens import layers_of, log


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", required=True, help="weight_structure_<weights>.json holds the scalars")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--drop-massive", type=float, default=0)
    a = ap.parse_args()
    set_threads()
    s = np.array(json.loads((RESULTS / f"weight_structure_{a.weights}.json").read_text())["layer_scalar"])
    names = layers_of(a.name)
    n0 = np.load(cache_dir(names[0]) / f"{a.split}_X.npy", mmap_mode="r").shape[0]
    idx = sample_idx(n0, a.n, 0)
    ld = lambda nm: np.load(cache_dir(nm) / f"{a.split}_X.npy", mmap_mode="r")[idx].astype(np.float32)
    out = {"update_rel": [], "update_cos": [], "update_rel_raw": [], "update_cos_raw": [], "turn_cos": []}
    drop = np.array([], dtype=int)
    if a.drop_massive:
        drop = np.unique(np.concatenate([np.flatnonzero(np.abs(m := ld(nm)).mean(0) > a.drop_massive * np.median(np.abs(m).mean(0)))
                                         for nm in names[:-1]]))
        log("dropping", len(drop), "massive coordinates:", drop.tolist()[:20])
    out["dropped"] = drop.tolist()
    ld0 = ld
    def ld(nm):
        Z = ld0(nm)
        Z[:, drop] = 0
        return Z
    X = ld(names[0])
    for k in range(len(names) - 2):
        Y = ld(names[k + 1])
        d = Y / s[k] - X
        out["update_rel"].append(float((np.linalg.norm(d, axis=1) / np.linalg.norm(X, axis=1)).mean()))
        out["update_cos"].append(float((unit(d) * unit(X)).sum(1).mean()))
        r = Y - X
        out["update_rel_raw"].append(float((np.linalg.norm(r, axis=1) / np.linalg.norm(X, axis=1)).mean()))
        out["update_cos_raw"].append(float((unit(r) * unit(X)).sum(1).mean()))
        out["turn_cos"].append(float((unit(X - X.mean(0)) * unit(Y - Y.mean(0))).sum(1).mean()))
        log(a.name, k, f"s {s[k]:.3f} write {out['update_rel'][-1]:.2f} cos {out['update_cos'][-1]:+.2f} | raw "
            f"{out['update_rel_raw'][-1]:.2f} {out['update_cos_raw'][-1]:+.2f}")
        X = Y
    ks = range(1, len(out["update_rel"]))
    sc = [s[k] for k in ks]
    out["spearman_vs_scalar"] = {m: float(spearmanr(sc, [out[m][k] for k in ks]).statistic) for m in
                                 ("update_rel", "update_cos", "update_rel_raw", "update_cos_raw")}
    log(out["spearman_vs_scalar"])
    tag = f"_nomassive{int(a.drop_massive)}" if a.drop_massive else ""
    (RESULTS / f"geometry_flow_noscale_{a.name}{tag}.json").write_text(json.dumps({"name": a.name, "scalars": s.tolist(), **out}))
    log("saved")


if __name__ == "__main__":
    main()
