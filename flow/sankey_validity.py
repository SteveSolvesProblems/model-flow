"""Can the flow sankey be made valid? (exploratory; criterion fixed before looking at any rebuilt sankey)

The old sankey (viewer_data.py: whiten-64 frame, K = 8 K-means per layer) draws regrouping that cannot be told from
bin noise (viz_validity.py: same-layer seed ARI median 0.26). Here, per layer of Gemma 4 E2B (articles; same 1,500 test
articles as the sankey), for several frames and K:
  frames   whiten64  the old frame (64 PCs, whitened)
           pca64     64 PCs of the centered vectors, not whitened (keeps the large directions large)
           robust64  per-dimension median/MAD scaling, then 64 PCs (outlier dimensions tamed)
           cos64     each layer centered, rows unit length (the geometry_probe view), then 64 PCs
  K        4, 8
  stab     mean pairwise ARI of single K-means runs over 5 seeds (fit on 10,000 train articles, assign the 1,500)
  cons     consensus bins: co-association of the 1,500 over 20 seeds, average-linkage cut at K; its stability is the
           ARI between two consensus runs on disjoint seed sets (0-19 vs 20-39)
Selection rule (fixed now): the frame/K whose consensus bins have the highest median stability over layers 1-35 wins,
if that median is >= 0.8; otherwise the sankey is not rebuilt from bins. The rebuilt sankey then shows per-layer
stability, and flows between shown layers are reported against the stability of the two layers.
Writes results/sankey_validity.json (+ consensus codes of every frame/K for the viewer).
"""
import json
from itertools import combinations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score as ari

from common import RESULTS, load_cache, set_threads
from lens import layers_of, log
from partition import assign, kmeans

NAME, N_DOCS, N_FIT, KS = "gemma4_e2b", 1500, 10000, (4, 8)


def frames(Xtr, Xte):
    mu = Xtr.mean(0)
    out = {}
    for f in ("whiten64", "pca64", "robust64", "cos64"):
        A, B = Xtr - mu, Xte - mu
        if f == "robust64":
            med = np.median(Xtr, 0)
            mad = np.median(np.abs(Xtr - med), 0) + 1e-6
            A, B = (Xtr - med) / mad, (Xte - med) / mad
        if f == "cos64":
            A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
            B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        m = A.mean(0)
        _, S, Vt = np.linalg.svd(A - m, full_matrices=False)
        W = Vt[:64].T
        if f == "whiten64":
            W = W / (S[:64] / np.sqrt(len(A)) + 1e-6)
        out[f] = ((A - m) @ W, (B - m) @ W)
    return out


def consensus(Ztr, Zte, K, seeds):
    n = len(Zte)
    co = np.zeros((n, n), np.float32)
    for s in seeds:
        c = assign(Zte, kmeans(Ztr, K, seed=s, nredo=1))[0]
        co += c[:, None] == c[None, :]
    D = 1 - co / len(seeds)
    np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform(D, checks=False), "average"), K, "maxclust") - 1


def main():
    set_threads()
    names = layers_of(NAME)
    yte = load_cache(names[0], "test", "l1")[1]
    idx = np.sort(np.random.default_rng(0).choice(len(yte), N_DOCS, replace=False))      # viewer_data.py's sample
    nf = np.load(__import__("common").cache_dir(names[0]) / "train_X.npy", mmap_mode="r").shape[0]
    fidx = np.sort(np.random.default_rng(3).choice(nf, N_FIT, replace=False))
    res = {}
    for li, nm in enumerate(names):
        Xtr, Xte = load_cache(nm, "train")[fidx], load_cache(nm, "test")[idx]
        for f, (Ztr, Zte) in frames(Xtr, Xte).items():
            for K in KS:
                runs = [assign(Zte, kmeans(Ztr, K, seed=s, nredo=1))[0] for s in range(5)]
                c1 = consensus(Ztr, Zte, K, range(20))
                c2 = consensus(Ztr, Zte, K, range(20, 40))
                r = res.setdefault(f"{f}_K{K}", {"stab": [], "cons": [], "codes": [], "l1": []})
                r["stab"].append(float(np.mean([ari(a, b) for a, b in combinations(runs, 2)])))
                r["cons"].append(float(ari(c1, c2)))
                r["codes"].append(c1.tolist())
                r["l1"].append(float(ari(c1, yte[idx])))
        log(li, {k: (round(v["stab"][-1], 2), round(v["cons"][-1], 2)) for k, v in res.items()})
    summ = {}
    for k, v in res.items():
        c = v["codes"]
        summ[k] = {"stab_median": float(np.median(v["stab"][1:])), "cons_median": float(np.median(v["cons"][1:])),
                   "cons_min": float(np.min(v["cons"][1:])),
                   "flow5_median": float(np.median([ari(c[l], c[l + 5]) for l in range(1, len(c) - 5)])),
                   "l1_median": float(np.median(v["l1"][1:]))}
    best = max(summ, key=lambda k: summ[k]["cons_median"])
    out = {"name": NAME, "n_docs": N_DOCS, "summary": summ, "best": best,
           "rebuild": summ[best]["cons_median"] >= 0.8, "per_layer": res}
    for k, s in sorted(summ.items(), key=lambda kv: -kv[1]["cons_median"]):
        log(k, {a: round(b, 3) for a, b in s.items()})
    log("best", best, "rebuild" if out["rebuild"] else "NOT rebuilt (stability < 0.8)")
    (RESULTS / "sankey_validity.json").write_text(json.dumps(out))
    log("saved")


if __name__ == "__main__":
    main()
