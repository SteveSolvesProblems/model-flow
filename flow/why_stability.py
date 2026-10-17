"""Why does the stability pick work? (research/PREREG_why_stability.md)

Part A, articles (the 11 held-out model-task pairs of robust_b1.json): per layer, in the selector's own frame
(64 whitened PCs fit on 20,000 train articles) and with its own partition (K = 512 K-means, seed 0):
  silhouette   mean silhouette of 5,000 test articles under the K = 512 partition (label-free separation)
  knn_margin   mean over test articles of 1 - d1 / d10 (Euclidean, in the frame): how much closer the nearest
               neighbour is than the 10th; high = tight local groups
  part_ratio   participation ratio of the centered test covariance, raw coordinates (effective dimension)
  top64_share  share of raw variance in the top 64 PCs (how much the frame keeps)
Part B, tokens (Gemma 4 E2B tok_base, Qwen3-1.7B tok_qwen3; 3,000 eval token positions, the geometry_probe sample):
  per layer, among each token's 10 nearest neighbours (centered cosine): share with the same next word, the same
  current word, the same document, and the same topic (DBpedia l1); plus chance levels.
Writes results/why_stability.json. Scoring: score_why_stability.py.
"""
import json
import sys

import numpy as np
from sklearn.metrics import silhouette_score

from common import CACHE_DIRS, RESULTS, load_cache, sample_idx, set_threads
from lens import frame, layers_of, log
from partition import assign, kmeans
from robustness import HELD, SUB

N_TEST = 5000


def part_a():
    out = {}
    for p in HELD:
        names = layers_of(p)
        rows = []
        for li, nm in enumerate(names):
            Xtr, Xte = load_cache(nm, "train"), load_cache(nm, "test")
            idx = np.random.default_rng(0).choice(len(Xtr), min(SUB, len(Xtr)), replace=False)
            tidx = np.sort(np.random.default_rng(1).choice(len(Xte), min(N_TEST, len(Xte)), replace=False))
            Ftr, Fte = frame(Xtr[idx], Xte[tidx])
            c = assign(Fte, kmeans(Ftr, 512, seed=0, nredo=1))[0]
            sil = float(silhouette_score(Fte, c)) if len(np.unique(c)) > 1 else float("nan")
            G = (Fte ** 2).sum(1)
            D = np.sqrt(np.maximum(G[:, None] + G[None, :] - 2 * Fte @ Fte.T, 0))
            np.fill_diagonal(D, np.inf)
            Ds = np.sort(D, 1)[:, :10]
            margin = float(np.mean(1 - Ds[:, 0] / (Ds[:, 9] + 1e-12)))
            Xc = Xte[tidx] - Xte[tidx].mean(0)
            ev = np.linalg.svd(Xc, compute_uv=False) ** 2
            rows.append({"silhouette": sil, "knn_margin": margin, "part_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
                         "top64_share": float(ev[:64].sum() / ev.sum())})
            log("A", p, li, {k: round(v, 3) for k, v in rows[-1].items()})
        out[p] = {k: [r[k] for r in rows] for k in rows[0]}
    return out


def part_b(pairs=(("tok_base", "tok_meta.npz"), ("tok_qwen3", "tok_meta_qwen3.npz"))):
    out = {}
    for name, metaf in pairs:
        meta = np.load(next(c / metaf for c in CACHE_DIRS if (c / metaf).exists()), allow_pickle=True)
        ev, l1 = meta["eval"], meta["eval_l1"]
        names = layers_of(name)
        idx = sample_idx(len(load_cache(names[0], "eval")), 3000, 0)
        doc, cur, nxt, top = ev[idx, 0], ev[idx, 3], ev[idx, 4], l1[idx]
        valid_n = nxt >= 0
        same = lambda lab, nb: (lab[nb] == lab[:, None]).mean(1)
        res = {"next": [], "current": [], "doc": [], "topic": []}
        for nm in names:
            X = load_cache(nm, "eval")[idx].astype(np.float32)
            X = X - X.mean(0)
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            S = X @ X.T
            np.fill_diagonal(S, -np.inf)
            nb = np.argsort(-S, 1)[:, :10]
            nx_ok = valid_n[nb] & valid_n[:, None]
            res["next"].append(float(((nxt[nb] == nxt[:, None]) & nx_ok).sum() / nx_ok.sum()))
            res["current"].append(float(same(cur, nb).mean()))
            res["doc"].append(float(same(doc, nb).mean()))
            res["topic"].append(float(same(top, nb).mean()))
            log("B", name, nm, {k: round(v[-1], 3) for k, v in res.items()})
        pr = lambda lab: float(sum((np.bincount(lab[lab >= 0]) / (lab >= 0).sum()) ** 2))
        res["chance"] = {"next": pr(nxt), "current": pr(cur), "doc": pr(doc), "topic": pr(top)}
        out[name] = res
    return out


if __name__ == "__main__":
    set_threads()
    parts = sys.argv[1:] or ["a", "b"]
    f = RESULTS / "why_stability.json"
    out = json.loads(f.read_text()) if f.exists() else {}
    if "b" in parts:
        out["tokens"] = part_b()
        f.write_text(json.dumps(out))
    if "bbos" in parts:      # Gemma re-extracted with <bos> (NOTE_geometry.md, 2026-10-05)
        out["tokens_bos"] = part_b((("tok_base_bos", "tok_meta_bos.npz"),))
        f.write_text(json.dumps(out))
    if "a" in parts:
        out["articles"] = part_a()
        f.write_text(json.dumps(out))
    log("saved")
