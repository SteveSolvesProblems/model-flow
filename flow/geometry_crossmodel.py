"""Control for geometry_linmap (exploratory): is "a rotation recovers the item" specific to layers of one model, or
does any pair of representations of the same articles align that well? Same procedure as geometry_linmap (fit split,
held-out 3,000-item sample, each layer centered by its fit mean, rows unit length, self_top1 among the 3,000):
  within      layer pairs inside model A (reference; matches geometry_linmap)
  cross       layer of model A -> layer of model B on the same articles (caches share the document order; checked
              via the level-3 labels). Different widths: semi-orthogonal Procrustes (U Vt of the d_A x d_B cross
              product, keeps distances inside A's span)
  shuffled    model A pair fitted on mispaired rows (fit targets permuted): the map learns nothing item-specific,
              so self_top1 should fall to about chance
  identity / linear (ridge, same lam as geometry_linmap) are reported alongside rotation for every pair.
Article-level top-1 is a coarse test (layer 0, the averaged word embeddings, already aligns across models), so each
map also reports harder measures:
  hard     self_top1 among items of the same level-3 class only (topic neighbours share more words), and
  hard_rand  the same with an equally sized random candidate pool (the pool-size baseline for hard; hard < hard_rand
           means topic neighbours are what the map confuses)
  resid    relative residual |E_a M - E_b|^2 / |E_b|^2 after the map (0 = exact, ~1 = unrelated)
  rank     mean rank of the true item among all 3,000 (1 = always first)
Writes results/geometry_crossmodel_<A>__<B>.json.

python geometry_crossmodel.py --a gemma4_e2b --b qwen3_17b
"""
import argparse
import json
import time

import numpy as np

from common import RESULTS, cache_dir, sample_idx, set_threads
from lens import layers_of, log


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def rows(name, split, idx):
    return np.load(cache_dir(name) / f"{split}_X.npy", mmap_mode="r")[idx].astype(np.float32)


def top1(P, T):
    return float(((unit(P) @ T.T).argmax(1) == np.arange(len(P))).mean())


def scores(P, T, cls):
    """top1, hard (same-class candidates only), resid, rank for mapped rows P against targets T."""
    S = unit(P) @ T.T
    n = len(P)
    true = S[np.arange(n), np.arange(n)]
    same = cls[:, None] == cls[None, :]
    Sh = np.where(same, S, -np.inf)
    rng = np.random.default_rng(7)
    rnd = np.argsort(rng.random((n, n)), 1) < same.sum(1, keepdims=True)    # random pool, same size as the class pool
    rnd[np.arange(n), np.arange(n)] = True
    Sr = np.where(rnd, S, -np.inf)
    return {"top1": float((S.argmax(1) == np.arange(n)).mean()),
            "hard": float((Sh.argmax(1) == np.arange(n)).mean()),
            "hard_rand": float((Sr.argmax(1) == np.arange(n)).mean()),
            "resid": float(((unit(P) - T) ** 2).sum() / (T ** 2).sum()),
            "rank": float(1 + (S > true[:, None]).sum(1).mean())}


def pick(L, k):
    return sorted(set(np.linspace(0, L - 1, k).round().astype(int).tolist()))


def maps(Fa, Fb, Ea, Eb, lam, cls):
    """identity (only when widths match), rotation (semi-orthogonal Procrustes), linear (ridge): top1 plus the harder
    measures (scores)."""
    C = Fa.T @ Fb
    U, _, Vt = np.linalg.svd(C, full_matrices=False)
    G = Fa.T @ Fa
    G[np.diag_indices(len(G))] += lam * np.trace(G) / len(G)
    rot, lin = scores(Ea @ (U @ Vt), Eb, cls), scores(Ea @ np.linalg.solve(G, C), Eb, cls)
    out = {"identity": top1(Ea, Eb) if Ea.shape[1] == Eb.shape[1] else None, "rotation": rot["top1"],
           "linear": lin["top1"], "rotation_scores": rot, "linear_scores": lin}
    if Ea.shape[1] == Eb.shape[1]:
        out["identity_scores"] = scores(Ea, Eb, cls)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--fit", default="train")
    ap.add_argument("--eval", default="test")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--n-fit", type=int, default=10000)
    ap.add_argument("--k", type=int, default=6, help="evenly spaced layers per model")
    ap.add_argument("--lam", type=float, default=1e-2)
    a = ap.parse_args()
    set_threads()
    na, nb = layers_of(a.a), layers_of(a.b)
    ya = [np.load(cache_dir(n[0]) / f"{s}_y_l3.npy") for n in (na, nb) for s in (a.fit, a.eval)]
    assert (ya[0] == ya[2]).all() and (ya[1] == ya[3]).all(), "caches are not in the same document order"
    ne = np.load(cache_dir(na[0]) / f"{a.eval}_X.npy", mmap_mode="r").shape[0]
    nf = np.load(cache_dir(na[0]) / f"{a.fit}_X.npy", mmap_mode="r").shape[0]
    eidx = sample_idx(ne, a.n, 0)
    cls = ya[1][eidx]
    fidx = np.sort(np.random.default_rng(2).choice(nf, min(a.n_fit, nf), replace=False))

    def prep(nm):
        Xf = rows(nm, a.fit, fidx)
        mu = Xf.mean(0)
        return unit(Xf - mu), unit(rows(nm, a.eval, eidx) - mu)

    la, lb = pick(len(na), a.k), pick(len(nb), a.k)
    A = {i: prep(na[i]) for i in la}
    B = {j: prep(nb[j]) for j in lb}
    perm = np.random.default_rng(5).permutation(len(fidx))
    t0 = time.time()
    out = {"a": a.a, "b": a.b, "n": int(len(eidx)), "n_fit": int(len(fidx)), "lam": a.lam, "chance_top1": 1.0 / len(eidx),
           "layers_a": la, "layers_b": lb, "within": {}, "cross": {}, "shuffled": {}}
    for i in la:
        for i2 in la:
            if i2 > i:
                out["within"][f"{i}-{i2}"] = maps(A[i][0], A[i2][0], A[i][1], A[i2][1], a.lam, cls)
                out["shuffled"][f"{i}-{i2}"] = maps(A[i][0], A[i2][0][perm], A[i][1], A[i2][1], a.lam, cls)
        for j in lb:
            out["cross"][f"{i}-{j}"] = maps(A[i][0], B[j][0], A[i][1], B[j][1], a.lam, cls)
        log(a.a, i, "cross rot top1/hard/hard_rand/resid", {j: tuple(round(out["cross"][f"{i}-{j}"]["rotation_scores"][m], 3)
            for m in ("top1", "hard", "hard_rand", "resid")) for j in lb}, f"({time.time() - t0:.0f}s)")
    for k, v in out["within"].items():
        log("within", k, {m: (None if x is None else round(x, 3)) for m, x in v.items() if not isinstance(x, dict)},
            "hard", round(v["rotation_scores"]["hard"], 3), "rand", round(v["rotation_scores"]["hard_rand"], 3), "resid", round(v["rotation_scores"]["resid"], 3),
            "shuffled rot", round(out["shuffled"][k]["rotation"], 4))
    (RESULTS / f"geometry_crossmodel_{a.a}__{a.b}.json").write_text(json.dumps(out))
    log("saved")


if __name__ == "__main__":
    main()
