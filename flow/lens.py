"""Attractor lens: per-layer, multi-scale, label-free partitions of a frozen model, compared coordinate-free.

Each layer gets its own frame (center + PCA-whiten, fit on train) and its own K-means attractors at several
scales K. Every test input then has a trace codes[layer, K] = cell id. Layers are never compared by
position, only by partition agreement, so layer-to-layer coordinate shifts do not matter.

  E1 flow map: layer×layer partition agreement (NMI) per scale, with a same-layer re-seed ceiling
     and linear CKA as the conventional comparison.
  E2 hierarchy depth: per layer/scale, NMI of the label-free partition with each label level, plus
     nearest-class-mean (NCC) and linear-probe accuracy as label-using references.
  E3 architecture: layer-to-layer churn (1 − adjacent NMI) grouped by the block type that produced
     the layer (e.g. Qwen3.5 linear_attention vs full_attention).

python lens.py --model qwen35_2b --levels l1 l2 l3 --ks 8 32 128 512
"""
import argparse
import json
import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score as nmi

from common import CACHE, CACHE_DIRS, cache_meta, load_cache, save_result, set_threads
from partition import assign, kmeans

warnings.filterwarnings("ignore")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def layers_of(model):
    return sorted({p.name for c in CACHE_DIRS for p in c.glob(f"{model}_L[0-9]*")})


def frame(Xtr, Xte, r=64, preproc="whiten", seed=0):
    """Per-layer coordinate frame, fit on train only.
    Default: whiten the top-64 PCs. Whitening all ~256+ PCs scales noise directions up to unit variance and
    destroyed partition stability (re-seed NMI 0.57 vs 0.80 at r=64 on Qwen3.5 L12, K=32)."""
    mu = Xtr.mean(0)
    Xtr, Xte = Xtr - mu, Xte - mu
    if preproc == "center":
        return Xtr, Xte
    rng = np.random.default_rng(seed)
    S_idx = rng.choice(len(Xtr), min(12000, len(Xtr)), replace=False)
    _, S, Vt = np.linalg.svd(Xtr[S_idx], full_matrices=False)
    r = min(r, len(S))
    W = Vt[:r].T if preproc == "pca" else Vt[:r].T / (S[:r] / np.sqrt(len(S_idx)) + 1e-6)
    return (Xtr @ W).astype(np.float32), (Xte @ W).astype(np.float32)


def linear_cka(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    hsic = np.linalg.norm(A.T @ B) ** 2
    return float(hsic / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="cache prefix, e.g. qwen35_2b")
    ap.add_argument("--levels", nargs="+", default=["l1", "l2", "l3"])
    ap.add_argument("--ks", nargs="+", type=int, default=[8, 32, 128, 512])
    ap.add_argument("--preproc", default="whiten", choices=["whiten", "pca", "center"])
    ap.add_argument("--probe-n", type=int, default=8000)
    ap.add_argument("--cka-n", type=int, default=2000)
    a = ap.parse_args()
    set_threads()

    names = layers_of(a.model)
    L = len(names)
    log(f"{a.model}: {L} layers, scales {a.ks}, preproc {a.preproc}")
    labels_tr = {lv: load_cache(names[0], "train", lv)[1] for lv in a.levels}
    labels_te = {lv: load_cache(names[0], "test", lv)[1] for lv in a.levels}
    layer_type = [cache_meta(n).get("layer_type") for n in names]
    rng = np.random.default_rng(0)
    probe_idx = rng.choice(len(labels_tr[a.levels[0]]), min(a.probe_n, len(labels_tr[a.levels[0]])), replace=False)
    cka_idx = rng.choice(len(labels_te[a.levels[0]]), min(a.cka_n, len(labels_te[a.levels[0]])), replace=False)

    codes = np.zeros((L, len(a.ks), len(labels_te[a.levels[0]])), dtype=np.int32)
    codes_seed2 = np.zeros_like(codes)
    cka_feats = []
    depth = []  # per layer rows
    for li, name in enumerate(names):
        Xtr = load_cache(name, "train")
        Xte = load_cache(name, "test")
        Ftr, Fte = frame(Xtr, Xte, preproc=a.preproc)
        cka_feats.append(Xte[cka_idx] - Xtr.mean(0))   # conventional CKA uses raw (centered) features
        row = {"layer": li, "layer_type": layer_type[li]}
        for ki, K in enumerate(a.ks):
            C = kmeans(Ftr, K, seed=0, nredo=1)
            codes[li, ki], _ = assign(Fte, C)
            codes_seed2[li, ki], _ = assign(Fte, kmeans(Ftr, K, seed=1, nredo=1))
            for lv in a.levels:
                row[f"nmi_{lv}_K{K}"] = float(nmi(labels_te[lv], codes[li, ki]))
            row[f"reseed_nmi_K{K}"] = float(nmi(codes[li, ki], codes_seed2[li, ki]))
        # label-using references: nearest class mean (Voronoi of class means) and linear probe
        for lv in a.levels:
            ytr, yte = labels_tr[lv], labels_te[lv]
            cls = np.unique(ytr)
            means = np.stack([Ftr[ytr == c].mean(0) for c in cls])
            j, _ = assign(Fte, means)
            row[f"ncc_{lv}"] = float((cls[j] == yte).mean())
            if lv != a.levels[-1] or len(cls) <= 100:
                clf = LogisticRegression(max_iter=500).fit(Ftr[probe_idx], ytr[probe_idx])
                row[f"probe_{lv}"] = float(clf.score(Fte, yte))
        depth.append(row)
        log(f"L{li:02d} {layer_type[li] or '':17s} " +
            " ".join(f"nmi_{a.levels[0]}K{a.ks[1]}={row[f'nmi_{a.levels[0]}_K{a.ks[1]}']:.3f}" for _ in [0]) +
            f" ncc_{a.levels[0]}={row[f'ncc_{a.levels[0]}']:.3f} reseed={row[f'reseed_nmi_K{a.ks[1]}']:.3f}")

    # E1: layer × layer agreement per scale, and CKA.
    # Cross-seed: layer i's seed-0 partition vs layer j's seed-1 partition (and vice versa). Same-seed
    # comparisons are inflated because adjacent layers share K-means initialization and converge together.
    # Diagonal = same-layer re-seed agreement (the ceiling); `flow_norm` divides by sqrt(ceil_i * ceil_j).
    flow, flow_norm, flow_same = {}, {}, {}
    for ki, K in enumerate(a.ks):
        M = np.zeros((L, L))
        S = np.eye(L)
        for i in range(L):
            M[i, i] = nmi(codes[i, ki], codes_seed2[i, ki])
            for j in range(i + 1, L):
                M[i, j] = M[j, i] = 0.5 * (nmi(codes[i, ki], codes_seed2[j, ki]) +
                                           nmi(codes_seed2[i, ki], codes[j, ki]))
                S[i, j] = S[j, i] = nmi(codes[i, ki], codes[j, ki])
        d = np.sqrt(np.diag(M))
        flow[f"K{K}"] = M.round(4).tolist()
        flow_norm[f"K{K}"] = (M / np.outer(d, d)).round(4).tolist()
        flow_same[f"K{K}"] = S.round(4).tolist()
    cka = np.eye(L)
    for i in range(L):
        for j in range(i + 1, L):
            cka[i, j] = cka[j, i] = linear_cka(cka_feats[i], cka_feats[j])

    # E3: churn between adjacent layers, attributed to the block that produced layer l
    churn = []
    for li in range(1, L):
        r = {"layer": li, "block_type": layer_type[li]}
        for ki, K in enumerate(a.ks):
            r[f"churn_K{K}"] = float(1 - flow_norm[f"K{K}"][li - 1][li])
        churn.append(r)
    by_type = {}
    for t in sorted({c["block_type"] for c in churn if c["block_type"]}):
        by_type[t] = {f"K{K}": float(np.mean([c[f"churn_K{K}"] for c in churn if c["block_type"] == t]))
                      for K in a.ks}

    np.save(CACHE / f"{a.model}_lens_codes_{a.preproc}.npy", codes)
    np.save(CACHE / f"{a.model}_lens_codes_seed1_{a.preproc}.npy", codes_seed2)
    out = save_result(a.model, f"lens_{a.preproc}", {
        "args": vars(a), "layers": names, "ks": a.ks, "depth": depth, "flow_nmi": flow,
        "flow_nmi_normalized": flow_norm, "flow_nmi_sameseed_biased": flow_same,
        "cka": cka.round(4).tolist(), "churn": churn, "churn_by_block_type": by_type})
    log("saved", out)


if __name__ == "__main__":
    main()
