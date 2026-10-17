"""Quantization through the attractor lens (research/PREREG_quantization_lens.md, Q1/Q4 + Q3 layer choice).

Per layer ℓ and scale K: the frame (whiten-64) and attractors (K-means seeds 0, 1) are fit on the REFERENCE
train features. Reference and quant test features both go through that frame and are assigned to those
attractors.
  flip_norm = 1 − mean_s NMI(ref_s, quant_s) / ceiling,  ceiling = NMI(ref_seed0, ref_seed1)
  changed   = share of test documents whose cell differs (mean over seeds)
Conventional baselines on the same layers: relative L2 ‖q − r‖/‖r‖ (per doc, mean), cosine, linear CKA.

python quant_lens.py --ref gq_bf16 --variants gq_q8_0 gq_q4km ...
"""
import argparse
import time

import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi

from common import load_cache, save_result, set_threads
from lens import layers_of, linear_cka
from partition import assign, kmeans


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fit_frame(Xtr, r=64, seed=0):
    """Same whiten-64 frame as lens.frame, returned as (mu, W) so other models' features can be projected."""
    mu = Xtr.mean(0)
    rng = np.random.default_rng(seed)
    S_idx = rng.choice(len(Xtr), min(12000, len(Xtr)), replace=False)
    _, S, Vt = np.linalg.svd(Xtr[S_idx] - mu, full_matrices=False)
    return mu, Vt[:r].T / (S[:r] / np.sqrt(len(S_idx)) + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="gq_bf16")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[8, 32, 128, 512])
    ap.add_argument("--cka-n", type=int, default=2000)
    ap.add_argument("--tag", default="quantlens")
    a = ap.parse_args()
    set_threads()

    ref_layers = layers_of(a.ref)
    var_layers = {v: layers_of(v) for v in a.variants}
    for v, ls in var_layers.items():
        assert len(ls) == len(ref_layers), f"{v}: {len(ls)} layers vs ref {len(ref_layers)}"
    cka_idx = np.random.default_rng(0).choice(6000, a.cka_n, replace=False)
    rows = []
    for li, rname in enumerate(ref_layers):
        Rtr, Rte = load_cache(rname, "train"), load_cache(rname, "test")
        cka_idx = cka_idx[cka_idx < len(Rte)]
        mu, W = fit_frame(Rtr)
        Ztr, Zte = (Rtr - mu) @ W, (Rte - mu) @ W
        C = {(K, s): kmeans(Ztr, K, seed=s) for K in a.ks for s in (0, 1)}
        ref_cells = {ks: assign(Zte, c)[0] for ks, c in C.items()}
        ceil = {K: nmi(ref_cells[(K, 0)], ref_cells[(K, 1)]) for K in a.ks}
        rn = np.linalg.norm(Rte, axis=1)
        for v in a.variants:
            Q = load_cache(var_layers[v][li], "test")
            Zq = (Q - mu) @ W
            r = {"variant": v, "layer": li,
                 "rel_l2": float(np.mean(np.linalg.norm(Q - Rte, axis=1) / (rn + 1e-12))),
                 "cos": float(np.mean(np.sum(Q * Rte, 1) / (np.linalg.norm(Q, axis=1) * rn + 1e-12))),
                 "cka": linear_cka(Rte[cka_idx].astype(np.float64), Q[cka_idx].astype(np.float64))}
            for K in a.ks:
                q = {s: assign(Zq, C[(K, s)])[0] for s in (0, 1)}
                agree = np.mean([nmi(ref_cells[(K, s)], q[s]) for s in (0, 1)])
                r[f"ceiling_K{K}"] = float(ceil[K])
                r[f"flip_norm_K{K}"] = float(1 - agree / ceil[K])
                r[f"changed_K{K}"] = float(np.mean([np.mean(ref_cells[(K, s)] != q[s]) for s in (0, 1)]))
            rows.append(r)
            log(f"L{li:02d} {v:10s} relL2={r['rel_l2']:.4f} cka={r['cka']:.4f} "
                + " ".join(f"flip{K}={r[f'flip_norm_K{K}']:.3f}" for K in a.ks))
    out = save_result(a.ref, a.tag, {"ref": a.ref, "variants": a.variants, "ks": a.ks, "rows": rows})
    log("saved", out)


if __name__ == "__main__":
    main()
