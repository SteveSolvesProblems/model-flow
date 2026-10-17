"""research/PREREG_overnight.md E5: per-document routing stability at layer S as a label-free predictor of retrieval
failure, against centroid distance, centroid margin and hubness. Needs results/codes/<pair>.npz (layer S) and
<pair>__baselines_a.json (per-document hits). Writes results/routing.json."""
import json

import faiss
import numpy as np
from scipy.stats import spearmanr

from common import RESULTS, load_cache, set_threads
from lens import layers_of
from metrics import auroc

HELD = ["qwen35_08b", "qwen35_08b_20ng", "qwen35_4b", "qwen35_4b_20ng", "gemma4_e4b", "gemma4_e4b_20ng", "smollm2_17b",
        "smollm2_17b_20ng", "qwen3_17b_20ng", "qwen35_2b_20ng", "gemma4_e2b_20ng"]


def routing_stability(c0, c1):
    """Jaccard of each document's co-members under seed 0 vs seed 1 (self excluded)."""
    n0 = np.bincount(c0)[c0] - 1
    n1 = np.bincount(c1)[c1] - 1
    joint = c0.astype(np.int64) * 4096 + c1
    _, inv, cnt = np.unique(joint, return_inverse=True, return_counts=True)
    inter = cnt[inv] - 1
    union = n0 + n1 - inter
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def hubness(X, mu, k=10):
    A = X - mu
    A = np.ascontiguousarray(A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12), dtype=np.float32)
    idx = faiss.IndexFlatIP(A.shape[1])
    idx.add(A)
    _, I = idx.search(A, k + 1)
    return np.bincount(I[:, 1:].ravel(), minlength=len(A)).astype(float)


def main():
    set_threads()
    rng = np.random.default_rng(0)
    out, q1, q2, q2_two = {}, 0, 0, 0
    for p in HELD:
        a = json.loads((RESULTS / f"{p}__baselines_a.json").read_text())
        S = a["S"]
        hits = np.array(a["perdoc_hits"][str(S)])
        z = np.load(RESULTS / "codes" / f"{p}.npz")
        c0, c1 = z["c0"][S].astype(np.int64), z["c1"][S].astype(np.int64)
        assert (c0 >= 0).all(), f"codes for layer {S} missing in {p}"
        d1, d2 = z["d1"][S].astype(float), z["d2"][S].astype(float)
        name = layers_of(p)[S]
        hub = hubness(load_cache(name, "test"), load_cache(name, "train").mean(0))
        stab = routing_stability(c0, c1)
        fail = (hits <= 0.5).astype(int)
        rho = spearmanr(stab, hits).statistic
        boots = []
        for _ in range(1000):
            i = rng.integers(0, len(hits), len(hits))
            boots.append(spearmanr(stab[i], hits[i]).statistic)
        ci = np.percentile(boots, [2.5, 97.5]).tolist()
        au = {"routing_stability": auroc(-stab, fail), "centroid_distance": auroc(d1, fail),
              "centroid_margin": auroc(d1 / np.maximum(d2, 1e-9), fail), "hubness": auroc(-hub, fail)}
        beats_all = all(au["routing_stability"] > au[k] for k in ("centroid_distance", "centroid_margin", "hubness"))
        beats_two = all(au["routing_stability"] > au[k] for k in ("centroid_distance", "centroid_margin"))
        q1 += ci[0] > 0
        q2 += beats_all
        q2_two += beats_two
        out[p] = {"layer": S, "spearman": float(rho), "ci95": ci, "failure_rate": float(fail.mean()), "auroc": au,
                  "beats_all_three": bool(beats_all), "beats_original_two": bool(beats_two),
                  "spearman_baselines": {"centroid_distance": float(spearmanr(-d1, hits).statistic),
                                         "centroid_margin": float(spearmanr(-d1 / np.maximum(d2, 1e-9), hits).statistic),
                                         "hubness": float(spearmanr(hub, hits).statistic)}}
        print(p, f"rho={rho:.3f} {ci}", {k: round(v, 3) for k, v in au.items()})
    out["Q1_pairs"] = int(q1); out["Q1_supported"] = bool(q1 >= 9)
    out["Q2_pairs"] = int(q2); out["Q2_supported"] = bool(q2 >= 7)
    out["Q2_original_two_baselines_pairs"] = int(q2_two)
    (RESULTS / "routing.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
