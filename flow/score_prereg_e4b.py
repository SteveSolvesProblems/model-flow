"""Score research/PREREG_scalar_e4b.md (S1-S3) exactly as written. Reads results/geometry_probe_gemma4_e4b.json.
Writes results/prereg_scalar_e4b.json."""
import json

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS
from score_heads import segment

SCALARS = [0.056, 0.146, 0.781, 0.641, 0.486, 0.57, 0.484, 0.609, 0.223, 0.412, 0.59, 0.531, 0.762, 0.746, 0.711, 0.57,
           0.676, 0.473, 0.566, 0.527, 0.441, 0.461, 0.142, 0.069, 0.447, 0.703, 0.715, 0.695, 0.613, 0.432, 0.703,
           0.812, 0.848, 0.832, 0.84, 0.828, 0.84, 0.875, 0.871, 0.828, 0.785, 0.432]   # copied from the prereg


def main():
    p = json.loads((RESULTS / "geometry_probe_gemma4_e4b.json").read_text())
    C, T = np.array(p["cos_centered"]), np.array(p["self_top1"])
    L = len(SCALARS)
    assert C.shape == (L + 1, L + 1)
    across = np.array([C[k, k + 1] for k in range(L)])                    # cosine across decoder layer k
    ks = np.arange(2, 41)
    two = ks[np.argsort(across[ks])[:2]].tolist()
    s1 = set(two) <= {22, 23, 24}
    k2 = np.arange(1, 41)
    rho = float(spearmanr(np.array(SCALARS)[k2], across[k2]).statistic)
    blocks = segment(T[1:, 1:])
    starts = [int(a) for a, _ in blocks]
    s3 = bool(set(starts) & {23, 24, 25})
    out = {"S1": {"two_sharpest": two, "cos": [round(float(across[k]), 3) for k in two], "supported": bool(s1)},
           "S2": {"spearman": round(rho, 3), "supported": rho >= 0.5},
           "S3": {"blocks": [[int(a), int(b)] for a, b in blocks], "supported": s3},
           "across_cos": [round(float(x), 3) for x in across], "sharpest_5": ks[np.argsort(across[ks])[:5]].tolist()}
    print(json.dumps(out, indent=1))
    (RESULTS / "prereg_scalar_e4b.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
