"""PREREG_quantization_lens.md Amendment 4: paired bootstrap CIs for the Q3 mixed-precision comparisons.
Writes results/gq__qkl_ci.json."""
import json

import numpy as np

from common import RESULTS


def main():
    orig = json.loads((RESULTS / "gq__qkl.json").read_text())["configs"]
    rerun = json.loads((RESULTS / "gq__qkl_rerun.json").read_text())["configs"]
    det = {k: abs(orig[k]["kl_token_mean"] - rerun[k]["kl_token_mean"]) for k in orig}
    z = np.load(RESULTS / "gq__qkl_perdoc.npz")
    S = {k[:-4]: z[k] for k in z.files if k.endswith("_sum")}
    N = {k[:-5]: z[k] for k in z.files if k.endswith("_ntok")}
    n = len(N["none"])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(10000, n))
    ntok = N["none"][idx].sum(1)
    boot = {k: S[k][idx].sum(1) / ntok for k in S}
    boot["random_mean"] = np.mean([boot[f"random{i}"] for i in range(5)], 0)
    point = {k: float(S[k].sum() / N["none"].sum()) for k in S}
    point["random_mean"] = float(np.mean([point[f"random{i}"] for i in range(5)]))
    out = {"determinism_max_abs_diff": max(det.values()), "deterministic": max(det.values()) < 1e-4, "point": point,
           "n_docs": int(n), "resamples": 10000, "diffs": {}}
    for a in ("lens", "lens_raw"):
        for b in ("actl2", "random_mean", "first4"):
            d = boot[a] - boot[b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            out["diffs"][f"{a} - {b}"] = {"point": point[a] - point[b], "ci95": [float(lo), float(hi)],
                                          "excludes_0": bool(lo > 0 or hi < 0)}
    (RESULTS / "gq__qkl_ci.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("determinism_max_abs_diff", "deterministic")}, indent=1))
    for k, v in out["diffs"].items():
        print(f"{k:22s} {v['point']:+.4f}  [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  {'excludes 0' if v['excludes_0'] else 'includes 0'}")


if __name__ == "__main__":
    main()
