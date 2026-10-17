"""research/PREREG_robustness.md B4-b/c. Reads results/robust_b4b.json (each block restored alone),
robust_b4c.json (20 random 4-block restorations), gq__qkl.json (original Q3 run) and gq_bf16__quantlens.json.
Writes results/robust_b4bc.json.

python score_b4bc.py
"""
import json

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS


def main():
    b = json.loads((RESULTS / "robust_b4b.json").read_text())["configs"]
    c = json.loads((RESULTS / "robust_b4c.json").read_text())["configs"]
    q3 = json.loads((RESULTS / "gq__qkl.json").read_text())
    rows = sorted((r for r in json.loads((RESULTS / "gq_bf16__quantlens.json").read_text())["rows"]
                   if r["variant"] == "gq_q4km"), key=lambda r: r["layer"])
    flip = np.diff([r["flip_norm_K512"] for r in rows])        # jump into lens layer j+1 = decoder block j
    act = np.diff([r["rel_l2"] for r in rows])
    n = len(flip)
    none = b["none"]["kl_token_mean"]
    red = np.array([none - b[f"block{j}"]["kl_token_mean"] for j in range(n)])
    rho_flip = float(spearmanr(flip, red).statistic)
    rho_act = float(spearmanr(act, red).statistic)
    # B4-c: reproducibility of draws 0-4 and the Q3 comparison against the 20-draw mean
    orig = {f"random{s}": q3["configs"][f"random{s}"] for s in range(5)}
    repro = {k: {"layers_equal": c[k]["layers"] == v["layers"],
                 "kl_new": c[k]["kl_token_mean"], "kl_orig": v["kl_token_mean"],
                 "abs_diff": abs(c[k]["kl_token_mean"] - v["kl_token_mean"])} for k, v in orig.items()}
    r20 = np.array([c[f"random{s}"]["kl_token_mean"] for s in range(20)])
    lens, actl2 = q3["configs"]["lens"]["kl_token_mean"], q3["configs"]["actl2"]["kl_token_mean"]
    first4 = q3["configs"]["first4"]["kl_token_mean"]
    out = {"B4b": {"spearman_flipjump_vs_klreduction": rho_flip, "spearman_actjump_vs_klreduction": rho_act,
                   "supported": rho_flip > rho_act, "kl_none": none, "kl_reduction_by_block": red.tolist(),
                   "top4_blocks_by_reduction": sorted(np.argsort(-red)[:4].tolist())},
           "B4c": {"draws_0_4": repro, "max_abs_diff_0_4": max(v["abs_diff"] for v in repro.values()),
                   "random20_mean": float(r20.mean()), "random20_sd": float(r20.std()),
                   "random5_mean_orig": q3["random_mean"],
                   "lens": lens, "actl2": actl2, "first4": first4,
                   "lens_below_random20_mean": bool(lens < r20.mean()), "lens_below_all20": bool(lens < r20.min()),
                   "lens_percentile_among_random": float((r20 < lens).mean()),
                   "first4_percentile_among_random": float((r20 < first4).mean())}}
    (RESULTS / "robust_b4bc.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"B4b": {k: v for k, v in out["B4b"].items() if k != "kl_reduction_by_block"},
                      "B4c": {k: v for k, v in out["B4c"].items() if k != "draws_0_4"}}, indent=1))
    print("draws 0-4 max |diff|:", out["B4c"]["max_abs_diff_0_4"], {k: v["layers_equal"] for k, v in repro.items()})


if __name__ == "__main__":
    main()
