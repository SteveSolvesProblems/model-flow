"""Post hoc checks on the stability pick's second test (2026-10-18, after scoring; not pre-registered).
Writes results/stabv2_posthoc.json from results/<pair>__stabv2.json and results/stabv2_score.json:
  fixed_depth_shortfall      shortfall of layer round(f L) for f in 0.75, 0.85, 0.9, 0.95
  partial_spearman_...       rank correlation of stability and p@10 over layers 1..L with layer depth partialled out
  spearman_depth_p10         rank correlation of depth and p@10
  single_seedpair_agrees...  share of the 10 seed pairs whose own argmax equals the 10-pair pick (S10)

python stabv2_posthoc.py
"""
import json

import numpy as np
from scipy.stats import rankdata, spearmanr

from common import RESULTS

FRACS = (0.75, 0.85, 0.9, 0.95)


def main():
    sc = json.loads((RESULTS / "stabv2_score.json").read_text())["pairs"]
    out = {}
    for k in sc:
        R = json.loads((RESULTS / f"{k}__stabv2.json").read_text())["rows"]
        L = len(R) - 1
        p = 100 * np.array([r["p10"] for r in R])
        S = np.array([np.mean(r["stability"]) for r in R])
        St = np.array([r["stability"] for r in R])
        lay = np.arange(1, L + 1)
        res = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)
        rs, rp, rd = rankdata(S[1:]), rankdata(p[1:]), rankdata(lay)
        s10 = sc[k]["picks"]["S10"]
        out[k] = {"fixed_depth_shortfall": {str(f): float(p.max() - p[int(round(f * L))]) for f in FRACS},
                  "partial_spearman_stability_p10_given_depth": float(np.corrcoef(res(rs, rd), res(rp, rd))[0, 1]),
                  "spearman_depth_p10": float(spearmanr(lay, p[1:])[0]),
                  "single_seedpair_agrees_with_S10": float(np.mean([int(np.flatnonzero(St[:, j] == St[:, j].max()).max()) == s10
                                                                    for j in range(St.shape[1])]))}
    (RESULTS / "stabv2_posthoc.json").write_text(json.dumps(
        {"note": "post hoc (2026-10-18), not pre-registered: fixed-depth rules, stability vs quality with depth "
                 "partialled out, seed agreement", "pairs": out}, indent=1))
    print("saved", len(out), "pairs")


if __name__ == "__main__":
    main()
