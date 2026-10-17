"""Score layer_skip results against geometry (exploratory). Layer k makes the change from cached state k to k+1.
  skip   Spearman(skip KL at k, geometric change at k = 1 - centered cosine), raw and partial (controlling for the
         relative update size at k, from geometry_flow), plus boundary vs interior layers (blocks from heads_vs_blocks)
  swap   swap KL for pairs (k, k+1) that straddle a block boundary vs pairs inside a block (excluding the first two and
         last layers, which are fragile everywhere)
Writes results/layer_skip_scored.json.
"""
import json

import numpy as np
from scipy.stats import mannwhitneyu, rankdata, spearmanr

from common import RESULTS

RUNS = [("gemma4_e2b", ["gemma4_e2b", "tok_base"]), ("qwen3_17b", ["qwen3_17b", "tok_qwen3"])]


def partial_spearman(x, y, z):
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def main():
    out = {}
    blocks_all = json.loads((RESULTS / "heads_vs_blocks.json").read_text())
    for wname, gnames in RUNS:
        r = json.loads((RESULTS / f"layer_skip_{wname}.json").read_text())["results"]
        L = sum(1 for k in r if k.startswith("skip_"))
        skl = np.array([r[f"skip_{k}"]["kl"] for k in range(L)])
        for g in gnames:
            C = np.array(json.loads((RESULTS / f"geometry_probe_{g}.json").read_text())["cos_centered"])
            upd = np.array(json.loads((RESULTS / f"geometry_flow_{g}.json").read_text())["update_rel"])
            change = np.array([1 - C[k, k + 1] for k in range(L)])
            ks = np.arange(1, L - 1)                                    # first and last layers excluded throughout
            blocks = blocks_all[g]["blocks"]
            starts = {b["layers"][0] for b in blocks}
            boundary = np.array([k in starts or (k + 1) in starts for k in range(L)])
            res = {"skip_vs_change": float(spearmanr(skl[ks], change[ks]).statistic),
                   "skip_vs_update": float(spearmanr(skl[ks], upd[ks]).statistic),
                   "skip_vs_change_partial_update": partial_spearman(skl[ks], change[ks], upd[ks]),
                   "skip_boundary_median": float(np.median(skl[ks][boundary[ks]])),
                   "skip_interior_median": float(np.median(skl[ks][~boundary[ks]])),
                   "blocks": [b["layers"] for b in blocks]}
            bi, bo = skl[ks][boundary[ks]], skl[ks][~boundary[ks]]
            res["skip_boundary_vs_interior_p"] = float(mannwhitneyu(bi, bo, alternative="greater").pvalue) if len(bi) and len(bo) else None
            if f"swap_0" in r:
                sw = np.array([r[f"swap_{k}"]["kl"] for k in range(L - 1)])
                pairs = np.arange(2, L - 2)
                across = np.array([(k + 1) in starts for k in pairs])
                res.update({"swap_across": [(int(k), round(float(sw[k]), 3)) for k in pairs[across]],
                            "swap_inside_median": float(np.median(sw[pairs[~across]])),
                            "swap_first_last": {"0-1": float(sw[0]), "1-2": float(sw[1]), f"{L-2}-{L-1}": float(sw[-1])}})
            out[g] = res
            print(g, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
    (RESULTS / "layer_skip_scored.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
