"""N2 scoring: unblind the second labelling (turn vs activation-matched vs random neurons; plan and result in
research/NOTE_neurons_vs_cells.md, "N2"). Reads results/blind2x8_key.json and results/blind2_raters/r{1,2}.json
(agent raters, blind); writes results/blind2_labels.json.

python blind2_score.py
"""
import json

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from common import RESULTS

KINDS = ("turn_var", "matched", "random")


def main():
    key = json.loads((RESULTS / "blind2x8_key.json").read_text())
    U = {u["id"]: u for u in key["units"]}
    ratio = {}
    for L, c in key["check"].items():
        base = 0 if L == "13" else 24                                   # pair ids are "<layer>-<pick index>"
        for u in U.values():
            if u["pair"] and u["layer"] == L:
                ratio[u["pair"]] = c["matched_std_ratio"][(int(u["pair"].split("-")[1]) - base) // 2]
    R = {r: {x["id"]: x for x in json.loads((RESULTS / "blind2_raters" / f"{r}.json").read_text())} for r in ("r1", "r2")}
    score = lambda r, i: (R[r][i]["pos_score"] + R[r][i]["neg_score"]) / 2
    out = {"design": "48 E2B units (layers 13-14): 16 turn neurons by write variance, 16 controls matched on activation "
                     "spread (outside the 8 selected per layer), the 16 random neurons of the first set; 6 strongest "
                     "tokens per pole, shuffled; two fresh blind LM raters (r1 Opus, r2 Sonnet)",
           "well_matched_threshold": 0.75, "n_well_matched": sum(v >= 0.75 for v in ratio.values()), "summary": {}}
    for r in R:
        sc = {k: [score(r, i) for i, u in U.items() if u["kind"] == k] for k in KINDS}
        pairs = {}
        for i, u in U.items():
            if u["pair"]:
                pairs.setdefault(u["pair"], {})[u["kind"]] = score(r, i)
        diff = lambda good: float(np.mean([p["turn_var"] - p["matched"] for k, p in pairs.items() if (ratio[k] >= 0.75) == good]))
        out["summary"][r] = {"mean_score": {k: float(np.mean(v)) for k, v in sc.items()},
                             "one_job": {k: int(sum(R[r][i]["one_job"] for i, u in U.items() if u["kind"] == k)) for k in KINDS},
                             "mannwhitney_p_turn_greater_matched": float(mannwhitneyu(sc["turn_var"], sc["matched"], alternative="greater").pvalue),
                             "mannwhitney_p_matched_greater_random": float(mannwhitneyu(sc["matched"], sc["random"], alternative="greater").pvalue),
                             "pair_diff_well_matched": diff(True), "pair_diff_poorly_matched": diff(False)}
    a = [score("r1", i) for i in sorted(U)]
    b = [score("r2", i) for i in sorted(U)]
    out["rater_spearman"] = float(spearmanr(a, b)[0])
    out["n2_holds"] = all(s["mannwhitney_p_turn_greater_matched"] < 0.05 for s in out["summary"].values())
    (RESULTS / "blind2_labels.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1), out["n_well_matched"], out["rater_spearman"], out["n2_holds"])


if __name__ == "__main__":
    main()
