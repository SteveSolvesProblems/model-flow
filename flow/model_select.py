"""research/PREREG_overnight.md E4 (exploratory): does a model's maximum map stability rank models by retrieval?
Writes results/model_select.json."""
import itertools
import json

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS

MODELS = ["qwen3_17b", "qwen35_08b", "qwen35_2b", "qwen35_4b", "gemma4_e2b", "gemma4_e4b", "smollm2_17b"]


def main():
    out = {}
    for ds, suffix, lab in (("dbpedia", "", "l2"), ("20ng", "_20ng", "fine")):
        stab, orc, at_s = [], [], []
        for m in MODELS:
            j = json.loads((RESULTS / f"{m}{suffix}__select.json").read_text())
            st = np.array([r["stability_K512"] for r in j["rows"]])
            p = np.array([r[f"p@10_{lab}"] for r in j["rows"]])
            S = j["selection"][lab]["S"]["layer"]
            stab.append(st.max()); orc.append(p.max()); at_s.append(p[S])
        rho = spearmanr(stab, orc).statistic
        perms = [spearmanr(stab, np.array(orc)[list(q)]).statistic for q in itertools.permutations(range(len(MODELS)))]
        p_val = float(np.mean(np.array(perms) >= rho - 1e-12))
        out[ds] = {"models": MODELS, "max_stability": stab, "oracle_p10": orc, "p10_at_S": at_s,
                   "spearman_stab_vs_oracle": float(rho), "perm_p_one_sided": p_val,
                   "spearman_stab_vs_p10_at_S": float(spearmanr(stab, at_s).statistic)}
        print(ds, round(rho, 3), p_val)
    (RESULTS / "model_select.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
