"""PREREG_overnight.md Amendment 1 (chance correction), applied to the frozen K=512 selector: does subtracting the
permutation floor from each layer's cross-seed NMI change which layer is picked, or the held-out regret?
Needs results/codes/<pair>.npz with all layers. Writes results/adjusted_select.json."""
import json

import numpy as np

from common import RESULTS
from token_lens import fast_nmi

HELD = ["qwen35_08b", "qwen35_08b_20ng", "qwen35_4b", "qwen35_4b_20ng", "gemma4_e4b", "gemma4_e4b_20ng", "smollm2_17b",
        "smollm2_17b_20ng", "qwen3_17b_20ng", "qwen35_2b_20ng", "gemma4_e2b_20ng"]


def main():
    rng = np.random.default_rng(0)
    out, reg, reg_adj = {}, [], []
    for p in HELD:
        lab = "fine" if p.endswith("_20ng") else "l2"
        j = json.loads((RESULTS / f"{p}__select.json").read_text())
        p10 = np.array([r[f"p@10_{lab}"] for r in j["rows"]])
        z = np.load(RESULTS / "codes" / f"{p}.npz")
        c0, c1 = z["c0"], z["c1"]
        raw = np.array([fast_nmi(c0[l], c1[l]) for l in range(len(c0))])
        floor = np.array([np.mean([fast_nmi(rng.permutation(c0[l]), c1[l]) for _ in range(3)]) for l in range(len(c0))])
        adj = raw - floor
        S = int(np.flatnonzero(raw == raw.max()).max())
        Sa = int(np.flatnonzero(adj == adj.max()).max())
        reg.append(p10.max() - p10[S]); reg_adj.append(p10.max() - p10[Sa])
        out[p] = {"S": S, "S_adjusted": Sa, "regret": float(reg[-1]), "regret_adjusted": float(reg_adj[-1]),
                  "floor_min": float(floor.min()), "floor_max": float(floor.max()), "raw": raw.tolist(), "floor": floor.tolist()}
        print(p, S, Sa, round(float(floor.min()), 4), round(float(floor.max()), 4))
    out["mean_regret"] = float(np.mean(reg)); out["mean_regret_adjusted"] = float(np.mean(reg_adj))
    out["same_layer"] = int(sum(out[p]["S"] == out[p]["S_adjusted"] for p in HELD))
    (RESULTS / "adjusted_select.json").write_text(json.dumps(out, indent=1))
    print(out["mean_regret"], out["mean_regret_adjusted"], out["same_layer"])


if __name__ == "__main__":
    main()
