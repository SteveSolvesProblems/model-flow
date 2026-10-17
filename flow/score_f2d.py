"""research/PREREG_flow.md Amendment 3, F2-d: settling at finer checkpoint resolution (Pythia-410m seeds 0, 1, 4).
Settle step per layer = first checkpoint whose seed-0 cells reach >= 0.9 of the final checkpoint's re-seed NMI.
Writes results/flow_f2d.json.

python score_f2d.py
"""
import json

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS
from token_lens import fast_nmi

STEPS = [0, 128, 256, 512, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 16000, 32000, 64000, 143000]
RUNS = ["pythia410m", "pythia410m_seed1", "pythia410m_seed4"]


def main():
    out, a, b = {}, 0, 0
    for m in RUNS:
        codes = {st: np.load(RESULTS / "codes" / f"{m}_s{st}.npz") for st in STEPS}
        fin = codes[STEPS[-1]]
        L = fin["c0"].shape[0]
        ceil = np.array([fast_nmi(fin["c0"][l], fin["c1"][l]) for l in range(L)])
        R = np.array([[fast_nmi(codes[st]["c0"][l], fin["c0"][l]) / ceil[l] for l in range(L)] for st in STEPS])
        s = np.array([STEPS[next((i for i in range(len(STEPS)) if R[i, l] >= 0.9), len(STEPS) - 1)] for l in range(L)])
        q = L // 3
        med = [float(np.median(s[:q])), float(np.median(s[q:2 * q])), float(np.median(s[2 * q:]))]
        rho = float(spearmanr(np.arange(L), s).statistic)
        r = {"settle_step": s.tolist(), "median_thirds": med, "spearman_depth": rho,
             "F2da": med[0] < med[1], "F2db": rho > 0, "earliest_layer_step": int(s[1:].min()),
             "ratio": R.tolist()}
        a, b = a + r["F2da"], b + r["F2db"]
        out[m] = r
        print(f"{m:18s} thirds {med} rho {rho:+.2f}  settle {s.tolist()}")
    out["summary"] = {"F2da_count": int(a), "F2db_count": int(b), "F2da": bool(a == 3), "F2db": bool(b == 3)}
    (RESULTS / "flow_f2d.json").write_text(json.dumps(out, indent=1, default=bool))
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
