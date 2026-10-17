"""research/PREREG_flow.md F2-a/F2-b from the per-checkpoint select.json files of the new Pythia runs.
F2-c (settling order) needs partition codes (partition_codes.py) and is scored separately when they exist.
Also reported (post hoc, red-team control): Spearman(training step, best-layer p@10), so the monitor can be compared
with the step trend, and the seed spread of S and of the oracle layer at each checkpoint.
Writes results/flow_f2.json.

python score_f2.py
"""
import json

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS

try:
    from token_lens import fast_nmi
except ImportError:
    fast_nmi = None

STEPS = [0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
NEW = ["pythia410m_seed1", "pythia410m_seed2", "pythia410m_seed3", "pythia410m_seed4", "pythia410m_dedup",
       "pythia14b_dedup", "pythia28b"]
REF = ["pythia410m", "pythia14b"]          # E1 runs (seed 0, standard data), for comparison only


def run(m):
    rows = []
    for st in STEPS:
        f = RESULTS / f"{m}_s{st}__select.json"
        if not f.exists():
            return None
        j = json.loads(f.read_text())
        stab = np.array([r["stability_K512"] for r in j["rows"]])
        p = np.array([r["p@10_l2"] for r in j["rows"]])
        rows.append({"step": st, "S": j["selection"]["l2"]["S"]["layer"], "oracle": int(p.argmax()),
                     "max_stability": float(stab.max()), "oracle_p10": float(p.max()), "n_layers": len(p)})
    later = [r for r in rows if r["step"] >= 512]
    ms, op = [r["max_stability"] for r in later], [r["oracle_p10"] for r in later]
    a = float(spearmanr(ms, op).statistic)
    step_rho = float(spearmanr([r["step"] for r in later], op).statistic)
    by = {r["step"]: r["oracle"] for r in rows}
    early = max(by[1000], by[2000])
    late_med = float(np.median([by[s] for s in STEPS if s >= 8000]))
    return {"checkpoints": rows, "F2a_spearman": a, "F2a_holds": a >= 0.7, "step_spearman_posthoc": step_rho,
            "F2b_early": early, "F2b_late_median": late_med, "F2b_holds": early > late_med}


def settling(m):
    """F2-c: per-layer settle step = first checkpoint whose seed-0 cells reach >= 0.9 of the final checkpoint's
    re-seed NMI (same definition as E1 P2)."""
    files = {st: RESULTS / "codes" / f"{m}_s{st}.npz" for st in STEPS}
    if fast_nmi is None or not all(f.exists() for f in files.values()):
        return None
    codes = {st: np.load(f) for st, f in files.items()}
    fin = codes[STEPS[-1]]
    L = fin["c0"].shape[0]
    ceil = np.array([fast_nmi(fin["c0"][l], fin["c1"][l]) for l in range(L)])
    R = np.array([[fast_nmi(codes[st]["c0"][l], fin["c0"][l]) / ceil[l] for l in range(L)] for st in STEPS])
    s = np.array([STEPS[next((i for i in range(len(STEPS)) if R[i, l] >= 0.9), len(STEPS) - 1)] for l in range(L)])
    q = L // 3
    return {"settle_step": s.tolist(), "spearman_depth": float(spearmanr(np.arange(L), s).statistic),
            "median_first_third": float(np.median(s[:q])), "median_middle_third": float(np.median(s[q:2 * q])),
            "median_last_third": float(np.median(s[2 * q:])), "ratio": R.tolist()}


def main():
    out = {m: run(m) for m in NEW + REF}
    for m in NEW + REF:
        if out[m]:
            out[m]["F2c"] = settling(m)
    done = [m for m in NEW if out[m]]
    na = sum(out[m]["F2a_holds"] for m in done)
    nb = sum(out[m]["F2b_holds"] for m in done)
    seeds = ["pythia410m"] + [f"pythia410m_seed{i}" for i in range(1, 5)]
    spread = {}
    if all(out[s] for s in seeds):
        for i, st in enumerate(STEPS):
            S = [out[s]["checkpoints"][i]["S"] for s in seeds]
            O = [out[s]["checkpoints"][i]["oracle"] for s in seeds]
            spread[st] = {"S": S, "oracle": O}
    out["seed_spread_410m"] = spread
    out["summary"] = {"runs_scored": done, "runs_missing": [m for m in NEW if not out[m]],
                      "F2a_count": int(na), "F2b_count": int(nb),
                      "F2a": bool(na >= 6) if len(done) == 7 else None, "F2b": bool(nb >= 6) if len(done) == 7 else None}
    (RESULTS / "flow_f2.json").write_text(json.dumps(out, indent=1))
    for m in NEW + REF:
        r = out[m]
        if r:
            print(f"{m:20s} F2a rho={r['F2a_spearman']:+.2f} (step rho {r['step_spearman_posthoc']:+.2f})  "
                  f"F2b early={r['F2b_early']} late_med={r['F2b_late_median']} -> {r['F2b_holds']}")
    for m in NEW + REF:
        c = out[m] and out[m]["F2c"]
        if c:
            print(f"F2c {m:20s} rho(depth,settle)={c['spearman_depth']:+.2f} thirds={c['median_first_third']:.0f}/"
                  f"{c['median_middle_third']:.0f}/{c['median_last_third']:.0f}  settle={c['settle_step']}")
    for st, v in spread.items():
        print("step", st, "S", v["S"], "oracle", v["oracle"])
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
