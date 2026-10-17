"""Score the held-out layer-selection results against research/PREREG_layer_selection.md.

Pre-registered: criteria 1 and 2 on the primary label; secondary churn hypothesis on Qwen3.5 held-out pairs.
Exploratory (added after the pre-registration, reported separately): last−2 and ¾-depth baselines, and the
secondary label.  "Beats" means strictly lower regret; a tie does not count.
"""
import json

import numpy as np

from common import RESULTS

HELDOUT = [("qwen35_08b", "l2", "l3"), ("qwen35_08b_20ng", "fine", "coarse"), ("qwen35_4b", "l2", "l3"),
           ("qwen35_4b_20ng", "fine", "coarse"), ("gemma4_e4b", "l2", "l3"), ("gemma4_e4b_20ng", "fine", "coarse"),
           ("smollm2_17b", "l2", "l3"), ("smollm2_17b_20ng", "fine", "coarse"), ("qwen3_17b_20ng", "fine", "coarse"),
           ("qwen35_2b_20ng", "fine", "coarse"), ("gemma4_e2b_20ng", "fine", "coarse")]
CHURN = ["qwen35_08b", "qwen35_08b_20ng", "qwen35_4b", "qwen35_4b_20ng"]


def regrets(d, lv):
    rows = d["rows"]
    p = np.array([r[f"p@10_{lv}"] for r in rows])
    L = len(rows)
    sel = {k: d["selection"][lv][k]["layer"] for k in ("S", "last", "middle", "rankme_min")}
    sel.update({"last-2": L - 3, "3/4 depth": round(0.75 * (L - 1))})
    return {k: float(p.max() - p[v]) for k, v in sel.items()}, sel, int(p.argmax()), L


def main():
    out = {"pairs": {}, "missing": []}
    for name, prim, sec in HELDOUT:
        f = RESULTS / f"{name}__select.json"
        if not f.exists():
            out["missing"].append(name)
            continue
        d = json.loads(f.read_text())
        rp, sel, orc, L = regrets(d, prim)
        rs, _, _, _ = regrets(d, sec)
        out["pairs"][name] = {"primary": rp, "secondary": rs, "layers": sel, "oracle_layer": orc, "n_layers": L,
                              "churn": d.get("churn")}
    P = out["pairs"]
    n = len(P)
    mean = {k: float(np.mean([P[p]["primary"][k] for p in P])) for k in next(iter(P.values()))["primary"]}
    mean_sec = {k: float(np.mean([P[p]["secondary"][k] for p in P])) for k in mean}
    beats_last = sum(P[p]["primary"]["S"] < P[p]["primary"]["last"] for p in P)
    c1 = mean["S"] < mean["last"] and beats_last >= 0.75 * n
    c2 = mean["S"] <= mean["rankme_min"] and mean["S"] <= mean["middle"]
    churn = {p: P[p]["churn"]["linear_minus_full"] for p in CHURN if p in P}
    churn_ok = sum(c["ci95"][0] > 0 for c in churn.values())
    out.update({"n_pairs": n, "mean_regret_primary": mean, "mean_regret_secondary": mean_sec,
                "S_beats_last": f"{beats_last}/{n}", "criterion1": c1, "criterion2": c2,
                "verdict": "pass both" if c1 and c2 else "pass 1 only" if c1 else "fail 1",
                "churn": churn, "churn_supported": f"{churn_ok}/{len(churn)} CIs exclude 0 (need ≥3 of 4)"})
    (RESULTS / "select_heldout_score.json").write_text(json.dumps(out, indent=2))
    print(f"{'pair':18s} {'L':>3s} {'orc':>4s} " + " ".join(f"{k:>10s}" for k in mean))
    for p, v in P.items():
        print(f"{p:18s} {v['n_layers']:3d} {v['oracle_layer']:4d} " + " ".join(f"{v['primary'][k]:10.4f}" for k in mean))
    print(f"{'MEAN primary':27s} " + " ".join(f"{mean[k]:10.4f}" for k in mean))
    print(f"{'MEAN secondary':27s} " + " ".join(f"{mean_sec[k]:10.4f}" for k in mean))
    print("S beats last:", out["S_beats_last"], "| C1:", c1, "| C2:", c2, "| verdict:", out["verdict"],
          "| missing:", out["missing"])
    for p, c in churn.items():
        print(f"churn {p}: {c['point']:.4f} CI {np.round(c['ci95'], 4).tolist()}")
    print(out["churn_supported"])


if __name__ == "__main__":
    main()
