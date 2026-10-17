"""Score research/PREREG_baselines.md (B1–B3). Writes results/baselines_score.json."""
import json

import numpy as np

from common import RESULTS

DEV = ["qwen3_17b", "qwen35_2b", "gemma4_e2b"]
HELD = ["qwen35_08b", "qwen35_08b_20ng", "qwen35_4b", "qwen35_4b_20ng", "gemma4_e4b", "gemma4_e4b_20ng", "smollm2_17b",
        "smollm2_17b_20ng", "qwen3_17b_20ng", "qwen35_2b_20ng", "gemma4_e2b_20ng"]
RIVALS = ["twonn", "dataset_entropy", "prompt_entropy", "curvature"]


def labels(p):
    return ("fine", "coarse") if p.endswith("_20ng") else ("l2", "l3")


def load(p):
    prim, sec = labels(p)
    sel = json.loads((RESULTS / f"{p}__select.json").read_text())
    a = json.loads((RESULTS / f"{p}__baselines_a.json").read_text())
    t = json.loads((RESULTS / f"{p}__token_metrics.json").read_text())["rows"]
    L = len(sel["rows"])
    assert len(a["rows"]) == L == len(t), (p, L, len(a["rows"]), len(t))
    prof = {"twonn": [r["twonn"] for r in a["rows"]], "dataset_entropy": [r["dataset_entropy"] for r in a["rows"]],
            "prompt_entropy": [r["prompt_entropy"] for r in t], "curvature": [r["curvature"] for r in t]}
    return {"p10": np.array([r[f"p@10_{prim}"] for r in sel["rows"]]),
            "p10s": np.array([r[f"p@10_{sec}"] for r in sel["rows"]]),
            "S": sel["selection"][prim]["S"]["layer"], "rankme_min": sel["selection"][prim]["rankme_min"]["layer"],
            "prof": {k: np.array(v, dtype=float) for k, v in prof.items()}, "a": a, "L": L}


def pick(x, direction):
    x = np.nan_to_num(x, nan=-np.inf if direction == "argmax" else np.inf)
    v = x.max() if direction == "argmax" else x.min()
    return int(np.flatnonzero(x == v).max())           # ties go deeper, as for rule S


def first_peak_above_median(x):
    med = np.nanmedian(x)
    for i in range(1, len(x) - 1):
        if x[i] >= x[i - 1] and x[i] >= x[i + 1] and x[i] > med:
            return i
    return int(np.nanargmax(x))


def main():
    D = {p: load(p) for p in DEV + HELD}
    reg = lambda d, l: float(d["p10"].max() - d["p10"][l])
    regs = lambda d, l: float(d["p10s"].max() - d["p10s"][l])

    # directions fixed on dev only
    directions = {}
    for r in RIVALS:
        dev = {dr: np.mean([reg(D[p], pick(D[p]["prof"][r], dr)) for p in DEV]) for dr in ("argmax", "argmin")}
        directions[r] = {"chosen": min(dev, key=lambda k: (dev[k], k != "argmax")), "dev_mean_regret": dev}

    sel = {}
    for p in HELD:
        d = D[p]
        s = {"S": d["S"], "last": d["L"] - 1, "rankme_min": d["rankme_min"],
             "twonn_first_peak": first_peak_above_median(d["prof"]["twonn"])}
        for r in RIVALS:
            s[r] = pick(d["prof"][r], directions[r]["chosen"])
        sel[p] = {k: {"layer": v, "regret": reg(d, v), "regret_secondary": regs(d, v)} for k, v in s.items()}
        sel[p]["oracle"] = int(d["p10"].argmax())
    mean = {k: float(np.mean([sel[p][k]["regret"] for p in HELD])) for k in sel[HELD[0]] if k != "oracle"}
    mean_sec = {k: float(np.mean([sel[p][k]["regret_secondary"] for p in HELD])) for k in sel[HELD[0]] if k != "oracle"}
    wins = {r: {"S_better": sum(sel[p]["S"]["regret"] < sel[p][r]["regret"] - 1e-12 for p in HELD),
                "tie": sum(abs(sel[p]["S"]["regret"] - sel[p][r]["regret"]) <= 1e-12 for p in HELD),
                "S_worse": sum(sel[p]["S"]["regret"] > sel[p][r]["regret"] + 1e-12 for p in HELD)} for r in RIVALS}
    b1 = all(mean["S"] < mean[r] for r in RIVALS)

    # B2: few-label
    few = {}
    for m in (1, 2, 5, 10):
        few[m] = float(np.mean([np.mean(D[p]["a"]["fewlabel"][str(m)]["regrets"]) for p in HELD]))
    mstar = next((m for m in (1, 2, 5, 10) if few[m] <= mean["S"]), None)

    # B3: bootstrap
    rng = np.random.default_rng(0)
    per_pair, diffs_by_pair = {}, {}
    for p in HELD:
        d = D[p]
        h = {int(k): np.array(v) for k, v in d["a"]["perdoc_hits"].items()}
        S, last, orc = d["S"], d["L"] - 1, int(d["p10"].argmax())
        n = len(h[S])
        idx = rng.integers(0, n, size=(2000, n))
        dl = (h[S] - h[last]) if S != last else np.zeros(n)
        do = (h[orc] - h[S]) if orc != S else np.zeros(n)
        bl, bo = dl[idx].mean(1), do[idx].mean(1)
        per_pair[p] = {"S_minus_last": {"point": float(dl.mean()), "ci95": np.percentile(bl, [2.5, 97.5]).tolist()},
                       "oracle_minus_S": {"point": float(do.mean()), "ci95": np.percentile(bo, [2.5, 97.5]).tolist()},
                       "same_layer_S_last": S == last}
        diffs_by_pair[p] = dl
    above = sum((not per_pair[p]["same_layer_S_last"]) and per_pair[p]["S_minus_last"]["ci95"][0] > 0 for p in HELD)
    two = []
    for _ in range(2000):
        ps = rng.choice(HELD, len(HELD))
        two.append(np.mean([diffs_by_pair[p][rng.integers(0, len(diffs_by_pair[p]), len(diffs_by_pair[p]))].mean()
                            for p in ps]))
    two_ci = np.percentile(two, [2.5, 97.5]).tolist()
    b3 = two_ci[0] > 0 and above >= 8

    out = {"directions": directions, "mean_regret": mean, "mean_regret_secondary": mean_sec, "wins_vs_rivals": wins,
           "B1_pass": b1, "fewlabel_mean_regret": few, "m_star": mstar,
           "B2_expectation_met": few[1] > mean["S"] and few[10] <= mean["S"],
           "per_pair_ci": per_pair, "pairs_S_above_last": above, "two_level_mean_S_minus_last":
               {"point": float(np.mean([diffs_by_pair[p].mean() for p in HELD])), "ci95": two_ci},
           "B3_pass": b3, "selections": sel}
    (RESULTS / "baselines_score.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("directions", "mean_regret", "wins_vs_rivals", "B1_pass", "fewlabel_mean_regret",
                                          "m_star", "B2_expectation_met", "pairs_S_above_last",
                                          "two_level_mean_S_minus_last", "B3_pass")}, indent=1))


if __name__ == "__main__":
    main()
