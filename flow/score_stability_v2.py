"""Score research/PREREG_stability_v2.md from results/<pair>__stabv2.json -> results/stabv2_score.json.

Picks are hidden_states indices 0..L. Shortfalls are in points (0-100) below the pair's best layer.
P2's bootstrap resamples pairs and, within each pair, the 10 seed pairs (with replacement) before re-picking S10, so
selection noise enters the interval as in B1.

python score_stability_v2.py [--dev]     (--dev: score the development pairs only, no criteria)
"""
import json
import sys

import numpy as np

from common import RESULTS

TOPIC = ["llama32_1b", "llama32_1b_20ng", "smollm2_360m", "smollm2_360m_20ng", "qwen35_4b_base", "qwen35_4b_base_20ng"]
RATING = [f"{m}_yelp" for m in ("llama32_1b", "smollm2_360m", "qwen35_4b_base", "qwen3_17b", "qwen35_2b", "gemma4_e2b",
                                "smollm2_17b", "qwen35_4b", "pythia28b")]
DEV = ["qwen35_08b"]
FIG = "llama32_1b"


def pick_max(v):
    v = np.asarray(v)
    return int(np.flatnonzero(v == v.max()).max())


def fepoid(d, w):
    """Wang et al. 2026 §3.5 on layers 1..L (d indexed by layer, d[0] unused). Local maximum: d(l) > d(l-1) and
    d(l) >= d(l+1), l in 2..L-1. A peak is discarded if d keeps rising strictly from l+1 to min(l+w, L) and
    d(l) < d(min(l+w, L)). Earliest surviving peak; else the shallowest maximum; else L."""
    L = len(d) - 1
    peaks = [l for l in range(2, L) if d[l] > d[l - 1] and d[l] >= d[l + 1]]
    for l in peaks:
        e = min(l + w, L)
        rising = all(d[i] < d[i + 1] for i in range(l + 1, e))
        if not (rising and d[l] < d[e]):
            return l
    return peaks[0] if peaks else L


def id_min(d):
    """Tang & Yang 2026 Eq. 4 minimum, excluding the first floor(0.2 L) layers: argmin over floor(0.2L)+1..L."""
    L = len(d) - 1
    lo = int(np.floor(0.2 * L)) + 1
    return lo + int(np.argmin(d[lo:]))


def load(p):
    r = json.loads((RESULTS / f"{p}__stabv2.json").read_text())
    rows = r["rows"]
    S = np.array([x["stability"] for x in rows])            # layers x 10
    return {"S": S, "id": np.array([x["twonn"] for x in rows]), "p10": 100 * np.array([x["p10"] for x in rows]),
            "probe": 100 * np.array([x["probe"] for x in rows]),
            "few1": 100 * np.array([x["fewlabel_m1"] for x in rows]), "model": r["model"]}


def picks(D, pair):
    L = len(D["id"]) - 1
    w = 3 if pair.startswith("llama32_1b") else 7
    return {"S10": pick_max(D["S"].mean(1)), "last": L, "second_last": L - 1, "fepoid": fepoid(D["id"], w),
            "id_min": id_min(D["id"])}


def pair_summary(pair):
    D = load(pair)
    P = picks(D, pair)
    out = {"picks": P, "n_layers": len(D["id"]) - 1}
    for t in ("p10", "probe"):
        v = D[t]
        out[t] = {"best_layer": int(np.argmax(v)), "best": float(v.max()),
                  "shortfall": {k: float(v.max() - v[l]) for k, l in P.items()},
                  "single_pair_shortfall": float(np.mean([v.max() - v[pick_max(D["S"][:, j])] for j in range(10)]))}
    few = [pick_max(D["few1"][:, s]) for s in range(D["few1"].shape[1])]
    out["p10"]["shortfall"]["one_label"] = float(np.mean([D["p10"].max() - D["p10"][l] for l in few]))
    out["spearman_stability"] = {t: float(_spearman(D["S"].mean(1)[1:], D[t][1:])) for t in ("p10", "probe")}
    out["depth_fraction"] = {k: l / out["n_layers"] for k, l in P.items()}
    out["depth_fraction"]["best_p10"] = out["p10"]["best_layer"] / out["n_layers"]
    return out, D


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def boot_p2(Ds, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        acc = []
        for i in rng.integers(0, len(Ds), len(Ds)):
            D = Ds[i]
            cols = rng.integers(0, 10, 10)
            s = pick_max(D["S"][:, cols].mean(1))
            v = D["p10"]
            acc.append((v.max() - v[len(v) - 2]) - (v.max() - v[s]))
        diffs.append(np.mean(acc))
    return np.percentile(diffs, [2.5, 97.5]).tolist()


def main():
    dev = "--dev" in sys.argv
    sets = {"dev": DEV} if dev else {"topic": TOPIC, "rating": RATING}
    out = {"pairs": {}}
    Dd = {}
    for name, pairs in sets.items():
        for p in pairs:
            if (RESULTS / f"{p}__stabv2.json").exists():
                out["pairs"][p], Dd[p] = pair_summary(p)
    if dev:
        print(json.dumps(out, indent=1))
        return
    missing = [p for ps in sets.values() for p in ps if p not in out["pairs"]]
    T = [p for p in TOPIC if p in out["pairs"]]
    R = [p for p in RATING if p in out["pairs"]]
    sf = lambda ps, t, k: [out["pairs"][p][t]["shortfall"][k] for p in ps]
    mean = lambda x: float(np.mean(x)) if x else None
    beats = lambda ps, t, k: int(sum(a < b - 1e-9 for a, b in zip(sf(ps, t, "S10"), sf(ps, t, k))))
    s = {"missing": missing}
    if T:
        s["P1"] = {"S10_mean": mean(sf(T, "p10", "S10")), "beats_last": beats(T, "p10", "last"), "n": len(T)}
        s["P1"]["pass"] = s["P1"]["S10_mean"] <= 1.0 and s["P1"]["beats_last"] >= 5
        ci = boot_p2([Dd[p] for p in T])
        d = mean(sf(T, "p10", "second_last")) - s["P1"]["S10_mean"]
        s["P2"] = {"second_last_mean": mean(sf(T, "p10", "second_last")), "diff": d, "ci95": ci,
                   "pass": d >= 0.5 and ci[0] > 0}
        s["P3"] = {k: {"mean": mean(sf(T, "p10", k)), "S10_lower_in": beats(T, "p10", k)} for k in ("fepoid", "id_min")}
        s["P3"]["pass"] = all(s["P3"][k]["mean"] > s["P1"]["S10_mean"] and s["P3"][k]["S10_lower_in"] >= 4
                              for k in ("fepoid", "id_min"))
        s["P5_topic"] = {"S10_mean": mean(sf(T, "probe", "S10")), "beats_last": beats(T, "probe", "last")}
        s["P5_topic"]["pass"] = s["P5_topic"]["S10_mean"] <= 1.5 and s["P5_topic"]["beats_last"] >= 4
        s["one_label_mean_topic"] = mean(sf(T, "p10", "one_label"))
    if R:
        s["P4"] = {"S10_mean": mean(sf(R, "p10", "S10")), "beats_last": beats(R, "p10", "last"), "n": len(R)}
        s["P4"]["failed_as_predicted"] = s["P4"]["S10_mean"] >= 1.5 or s["P4"]["beats_last"] <= 6
        s["P5_rating"] = {"S10_mean": mean(sf(R, "probe", "S10")), "beats_last": beats(R, "probe", "last")}
    s["all_comparators"] = {t: {k: {"topic": mean(sf(T, t, k)), "rating": mean(sf(R, t, k))}
                                for k in ("S10", "last", "second_last", "fepoid", "id_min")} for t in ("p10", "probe")}
    out["summary"] = s
    if FIG in Dd:
        D = Dd[FIG]
        out["figure"] = {"pair": FIG, "stability": D["S"].mean(1).tolist(), "twonn": D["id"].tolist(),
                         "p10": D["p10"].tolist(), "probe": D["probe"].tolist(), "picks": out["pairs"][FIG]["picks"]}
    (RESULTS / "stabv2_score.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
