"""Score research/PREREG_why_stability.md from results/why_stability.json and robust_b1.json.
Writes results/why_stability_score.json.
"""
import json

import numpy as np
from scipy.stats import rankdata

from common import RESULTS

MEDIATORS = ("silhouette", "knn_margin", "part_ratio", "top64_share")


def partial(a, b, controls):
    X = np.column_stack([np.ones(len(a))] + [np.asarray(c, float) for c in controls])
    r = lambda y: rankdata(y) - X @ np.linalg.lstsq(X, rankdata(y), rcond=None)[0]
    return float(np.corrcoef(r(a), r(b))[0, 1])


def main():
    W = json.loads((RESULTS / "why_stability.json").read_text())
    B = json.loads((RESULTS / "robust_b1.json").read_text())
    out = {"articles": {}, "tokens": {}}
    for p, m in W["articles"].items():
        stab = np.array(B[p]["stability"]).mean(1)[1:]
        q = np.array(B[p]["p10"])[1:]
        d = np.arange(1, len(q) + 1, dtype=float)
        dep = [d, d ** 2]
        r = {"stab_quality": partial(stab, q, dep)}
        for k in MEDIATORS:
            v = np.array(m[k])[1:]
            r[f"{k}_stab"] = partial(v, stab, dep)
            r[f"{k}_quality"] = partial(v, q, dep)
            r[f"stab_quality_given_{k}"] = partial(stab, q, dep + [rankdata(v)])
            pick = int(np.flatnonzero(v == v.max()).max())
            r[f"{k}_pick_shortfall"] = float(100 * (q.max() - q[pick]))
        r["stab_pick_shortfall"] = float(100 * B[p]["regret_S10"])
        out["articles"][p] = r
    A = out["articles"]
    halved = lambda k: sum(abs(A[p][f"stab_quality_given_{k}"]) <= abs(A[p]["stab_quality"]) / 2 for p in A)
    out["A1"] = {"n_pos": sum(A[p]["silhouette_stab"] > 0 for p in A), "pass": sum(A[p]["silhouette_stab"] > 0 for p in A) >= 9}
    out["A2"] = {"n_halved": halved("silhouette"), "pass": halved("silhouette") >= 7}
    out["A3"] = {"n_halved": {k: halved(k) for k in MEDIATORS},
                 "pass": all(halved(k) < halved("silhouette") for k in ("part_ratio", "top64_share"))}
    out["mean_pick_shortfall"] = {k: float(np.mean([A[p][f"{k}_pick_shortfall"] for p in A]))
                                  for k in MEDIATORS + ("stab",)}
    b1 = b2 = b3 = True
    for name, t in W["tokens"].items():
        nx, tp, cu = (np.array(t[k])[1:] for k in ("next", "topic", "current"))
        L = len(nx)
        q3 = int(np.floor(0.75 * L))
        r = {"next_last": float(nx[-1]), "next_max_first3q": float(nx[:q3].max()),
             "topic_peak_layer": int(np.argmax(tp)) + 1, "topic_last": float(tp[-1]), "topic_peak": float(tp.max()),
             "current_last": float(cu[-1]), "current_max": float(cu.max()), "L": L}
        r["B1"] = r["next_last"] > r["next_max_first3q"]
        r["B2"] = (np.argmax(tp) < q3) and (tp[-1] < tp.max())
        r["B3"] = cu[-1] < cu.max()
        b1, b2, b3 = b1 and r["B1"], b2 and r["B2"], b3 and r["B3"]
        out["tokens"][name] = r
    out["B1"], out["B2"], out["B3"] = bool(b1), bool(b2), bool(b3)
    (RESULTS / "why_stability_score.json").write_text(json.dumps(out, indent=1, default=bool))
    for k in ("A1", "A2", "A3", "B1", "B2", "B3", "mean_pick_shortfall"):
        print(k, out[k])
    for p, r in A.items():
        print(f"{p:20s}", {k: round(v, 2) for k, v in r.items() if "given" in k or k == "stab_quality"})
    for n, r in out["tokens"].items():
        print(n, r)


if __name__ == "__main__":
    main()
