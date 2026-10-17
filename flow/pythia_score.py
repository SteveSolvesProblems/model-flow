"""research/PREREG_overnight.md E1: score P1–P3 from the per-checkpoint select.json files and partition codes.
Writes results/pythia_score.json."""
import json

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS
from token_lens import fast_nmi

STEPS = [0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]


def adj(a, b, rng, n=3):
    return fast_nmi(a, b) - np.mean([fast_nmi(rng.permutation(a), b) for _ in range(n)])


def main():
    rng = np.random.default_rng(0)
    out = {}
    for m in ("pythia410m", "pythia14b"):
        rows = []
        for st in STEPS:
            j = json.loads((RESULTS / f"{m}_s{st}__select.json").read_text())
            stab = np.array([r["stability_K512"] for r in j["rows"]])
            p = np.array([r["p@10_l2"] for r in j["rows"]])
            S, L = j["selection"]["l2"]["S"]["layer"], len(p)
            rows.append({"step": st, "S": S, "oracle": int(p.argmax()), "max_stability": float(stab.max()),
                         "oracle_p10": float(p.max()), "regret_S": float(p.max() - p[S]),
                         "regret_last": float(p.max() - p[L - 1]), "stability": stab.tolist(), "p10": p.tolist()})
        later = [r for r in rows if r["step"] >= 512]
        p1 = sum(r["regret_S"] <= r["regret_last"] + 1e-12 for r in later)
        p3 = spearmanr([r["max_stability"] for r in later], [r["oracle_p10"] for r in later]).statistic
        # P2: settling relative to the final checkpoint's own re-seed agreement
        codes = {st: np.load(RESULTS / "codes" / f"{m}_s{st}.npz") for st in STEPS}
        fin = codes[STEPS[-1]]
        L = fin["c0"].shape[0]
        ceil = np.array([fast_nmi(fin["c0"][l], fin["c1"][l]) for l in range(L)])
        ceil_adj = np.array([adj(fin["c0"][l], fin["c1"][l], rng) for l in range(L)])
        ratio = np.array([[fast_nmi(codes[st]["c0"][l], fin["c0"][l]) / ceil[l] for l in range(L)] for st in STEPS])
        ratio_adj = np.array([[adj(codes[st]["c0"][l], fin["c0"][l], rng) / ceil_adj[l] for l in range(L)] for st in STEPS])

        def settle(R):
            return np.array([STEPS[next((i for i in range(len(STEPS)) if R[i, l] >= 0.9), len(STEPS) - 1)] for l in range(L)])
        res = {}
        for tag, R in (("nmi", ratio), ("adjusted", ratio_adj)):
            s = settle(R)
            rho = spearmanr(np.arange(L), s).statistic
            boots = []
            for _ in range(2000):
                i = rng.integers(0, L, L)
                boots.append(spearmanr(i, s[i]).statistic)
            res[tag] = {"settle_step": s.tolist(), "spearman": float(rho),
                        "ci95": np.nanpercentile(boots, [2.5, 97.5]).tolist(), "ratio": R.tolist()}
        out[m] = {"checkpoints": rows, "P1_count": int(p1), "P1_holds": bool(p1 >= 7),
                  "P3_spearman": float(p3), "P3_holds": bool(p3 >= 0.7), "P2": res,
                  "P2_holds": bool(res["nmi"]["ci95"][0] > 0)}
        print(m, "P1", p1, "P3", round(p3, 3), "P2", round(res["nmi"]["spearman"], 3), res["nmi"]["ci95"])
    out["P1_supported"] = all(out[m]["P1_holds"] for m in ("pythia410m", "pythia14b"))
    out["P2_supported"] = all(out[m]["P2_holds"] for m in ("pythia410m", "pythia14b"))
    out["P3_supported"] = all(out[m]["P3_holds"] for m in ("pythia410m", "pythia14b"))
    (RESULTS / "pythia_score.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
