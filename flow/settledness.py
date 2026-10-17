"""research/PREREG_null_followups.md P2: is map stability mostly settledness?

For the 11 held-out pairs (robustness.HELD), in why_stability.py part A's frame and documents (whiten-64 fit on
20,000 train documents, seed 0; 5,000 test documents, seed 1): set_l = kNN10 overlap between layer l and l+1.
Stability and quality come from <pair>__select.json (stability_K512; p@10 on the topical label).
Writes results/settledness.json.
"""
import json

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS, load_cache, set_threads
from lens import frame, layers_of, log
from robustness import HELD, SUB

N_TEST, K = 5000, 10


def knn(F, k=K):
    G = (F ** 2).sum(1)
    D = G[:, None] + G[None, :] - 2 * F @ F.T
    np.fill_diagonal(D, np.inf)
    return np.argpartition(D, k, axis=1)[:, :k]


def pair(p):
    nb = []
    for nm in layers_of(p):
        Xtr, Xte = load_cache(nm, "train"), load_cache(nm, "test")
        idx = np.random.default_rng(0).choice(len(Xtr), min(SUB, len(Xtr)), replace=False)
        tidx = np.sort(np.random.default_rng(1).choice(len(Xte), min(N_TEST, len(Xte)), replace=False))
        _, Fte = frame(Xtr[idx], Xte[tidx])
        nb.append(knn(Fte.astype(np.float32)))
        log(p, nm)
    return [float(np.mean([len(set(a) & set(b)) / K for a, b in zip(nb[l], nb[l + 1])])) for l in range(len(nb) - 1)]


def main():
    set_threads()
    out = {}
    for p in HELD:
        lab = "fine" if p.endswith("_20ng") else "l2"
        sel = json.loads((RESULTS / f"{p}__select.json").read_text())
        rows = sel["rows"]
        st = np.array([r["stability_K512"] for r in rows])
        q = np.array([r[f"p@10_{lab}"] for r in rows])
        s = np.array(pair(p))
        L = len(q) - 1
        ly = np.arange(1, L)                                      # layers 1..L-1 (set_l defined for l < L)
        S = int(sel["selection"][lab]["S"]["layer"])             # the stability pick as select_layers.py made it
        oracle = int(q.argmax())
        set_pick = int(ly[np.argmax(s[ly])])
        out[p] = {"settledness": s.tolist(), "rho_stab_set": float(spearmanr(st[ly], s[ly])[0]),
                  "rho_stab_q": float(spearmanr(st[ly], q[ly])[0]), "S": S, "oracle": oracle,
                  "regret_S": float(q.max() - q[S]), "set_pick": set_pick, "regret_set": float(q.max() - q[set_pick]),
                  "churn_after_oracle": float(np.mean(1 - s[oracle:L])) if oracle < L else 0.0}
        log(p, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[p].items() if k != "settledness"})
    P = list(out.values())
    a = sum(r["rho_stab_set"] > abs(r["rho_stab_q"]) for r in P)
    reg, ch = np.array([r["regret_S"] for r in P]), np.array([r["churn_after_oracle"] for r in P])
    rho = float(spearmanr(ch, reg)[0])
    rng = np.random.default_rng(0)
    null = np.array([spearmanr(ch, rng.permutation(reg))[0] for _ in range(10000)])
    c = sum(abs(r["regret_set"] - r["regret_S"]) <= 0.01 for r in P)
    score = {"P2a": {"count": a, "of": len(P), "supported": a >= 8},
             "P2b": {"rho": rho, "p_one_sided": float((null >= rho).mean()), "supported": bool(rho > 0 and (null >= rho).mean() < 0.05)},
             "P2c": {"count": c, "of": len(P), "supported": c >= 8}, "killed": a < 8}
    (RESULTS / "settledness.json").write_text(json.dumps({"pairs": out, "score": score}, indent=1))
    print(json.dumps(score, indent=1))


if __name__ == "__main__":
    main()
