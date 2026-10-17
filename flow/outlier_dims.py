"""research/PREREG_outlier_dims.md (OD1-OD3): massive dimensions and the last-layer gap.
Writes results/outlier_dims.json. CPU, existing caches.

python outlier_dims.py
"""
import json

import numpy as np

from common import RESULTS, load_cache, set_threads
from lens import frame, layers_of, log
from partition import assign, kmeans
from robustness import HELD, p_at_10, pick, topical
from token_lens import fast_nmi

THR = 20
THR_DESC = (10, 50)


def massive(Xtr, thr):
    m = np.abs(Xtr).mean(0)
    return np.flatnonzero(m > thr * np.median(m))


def main():
    set_threads()
    out = {}
    for p in HELD:
        lab, p10 = topical(p)
        names = layers_of(p)
        ytr, yte = load_cache(names[0], "train", lab)[1], load_cache(names[0], "test", lab)[1]
        rows = []
        for nm in names:
            Xtr, Xte = load_cache(nm, "train"), load_cache(nm, "test")
            idx = massive(Xtr, THR)
            mu = Xtr.mean(0)
            A, B = Xtr - mu, Xte - mu
            Ar, Br = A.copy(), B.copy()
            Ar[:, idx] = 0
            Br[:, idx] = 0
            med = np.median(Xtr, 0)
            mad = np.median(np.abs(Xtr - med), 0) + 1e-6
            r = {"n_massive": int(len(idx)), "massive_dims": idx.tolist(),
                 **{f"n_massive_{t}": int(len(massive(Xtr, t))) for t in THR_DESC},
                 "centered": p_at_10(A, B, ytr, yte), "removed": p_at_10(Ar, Br, ytr, yte),
                 "robust": p_at_10((Xtr - med) / mad, (Xte - med) / mad, ytr, yte)}
            Xtr_r, Xte_r = Xtr.copy(), Xte.copy()
            Xtr_r[:, idx] = 0
            Xte_r[:, idx] = 0
            Ftr, Fte = frame(Xtr_r, Xte_r)
            del Xtr_r, Xte_r
            c = [assign(Fte, kmeans(Ftr, 512, seed=s, nredo=1))[0] for s in (0, 1)]
            r["stability_removed"] = float(fast_nmi(c[0], c[1]))
            rows.append(r)
            log("od", p, nm, {k: v for k, v in r.items() if k != "massive_dims"})
        sel = json.loads((RESULTS / f"{p}__select.json").read_text())["rows"]
        S = pick([x["stability_K512"] for x in sel])
        S_rm = pick([x["stability_removed"] for x in rows])
        cen = np.array([x["centered"] for x in rows])
        res = {"S": S, "S_removed": S_rm, "regret_S": float(cen.max() - cen[S]), "regret_S_removed": float(cen.max() - cen[S_rm]),
               "has_massive_last": rows[-1]["n_massive"] > 0, "rows": rows}
        for k in ("centered", "removed", "robust"):
            v = np.array([x[k] for x in rows])
            res[f"gap_{k}"] = float(v.max() - v[-1])
            res[f"best_{k}"] = int(v.argmax())
        out[p] = res
        log("od", p, {k: v for k, v in res.items() if k != "rows"})
    M = [p for p in HELD if out[p]["has_massive_last"]]
    N = [p for p in HELD if not out[p]["has_massive_last"]]

    def closes(p, k):
        g0 = out[p]["gap_centered"]
        return g0 > 1e-9 and (g0 - out[p][f"gap_{k}"]) / g0 >= 0.5
    summ = {"massive_pairs": M, "other_pairs": N,
            "OD1_massive_closed": sum(closes(p, "removed") for p in M),
            "OD1_other_unchanged": sum(abs(out[p]["gap_removed"] - out[p]["gap_centered"]) < 0.01 for p in N),
            "OD2_unchanged": sum(abs(out[p]["regret_S_removed"] - out[p]["regret_S"]) <= 0.005 for p in HELD),
            "OD3_closed": sum(closes(p, "robust") for p in HELD)}
    summ["OD1"] = bool(M) and summ["OD1_massive_closed"] >= 2 * len(M) / 3 and (not N or summ["OD1_other_unchanged"] >= 2 * len(N) / 3)
    summ["OD2"] = summ["OD2_unchanged"] >= 9
    summ["OD3"] = summ["OD3_closed"] >= 6
    out["summary"] = summ
    (RESULTS / "outlier_dims.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
