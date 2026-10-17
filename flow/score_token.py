"""Score research/PREREG_token_lens.md hypotheses T1–T6 from results/tok__token_lens.json."""
import json

import numpy as np

from common import RESULTS


def main():
    d = json.loads((RESULTS / "tok__token_lens.json").read_text())
    rows = d["rows"]
    out = {}
    for m in ("base", "it", "it2bit"):
        R = sorted([r for r in rows if r["model"] == m], key=lambda r: r["layer"])
        L = len(R) - 1
        idv = np.array([r["id_K512"] for r in R]); nx = np.array([r["next_K512"] for r in R])
        ctx = np.array([r["ctx_K512"] for r in R]); st = np.array([r["stab_K512"] for r in R])
        pk = int(idv.argmax())
        t1 = pk <= 3 and (idv[pk] - idv[-1]) / idv[pk] >= 0.30
        t2 = int(nx.argmax()) >= 27
        t3 = idv[0] > nx[0] and nx[-1] > idv[-1]
        cpk = int(ctx.argmax())
        t4 = 9 <= cpk <= 27 and R[cpk]["ctx_K512_ci"][0] > 0
        cross = next((i for i in range(len(R)) if nx[i] > idv[i]), None)
        out[m] = {"T1": bool(t1), "id_peak_layer": pk, "id_peak": round(float(idv[pk]), 3), "id_last": round(float(idv[-1]), 3),
                  "T2": bool(t2), "next_peak_layer": int(nx.argmax()), "next_peak": round(float(nx.max()), 3),
                  "T3": bool(t3), "first_layer_next_gt_id": cross,
                  "T4": bool(t4), "ctx_peak_layer": cpk, "ctx_peak": round(float(ctx[cpk]), 4), "ctx_peak_ci": R[cpk]["ctx_K512_ci"],
                  "stab_peak_layer": int(st.argmax()), "stab_min_layer": int(st.argmin())}
        if m == "it2bit":
            deep = [r for r in R if r["layer"] >= 5]
            wins = sum(r["flip_by_band"]["rarer"] > r["flip_by_band"]["top100"] for r in deep)
            out[m]["T5"] = wins > len(deep) / 2
            out[m]["T5_layers_rarer_gt_top100"] = f"{wins}/{len(deep)}"
            out[m]["flip_mean_by_band"] = {b: round(float(np.mean([r["flip_by_band"][b] for r in deep])), 3) for b in ("top100", "101-1000", "rarer")}
            types = sorted(deep[0]["flip_by_type"])
            out[m]["flip_mean_by_type"] = {t: round(float(np.mean([r["flip_by_type"][t] for r in deep])), 3) for t in types}
            out[m]["flip_by_layer"] = [round(r["flip_K512"], 3) for r in R]
    (RESULTS / "token_score.json").write_text(json.dumps(out, indent=2))
    for m, v in out.items():
        print(m, json.dumps(v))
    for m in ("base", "it"):
        R = sorted([r for r in rows if r["model"] == m], key=lambda r: r["layer"])
        print(m, "layer: id / next / ctx / stab")
        for r in R[::3] + [R[-1]]:
            print(f"  L{r['layer']:02d} {r['id_K512']:.3f} {r['next_K512']:.3f} {r['ctx_K512']:.4f} {r['stab_K512']:.3f}")


if __name__ == "__main__":
    main()
