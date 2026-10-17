"""Score PREREG_quantization_lens.md Q1, Q2 and Q4 from saved results; Q3 lives in gq__qkl.json.
Writes results/quant_score.json. Lens layer ℓ = hidden_states[ℓ] = output of decoder layer ℓ−1."""
import json

import numpy as np

from common import RESULTS

BITS = {"gq_q8_0": 8, "gq_q6k": 6, "gq_q5km": 5, "gq_q4km": 4, "gq_q4_0": 4, "gq_qat4_0": 4, "gq_q3km": 3,
        "gq_q2k": 2, "gq_iq2m": 2}
PTQ = ["gq_q8_0", "gq_q6k", "gq_q5km", "gq_q4km", "gq_q4_0", "gq_q3km", "gq_q2k", "gq_iq2m"]


def main():
    q = json.loads((RESULTS / "gq_bf16__quantlens_all.json").read_text())
    rows = q["rows"]
    curve = {}
    for v in q["variants"]:
        rs = sorted((r for r in rows if r["variant"] == v), key=lambda r: r["layer"])
        c = {"rel_l2": np.array([r["rel_l2"] for r in rs])}
        for K in q["ks"]:
            c[f"norm_K{K}"] = np.array([r[f"flip_norm_K{K}"] for r in rs])
            c[f"raw_K{K}"] = np.array([1 - r[f"ceiling_K{K}"] * (1 - r[f"flip_norm_K{K}"]) for r in rs])
            c[f"changed_K{K}"] = np.array([r[f"changed_K{K}"] for r in rs])
        curve[v] = c

    def top_jumps(x, n):
        d = np.diff(x)                      # d[i] = x[i+1] − x[i], a jump *into* lens layer i+1
        return [int(i) + 1 for i in np.argsort(-d)[:n]]

    q1 = {v: {"bits": BITS[v],
              **{f"mean_{m}_K{K}": float(curve[v][f"{m}_K{K}"].mean()) for m in ("norm", "raw", "changed") for K in q["ks"]},
              "top2_jump_norm_K512": top_jumps(curve[v]["norm_K512"], 2),
              "top2_jump_raw_K512": top_jumps(curve[v]["raw_K512"], 2),
              "top4_jump_rel_l2": sorted(top_jumps(curve[v]["rel_l2"], 4))} for v in q["variants"]}
    le5 = [v for v in q["variants"] if BITS[v] <= 5]
    l15 = [v for v in le5 if 15 in q1[v]["top2_jump_norm_K512"]]
    ptq_mean = [q1[v]["mean_raw_K512"] for v in PTQ]
    q1_summary = {
        "raw_K512_mean_by_ptq_variant": dict(zip(PTQ, ptq_mean)),
        "raw_K512_monotone_in_bits": bool(all(np.diff([np.mean([q1[v]["mean_raw_K512"] for v in PTQ if BITS[v] == b])
                                                         for b in (8, 6, 5, 4, 3, 2)]) >= 0)),
        "fine_gt_coarse_raw": f"{sum(q1[v]['mean_raw_K512'] > q1[v]['mean_raw_K8'] for v in q['variants'])}/{len(q['variants'])}",
        "lens15_in_top2_norm_jumps_le5bit": f"{len(l15)}/{len(le5)}", "lens15_variants": l15,
        "rel_l2_top4_sets": sorted({tuple(q1[v]["top4_jump_rel_l2"]) for v in q["variants"]}),
    }

    ref = json.loads((RESULTS / "gq_bf16__select.json").read_text())["selection"]["l2"]
    q2 = {}
    for v in q["variants"]:
        s = json.loads((RESULTS / f"{v}__select.json").read_text())["selection"]["l2"]
        q2[v] = {"bits": BITS[v], "S": s["S"]["layer"], "oracle": s["oracle_layer"], "regret": s["S"]["regret"],
                 "ok": abs(s["S"]["layer"] - ref["S"]["layer"]) <= 1 and s["S"]["regret"] <= 0.01}
    ge4 = [v for v in q["variants"] if BITS[v] >= 4]
    fails = sum(not q2[v]["ok"] for v in ge4)

    a, b = curve["gq_qat4_0"]["norm_K512"], curve["gq_q4_0"]["norm_K512"]
    q4 = {"qat_lower_layers": f"{int((a < b).sum())}/{len(a)}", "supported": bool((a < b).mean() > 0.5),
          "retrieval_S_regret": {v: q2[v]["regret"] for v in ("gq_qat4_0", "gq_q4_0")}}

    out = {"Q1": {"per_variant": q1, "summary": q1_summary},
           "Q2": {"ref_S": ref["S"]["layer"], "per_variant": q2, "fails_ge4bit": fails, "killed": fails >= 2,
                  "pass_all": f"{sum(q2[v]['ok'] for v in q['variants'])}/{len(q['variants'])}"},
           "Q4": q4}
    (RESULTS / "quant_score.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"Q1": q1_summary, "Q2": {k: out["Q2"][k] for k in ("ref_S", "fails_ge4bit", "pass_all")}, "Q4": q4}, indent=1))


if __name__ == "__main__":
    main()
