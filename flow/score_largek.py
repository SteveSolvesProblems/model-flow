"""Score PREREG_large_k.md K1–K4 from saved results. Writes results/largek_score.json."""
import json

import numpy as np

from common import RESULTS

HELD = ["qwen35_08b", "qwen35_08b_20ng", "qwen35_4b", "qwen35_4b_20ng", "gemma4_e4b", "gemma4_e4b_20ng", "smollm2_17b",
        "smollm2_17b_20ng", "qwen3_17b_20ng", "qwen35_2b_20ng", "gemma4_e2b_20ng"]
BIG = ["qwen35_2b_big", "gemma4_e2b_big"]
KS = [512, 1024, 2048, 4096]


def main():
    k1 = {}
    reg = {K: [] for K in (512, 1024, 2048)}
    for p in HELD:
        sel = json.loads((RESULTS / f"{p}__largek_slim.json").read_text())["selection"]
        for K in reg:
            s = sel[f"K{K}"]
            reg[K].append(s["regret_l2"] if "regret_l2" in s else s["regret_fine"])   # primary label: DBpedia l2 / 20NG fine
    base = np.array(reg[512])
    for K in (1024, 2048):
        r = np.array(reg[K])
        d = r.mean() - base.mean()
        verdict = "helps" if d < -0.002 and (r <= base).sum() >= 8 else "hurts" if d > 0.005 else "neutral"
        k1[f"K{K}"] = {"mean_regret": float(r.mean()), "delta_vs_512": float(d), "le_512_pairs": int((r <= base).sum()),
                       "verdict": verdict}
    k1["K512_mean_regret"] = float(base.mean())

    k2, k3, k4 = {}, {}, {}
    for m in BIG:
        j = json.loads((RESULTS / f"{m}__largek.json").read_text())
        rows, sel, cl = j["rows"], j["selection"], j["classifiers"]
        by = {(r["N"], r["layer"]): r for r in rows}
        layers = sorted({r["layer"] for r in rows})
        rises = {K: int(sum(by[(120000, l)][f"stab_K{K}"] > by[(24000, l)][f"stab_K{K}"] for l in layers)) for K in KS}
        k2[m] = {"layers": len(layers), "stab_rises_N24k_to_120k": rises,
                 "S4096_regret_120k": sel["N120000_K4096"]["regret_l2"], "S512_regret_120k": sel["N120000_K512"]["regret_l2"],
                 "S_layers_120k": {K: sel[f"N120000_K{K}"]["layer"] for K in KS}, "oracle_120k": sel["N120000_oracle"]["l2"]}
        k2[m]["holds"] = all(rises[K] == len(layers) for K in (1024, 2048, 4096)) and \
            k2[m]["S4096_regret_120k"] <= k2[m]["S512_regret_120k"]
        c = cl["N120000_S"]
        acc = [c[f"cellacc_l2_K{K}"] for K in KS]
        k3[m] = {"layer": c["layer"], "cellacc_l2": dict(zip(KS, acc)), "knn10_l2": c["knn10_l2"], "probe_l2": c["probe_l2"],
                 "rises_with_K": bool(np.all(np.diff(acc) > 0)), "below_probe_all_K": bool(max(acc) < c["probe_l2"]),
                 "last_layer": {k: v for k, v in cl["N120000_last"].items() if k.endswith("_l2") or k == "layer"}}
        S = c["layer"]
        r = by[(120000, S)]
        nmi = [r[f"nmi_l3_K{K}"] for K in KS]
        k4[m] = {"layer": S, "nmi_l3": dict(zip(KS, nmi)), "rises_to_1024": bool(nmi[1] > nmi[0]),
                 "nest": {K: r[f"nest_K{K}"] for K in KS[1:]}, "nest_1024_ge_0.8": bool(r["nest_K1024"] >= 0.8)}
    out = {"K1": k1, "K2": {"per_model": k2, "supported": all(v["holds"] for v in k2.values())},
           "K3": {"per_model": k3, "supported": all(v["rises_with_K"] and v["below_probe_all_K"] for v in k3.values())},
           "K4": {"per_model": k4, "supported": all(v["rises_to_1024"] and v["nest_1024_ge_0.8"] for v in k4.values())}}
    (RESULTS / "largek_score.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
