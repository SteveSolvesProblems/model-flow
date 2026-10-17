"""Assemble the data for the findings page (findings.html) from existing results; nothing is recomputed.
Writes results/findings_data.json.

python viewer/build_findings_data.py
"""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
load = lambda f: json.loads((R / f).read_text())
rnd = lambda x, n=3: np.round(np.asarray(x, dtype=float), n).tolist()

PAIR_NAMES = {"qwen35_08b": "Qwen3.5-0.8B", "qwen35_4b": "Qwen3.5-4B", "gemma4_e4b": "Gemma 4 E4B",
              "smollm2_17b": "SmolLM2-1.7B", "qwen3_17b": "Qwen3-1.7B", "qwen35_2b": "Qwen3.5-2B", "gemma4_e2b": "Gemma 4 E2B"}


def main():
    out = {}
    # 1. pre-registered layer selection (B1): regret of the label-free pick vs the last layer, per model x task
    b1 = load("robust_b1.json")
    rows = []
    for k, v in b1.items():
        if k == "summary":
            continue
        base, task = (k[:-5], "20 Newsgroups") if k.endswith("_20ng") else (k, "DBpedia")
        rows.append({"model": PAIR_NAMES.get(base, base), "task": task, "L": len(v["p10"]) - 1, "pick": v["S10"],
                     "regret_pick": round(100 * v["regret_S10"], 2), "regret_last": round(100 * v["regret_last"], 2)})
    out["b1"] = {"rows": sorted(rows, key=lambda r: r["regret_last"] - r["regret_pick"], reverse=True),
                 "summary": b1["summary"]}
    # post-hoc simple baselines (not pre-registered): fixed layers, mean shortfall in points over the same pairs
    reg = lambda v, l: 100 * (max(v["p10"]) - v["p10"][l])
    pairs = [v for k, v in b1.items() if k != "summary"]
    fixed = {f: float(np.mean([reg(v, int(round(f * (len(v["p10"]) - 1)))) for v in pairs]))
             for f in np.round(np.arange(0.30, 0.96, 0.01), 2)}
    fbest = min(fixed, key=fixed.get)
    out["b1"]["baselines"] = {"pick": round(float(np.mean([100 * v["regret_S10"] for v in pairs])), 2),
                              "last": round(float(np.mean([100 * v["regret_last"] for v in pairs])), 2),
                              "second_last": round(float(np.mean([reg(v, len(v["p10"]) - 2) for v in pairs])), 2),
                              "middle": round(fixed[0.5], 2), "best_fixed_depth": float(fbest),
                              "best_fixed": round(fixed[fbest], 2)}

    # 2. flow map: Gemma 4 E2B base, 1,500 articles. Bins = consensus K-means in each layer's 64-PC frame (not whitened),
    # chosen by the fixed rule in sankey_validity.py; per-layer repeatability (ARI of two independent consensus runs)
    D, SV = load("viewer_data.json"), load("sankey_validity.json")
    out["flow"] = {"l1": [d["l1"] for d in D["docs"]], "n_layers": D["n_layers"], "default_K": int(SV["best"].split("_K")[1]),
                   "frame": SV["best"].split("_K")[0],
                   "byK": {K: {"codes": SV["per_layer"][f"pca64_K{K}"]["codes"],
                               "repeat": rnd(SV["per_layer"][f"pca64_K{K}"]["cons"]),
                               "repeat_median": round(SV["summary"][f"pca64_K{K}"]["cons_median"], 3)} for K in (4, 8)},
                   "old_seed_ari_median": round(SV["summary"]["whiten64_K8"]["stab_median"], 3)}
    assert SV["best"].startswith("pca64"), SV["best"]
    from sklearn.metrics import adjusted_mutual_info_score as ami
    l1 = [d["l1"] for d in D["docs"]]
    for K in (4, 8):
        out["flow"]["byK"][K]["topic_ami"] = rnd([ami(l1, c) for c in SV["per_layer"][f"pca64_K{K}"]["codes"]])
    pe = load("prereg_scalar_e4b.json")
    out["e4b"] = {"across_cos": pe["across_cos"], "scalar": [0.056, 0.146, 0.781, 0.641, 0.486, 0.57, 0.484, 0.609, 0.223,
                  0.412, 0.59, 0.531, 0.762, 0.746, 0.711, 0.57, 0.676, 0.473, 0.566, 0.527, 0.441, 0.461, 0.142, 0.069,
                  0.447, 0.703, 0.715, 0.695, 0.613, 0.432, 0.703, 0.812, 0.848, 0.832, 0.84, 0.828, 0.84, 0.875, 0.871,
                  0.828, 0.785, 0.432], "S": {k: pe[k] for k in ("S1", "S2", "S3")}}

    # 3. heatmaps (same place / same neighbours) for six samples
    G = load("geometry_viewer.json")
    out["heat"] = {k: {"model": v["model"], "kind": v["kind"], "self_top1": v["heat"]["self_top1"],
                       "knn10": v["heat"]["knn10_overlap"], "chance_knn": v["chance"]["knn10_overlap"]}
                   for k, v in G.items()}
    # decoders and encoders: outlier-robust scaling (per-dimension median / MAD) where that run has neighbour data
    for key, probe in (("gemma_tok", "tok_base_bos"), ("qwen_tok", "tok_qwen3"), ("gemma_doc", "gemma4_e2b"),
                       ("qwen_doc", "qwen3_17b"), ("mbert_doc", "mbert_large"), ("laya_doc", "laya")):
        f = R / f"geometry_probe_{probe}_robust.json"
        rb = json.loads(f.read_text()) if f.exists() else {}
        if "knn10_overlap" in rb:
            out["heat"][key].update({"self_top1": rnd(rb["self_top1"]), "knn10": rnd(rb["knn10_overlap"]), "prep": "robust"})
        else:
            print("heat: no robust neighbour data for", probe, "(centered only)")
    PY = load("pythia_geometry.json")
    for st, nm in (("0", "Pythia-410M before training (step 0)"), ("143000", "Pythia-410M after training")):
        out["heat"][f"pythia_{st}"] = {"model": nm, "kind": "document", "self_top1": rnd(PY[st]["self_top1"]),
                                       "knn10": rnd(PY[st]["knn10_overlap"]), "chance_knn": 10 / 2999}
    # more decoders (robust), and one plain (centered only) view to show what robust scaling removes
    for key, probe, nm in (("smollm2_doc", "smollm2_17b_robust", "SmolLM2-1.7B"), ("qwen35_doc", "qwen35_2b_robust", "Qwen3.5-2B"),
                           ("qwen_doc_plain", "qwen3_17b", "Qwen3-1.7B, plain (no robust scaling)")):
        pr = load(f"geometry_probe_{probe}.json")
        out["heat"][key] = {"model": nm, "kind": "document", "self_top1": rnd(pr["self_top1"]),
                            "knn10": rnd(pr["knn10_overlap"]), "chance_knn": pr["chance_knn10"]}
    # gemma_tok: re-extracted with <bos> (tok_base_bos; NOTE_geometry.md, 2026-10-05/06)
    order = ["pythia_0", "pythia_143000", "gemma_doc", "gemma_tok", "qwen_doc", "qwen_doc_plain", "qwen_tok",
             "qwen35_doc", "smollm2_doc", "mbert_doc"]
    out["heat"] = {k: out["heat"][k] for k in order}

    # 4. can a fitted rotation find the article again? within one model vs across models (geometry_crossmodel.py)
    gq, ge, qs = (load(f"geometry_crossmodel_{f}.json") for f in ("gemma4_e2b__qwen3_17b", "gemma4_e2b__gemma4_e4b",
                                                                   "qwen3_17b__smollm2_17b"))
    lm = load("geometry_linmap_gemma4_e2b.json")
    out["rotation"] = [
        {"label": "Gemma E2B layer 9 → 18, no map", "v": round(lm["identity_top1"][9][18], 3), "kind": "none"},
        {"label": "Gemma E2B layer 9 → 18, rotation", "v": round(lm["rotation_top1"][9][18], 3), "kind": "within"},
        {"label": "Gemma E2B layer 14 → 21, rotation", "v": round(ge["within"]["14-21"]["rotation"], 3), "kind": "within"},
        {"label": "Gemma E2B 14 → Gemma E4B 17", "v": round(ge["cross"]["14-17"]["rotation"], 3), "kind": "cross"},
        {"label": "Gemma E2B 14 → Qwen3 22", "v": round(gq["cross"]["14-22"]["rotation"], 3), "kind": "cross"},
        {"label": "Qwen3 22 → SmolLM2 19", "v": round(qs["cross"]["22-19"]["rotation"], 3), "kind": "cross"},
        {"label": "Gemma E2B 0 → Qwen3 0 (word embeddings)", "v": round(gq["cross"]["0-0"]["rotation"], 3), "kind": "cross"},
        {"label": "Gemma E2B 14 → 21, pairs shuffled", "v": round(ge["shuffled"]["14-21"]["rotation"], 4), "kind": "null"}]

    # 5. meaning before prediction (tokens): relations vs prediction, each compared with the last layer
    mp = []
    for name, label in (("tok_base", "Gemma 4 E2B"), ("tok_qwen3", "Qwen3-1.7B")):
        p, r = load(f"geometry_probe_{name}.json"), load(f"geometry_readout_{name}.json")
        T, K = np.array(p["self_top1"]), np.array(p["knn10_overlap"])
        mp.append({"model": label, "same_place_as_last": rnd(T[:, -1]), "neighbours_as_last": rnd(K[:, -1]),
                   "prediction_agree": rnd(r["agree_mapped"]), "logit_lens_agree": rnd(r["agree_logit"])})
    out["meaning_vs_prediction"] = mp

    # 6. Gemma tracks by decoder layer k (k makes the change from cached state k to k+1)
    w, h = load("weight_structure_gemma4_e2b.json"), load("attention_heads_gemma4_e2b.json")
    pg = load("geometry_probe_gemma4_e2b.json")
    C = np.array(pg["cos_centered"])
    L = w["n_layers"]
    hb = load("heads_vs_blocks.json")["gemma4_e2b"]["blocks"]
    out["tracks"] = {"n_layers": L,
                     "turn": rnd([1 - C[k, k + 1] for k in range(L)]),
                     "layer_scalar": rnd(w["layer_scalar"]),
                     "full_attention": [t == "full_attention" for t in w["layer_types"]],
                     "kv_shared_from": L - w["num_kv_shared_layers"],
                     "induction_max": rnd([max(l["induction"]) for l in h["layers"]], 2),
                     "sink_mean": rnd([float(np.mean(l["sink"])) for l in h["layers"]], 2),
                     "write_norm": rnd([p["write_norm"] for p in w["per_layer"]], 1),
                     "blocks": [b["layers"] for b in hb],
                     "write_free": rnd(load("geometry_flow_noscale_gemma4_e2b.json")["update_rel"], 2)}
    try:
        ba = load("block_ablation_gemma4_e2b.json")["results"]
        out["block_ablation"] = {k: {kk: v[kk] for kk in ("turn_13_15", "turn_12_16", "kl", "top1") if kk in v}
                                 for k, v in ba.items()}
    except FileNotFoundError:
        out["block_ablation"] = None
    # 7. knockouts and micro-macro summary numbers used in text
    ha = load("head_ablation_gemma4_e2b.json")["results"]
    out["knockouts"] = {k: {kk: v.get(kk) for kk in ("kl", "top1", "ind_loss", "turn_13_15")} for k, v in ha.items()}
    # references cited on the page (viewer/refs.json, shared with the lab notebook)
    import re
    here = Path(__file__).resolve().parent
    page = "".join((here / f).read_text() for f in ("post.html", "stability.html", "archive/findings.html") if (here / f).exists())
    allrefs = json.loads((Path(__file__).resolve().parent / "refs.json").read_text())
    used = sorted(set(re.findall(r'href="#r-([a-z0-9]+)"', page)))
    missing = [k for k in used if k not in allrefs]
    assert not missing, f"cited but not in refs.json: {missing}"
    out["refs"] = {k: allrefs[k] for k in used}
    out["example"] = json.loads((R / "post_example.json").read_text())
    (R / "findings_data.json").write_text(json.dumps(out, separators=(",", ":")))
    print("findings_data.json", round((R / "findings_data.json").stat().st_size / 1e6, 2), "MB",
          "| block ablation:", "yes" if out["block_ablation"] else "pending")


if __name__ == "__main__":
    main()
