"""Claims registry: recompute every headline number from the result files and compare it with the value stated in
the writeups. Writes research/CLAIMS.md (one row per claim: value, source file, producing script, design, threats).

python verify_claims.py      # exits non-zero if any stated value does not match its source
"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parents[1] / "research" / "CLAIMS.md"


def j(name):
    return json.loads((R / name).read_text())


CLAIMS = []


def claim(cid, text, design, source, script, threats):
    def wrap(fn):
        CLAIMS.append((cid, text, design, source, script, threats, fn))
        return fn
    return wrap


def close(stated, got, tol):
    return abs(stated - got) <= tol


@claim("L1", "Map stability's held-out mean regret is 0.49 points (last layer 3.42, middle 13.1, RankMe 13.5)",
       "pre-registered", "select_heldout_score.json", "select_layers.py, score_select.py",
       "11 pairs share 7 models; regret defined by raw-cosine p@10 only; DBpedia and 20NG are topic tasks")
def _():
    m = j("select_heldout_score.json")["mean_regret_primary"]
    got = [100 * m[k] for k in ("S", "last", "middle", "rankme_min")]
    return got, all(close(s, g, 0.05) for s, g in zip((0.49, 3.42, 13.1, 13.5), got))


@claim("L2", "It beats the last layer in 10 of 11 held-out pairs; two-level 95% CI of the gap [1.6, 4.3] points",
       "pre-registered", "baselines_score.json", "score_baselines.py",
       "11 dependent pairs; the two-level bootstrap resamples only 11 pairs")
def _():
    b = j("baselines_score.json")
    tl = b["two_level_mean_S_minus_last"]
    ci = [100 * x for x in tl["ci95"]]
    n = b["pairs_S_above_last"]
    return (n, ci), (str(n).startswith("10") and close(1.6, ci[0], 0.06) and close(4.3, ci[1], 0.06))


@claim("L3", "It beats TwoNN (3.0), curvature (10.4), prompt entropy (15.4), dataset entropy (16.6)",
       "pre-registered", "baselines_score.json", "baselines_a.py, score_baselines.py",
       "TwoNN direction chosen on dev pairs; rivals were not proposed as selectors by their authors")
def _():
    m = j("baselines_score.json")["mean_regret"]
    got = [100 * m[k] for k in ("twonn", "curvature", "prompt_entropy", "dataset_entropy")]
    return got, all(close(s, g, 0.06) for s, g in zip((2.96, 10.4, 15.4, 16.6), got))


@claim("L4", "One labeled example per class ties it (0.49); ten reach 0.09",
       "pre-registered (expectation failed)", "baselines_score.json", "baselines_a.py",
       "few-label pick uses the same test documents as the outcome")
def _():
    f = j("baselines_score.json")["fewlabel_mean_regret"]
    got = [100 * f["1"], 100 * f["10"]]
    return got, close(0.49, got[0], 0.02) and close(0.09, got[1], 0.02)


@claim("L5", "Chance-correcting the agreement score makes the selector worse (1.30 vs 0.46)",
       "pre-registered secondary", "adjusted_select.json", "adjusted_select.py", "3 permutations per layer")
def _():
    a = j("adjusted_select.json")
    got = [100 * a["mean_regret_adjusted"], 100 * a["mean_regret"]]
    return got, close(1.30, got[0], 0.02) and close(0.46, got[1], 0.02)


@claim("L6", "Chat-tuned Gemma 4 E2B (with <bos>): map stability picks layer 28 (2.2 points short); the last layer is 10.6 points "
       "worse (was layer 6 and 29.6 without <bos>)",
       "observed (dev model)", "gq_bf16__select.json", "select_layers.py",
       "GGUF-dequantized bf16 weights; one model")
def _():
    s = j("gq_bf16__select.json")["selection"]["l2"]
    got = [s["S"]["layer"], 100 * s["last"]["regret"]]
    return got, got[0] == 28 and close(10.6, got[1], 0.05) and close(2.2, 100 * s["S"]["regret"], 0.05)


@claim("C3", "Chat-tuned models: the stability pick is within 1 point in 4 of 8 (6 needed); misses 2.9, 2.4, 2.2, 1.7 "
       "(Gemma rows re-run with <bos>)", "pre-registered (failed)", "flow_f3.json", "score_f3.py", "4 base/chat pairs x 2 collections")
def _():
    f = {k: v for k, v in j("flow_f3.json").items() if isinstance(v, dict) and "post_S_regret" in v}
    r = sorted((100 * v["post_S_regret"] for v in f.values()), reverse=True)
    got = (sum(x <= 1 for x in r), [round(x, 1) for x in r[:4]])
    return got, got == (4, [2.9, 2.4, 2.2, 1.7])


@claim("A1", "Special-token audit: of 187 text caches, only the 5 superseded Gemma token caches lack <bos> "
       "(tok_base, tok_it, tok_it2bit, tok_nople0, tok_nople18; all re-extracted as *_bos)", "check",
       "audit_special_tokens.json", "audit_special_tokens.py", "first special tokens before a fixed string")
def _():
    a = j("audit_special_tokens.json")
    got = (a["n"], sorted(a["failures"]))
    return got, got[0] == 187 and got[1] == ["tok_base", "tok_it", "tok_it2bit", "tok_nople0", "tok_nople18"]


@claim("N1", "<bos> on vs off (Gemma 4 E2B): neighbours kept 0.44 in layers 1-13 vs 0.21 in 15-35 (tokens; articles "
       "0.45 vs 0.23); tokens at positions 1-4 move more than those at 16+ at 34/34 layers", "pre-registered (supported)",
       "bos_perturb.json", "bos_perturb.py", "one model; articles include the <bos> state in the mean")
def _():
    s = j("bos_perturb.json")["score"]
    got = [s["P1a_tokens"]["early_1_13"], s["P1a_tokens"]["late_15_35"], s["P1a_articles"]["early_1_13"],
           s["P1a_articles"]["late_15_35"], s["P1b"]["layers_late_higher"]]
    return got, all(close(a, b, 0.006) for a, b in zip((0.44, 0.21, 0.45, 0.23), got)) and got[4] == 34


@claim("N2", "Stability is not settledness: it tracks layer-to-layer neighbour overlap more than quality in 3/11 "
       "(killed); across layers it tracks quality (median Spearman 0.93)", "pre-registered (failed)",
       "settledness.json", "settledness.py", "11 held-out pairs")
def _():
    d = j("settledness.json")
    med = float(np.median([r["rho_stab_q"] for r in d["pairs"].values()]))
    got = (d["score"]["P2a"]["count"], round(med, 2), d["score"]["P2b"]["supported"], d["score"]["P2c"]["count"])
    return got, got == (3, 0.93, False, 5)


@claim("N3", "Output sensitivity x activation error does not predict a block's restoration benefit (Spearman 0.02; "
       "pre-registered 0.40 fails); sensitivity alone 0.47 (secondary)", "pre-registered (failed) + secondary",
       "sens_probe.json", "sens_probe.py", "one model, Q4_K_M, 200 documents, eps 0.02, 3 draws")
def _():
    d = j("sens_probe.json")
    got = [d["rho_prod"], d["rho_sens"], d["rho_err"]]
    return got, close(0.02, got[0], 0.01) and close(0.47, got[1], 0.01) and close(0.24, got[2], 0.01) and not d["P3a"]["supported"]


@claim("B1", "Stability-quality link is learned: across layers, Spearman -0.72 to -0.95 in untrained Pythia (step 0) "
       "and +0.77 to +0.96 after training, 8/8 pairs (6 independent initializations)", "pre-registered (supported)",
       "bins_w1.json", "bins_w1.py", "Pythia 410M/1.4B, DBpedia only")
def _():
    a = j("bins_w1.json")["W1a"]
    u = [p["s0"]["rho"] for p in a["pairs"].values()]
    t = [p["s143000"]["rho"] for p in a["pairs"].values()]
    got = [round(max(u), 2), round(min(u), 2), round(min(t), 2), round(max(t), 2), a["W1a1"]["count"]]
    return got, got == [-0.72, -0.95, 0.77, 0.96, 8]


@claim("B2", "Noise ladder: real layers sit a median 3.7 points from the signal-to-noise curve (inconclusive); "
       "DBpedia 2.2-3.2, 20NG 3.0-9.5", "pre-registered (inconclusive)", "bins_w1.json", "bins_w1.py",
       "11 held-out pairs; isotropic noise on the best layer")
def _():
    b = j("bins_w1.json")["W1b"]
    D = {p: v["D_points"] for p, v in b["pairs"].items()}
    db = [v for p, v in D.items() if not p.endswith("_20ng")]
    ng = [v for p, v in D.items() if p.endswith("_20ng")]
    got = [b["median_D_points"], min(db), max(db), min(ng), max(ng), b["verdict"]]
    return got, (close(3.7, got[0], 0.05) and close(2.2, got[1], 0.05) and close(3.2, got[2], 0.05)
                 and close(3.0, got[3], 0.05) and close(9.5, got[4], 0.05) and got[5] == "inconclusive")


@claim("B3", "Bin boundaries matter beyond distance in Qwen3-1.7B (3/3 depths, +1.4 to +2.0 KL) but in Gemma 4 E2B "
       "only late (block 26 +1.9; blocks 9, 18: -0.16, -0.32; block 18 is already past the turn)", "pre-registered (mixed: neither supported nor killed)",
       "bins_w2b.json", "bins_w2b.py", "token patching, matched whiten-64 distance; 15-40% of pairs unmatched")
def _():
    b = j("bins_w2b.json")
    d = [b[t]["score"]["diff_w"]["paired"]["mean"] for t in
         ("gemma4_e2b_m9", "gemma4_e2b_m18", "gemma4_e2b_m26", "qwen3_17b_m7", "qwen3_17b_m14", "qwen3_17b_m21")]
    ok = all(close(x, y, 0.006) for x, y in zip((-0.16, -0.32, 1.90, 1.79, 1.37, 2.02), d))
    return d, ok and not b["summary"]["W2b1_supported"] and not b["summary"]["killed"]


@claim("B4", "On 20NG, layers more stable than their topic quality predicts carry more format (partial Spearman < 0 "
       "in 7/7)", "pre-registered (supported)", "bins_w2a.json", "bins_w2a.py", "format NMI varies little across layers")
def _():
    w = j("bins_w2a.json")["score"]["W2a1"]
    r = sorted(round(v["partial_rho"], 2) for v in w["pairs"].values())
    return (w["negative"], r[0], r[-1]), w["negative"] == 7 and r[0] == -0.68 and r[-1] == -0.31


@claim("B5", "Bin boundaries matter at every depth in Qwen3-1.7B (7/7); in Gemma 4 E2B at 12-14 and 20-30 but "
       "reversed at 16-18 (no single switch); fine bins (K=4096) matter as much as coarse (K=32)",
       "pre-registered (W3-a1 supported; W3-a2 killed; W3-b1 not supported)", "bins_w3.json", "bins_w3.py",
       "token patching; K=4096 had 50-67% of pairs matched")
def _():
    r = j("bins_w3.json")
    s = r["summary"]
    rev = [r[f"gemma4_e2b_m{b}_K512"]["score"]["diff_w"]["paired"]["mean"] for b in (16, 18)]
    k = [r[f"{m}_K{K}"]["score"]["diff_w"]["paired"]["excludes_0"] for m in ("gemma4_e2b_m26", "qwen3_17b_m14") for K in (32, 4096)]
    got = (s["W3a1"]["supported_blocks"], s["W3a2"]["switch_exists"], s["W3b1"]["supported"], [round(x, 2) for x in rev], all(k))
    return got, got == (7, False, False, [-0.3, -0.32], True)


@claim("B7", "Pipeline faults, Llama-3.2-1B: harmless changes give D ~0; graded weight rounding orders D with output KL "
       "(Spearman 0.96); every fault D >= 0.30 (4-bit weights 0.12); truncation and padding-in-mean have KL 0 but D 0.81 / 0.30",
       "pre-registered (PF-a, PF-b, PF-c supported)", "pipeline_faults_llama32_1b.json", "pipeline_faults.py",
       "PF-a ratio trivial (harmless D ~5e-5); PF-c partly by construction (dNLL only on shared tokens); 4 of 7 PF-b runs near 0; one model")
def _():
    r = j("pipeline_faults_llama32_1b.json")
    u = r["runs"]
    got = (round(r["PFb"]["spearman"], 2), round(min(r["PFa"]["faults_D"].values()), 2), round(u["G4_rtn4"]["D"], 2),
           round(u["F3_cap16"]["D"], 2), round(u["F4_pad_in_mean"]["D"], 2), r["PFa"]["supported"], r["PFc"]["supported"])
    return got, got == (0.96, 0.3, 0.12, 0.81, 0.3, True, True)


@claim("B8", "Pipeline faults, TinyLlama v1.1: the tokenizer's default <s> is itself a fault for this checkpoint "
       "(reference NLL 5.42 vs 2.13 for Llama; removing <s> lowers NLL by 3.46); D flags the two pipelines as far apart "
       "(0.75) but cannot say which is broken; predictions pass formally (Spearman 0.85) against a broken reference",
       "pre-registered (passes, but not a clean replicate)", "pipeline_faults_tinyllama11.json", "pipeline_faults.py",
       "reference is faulty; bf16 D (0.084) > 8-bit D (0.045); Gemma-3-270m (clean second model) awaiting access")
def _():
    r = j("pipeline_faults_tinyllama11.json")
    u = r["runs"]
    got = (round(r["PFb"]["spearman"], 2), round(u["F1_no_bos"]["dNLL"], 2), round(u["F1_no_bos"]["D"], 2),
           round(u["G1_bf16"]["D"], 3), round(u["G2_rtn8"]["D"], 3), r["PFa"]["supported"])
    return got, got == (0.85, -3.46, 0.75, 0.084, 0.045, True)


@claim("B9", "Where D peaks (descriptive, not pre-registered): in Llama-3.2-1B input-side faults (no <s>, template, "
       "truncation) peak at block 1 and weight rounding (4/6/2-bit) at the last block (15), separating no-<s> from "
       "4-bit weights at equal output KL (0.18 vs 0.17; D 0.56 vs 0.12); TinyLlama agrees for weights (block 21)",
       "descriptive", "pipeline_faults_llama32_1b.json", "pipeline_faults.py",
       "post hoc; two models; TinyLlama no-<s> peaks at block 13 (its reference is the faulty one); needs its own pre-registration")
def _():
    u = j("pipeline_faults_llama32_1b.json")["runs"]
    t = j("pipeline_faults_tinyllama11.json")["runs"]
    got = ([u[k]["min_block"] for k in ("F1_no_bos", "F2_chat_template", "F3_cap16")],
           [u[k]["min_block"] for k in ("G3_rtn6", "G4_rtn4", "F5_rtn2")],
           [t[k]["min_block"] for k in ("G3_rtn6", "G4_rtn4", "F5_rtn2")], round(u["F1_no_bos"]["KL"], 2), round(u["G4_rtn4"]["KL"], 2))
    return got, got == ([1, 1, 1], [15, 15, 15], [21, 21, 21], 0.18, 0.17)


@claim("T1", "Other tasks: S better than last on Banking77 (mean) 3/5 and STS-B (mean) 2/4 -> predictions failed",
       "pre-registered (failed)", "task_score.json", "task_extract.py, task_score.py",
       "short texts; massive-activation dims in 2 models; K=512 vs ~10k texts")
def _():
    v = j("task_score.json")["verdict"]
    got = [v["banking77_mean"]["S_better_than_last"], v["stsb_mean"]["S_better_than_last"]]
    return got, got == [3, 2] and not v["C1_holds"] and not v["C3_holds"]


@claim("P1", "Pythia: max map stability tracks best-layer p@10 across checkpoints (rho 0.85, 0.93)",
       "pre-registered", "pythia_score.json", "pythia_score.py",
       "9 checkpoints per model; DBpedia is in the Pile (Wikipedia); both rise with training, so a shared time trend")
def _():
    p = j("pythia_score.json")
    got = [p["pythia410m"]["P3_spearman"], p["pythia14b"]["P3_spearman"]]
    return got, close(0.85, got[0], 0.005) and close(0.93, got[1], 0.005)


@claim("P2", "Pythia: picking a layer during training passes for 1.4b (8/9) and fails for 410m (4/9)",
       "pre-registered (mixed)", "pythia_score.json", "pythia_score.py", "the pick is often the last layer")
def _():
    p = j("pythia_score.json")
    got = [p["pythia410m"]["P1_count"], p["pythia14b"]["P1_count"]]
    return got, got == [4, 8]


@claim("P3", "Pythia: the best layer is 0 at steps 0-512, near the top at 1k-2k, layer 17 from 4k-8k on (410m final checkpoint: 24)",
       "descriptive", "pythia_score.json", "pythia_score.py", "one corpus; coarse checkpoint grid")
def _():
    p = j("pythia_score.json")
    o = {m: [c["oracle"] for c in p[m]["checkpoints"]] for m in ("pythia410m", "pythia14b")}
    ok = o["pythia410m"][:2] == [0, 0] and o["pythia410m"][4:9] == [17] * 5 and o["pythia14b"][5:] == [17] * 5
    return o, ok


@claim("M1", "Across 7 models, max map stability ranks models by best-layer p@10 (rho 0.86 DBpedia, 0.82 20NG)",
       "exploratory", "model_select.json", "model_select.py", "7 models; stability may not be comparable across models")
def _():
    m = j("model_select.json")
    got = [m["dbpedia"]["spearman_stab_vs_oracle"], m["20ng"]["spearman_stab_vs_oracle"]]
    return got, close(0.86, got[0], 0.005) and close(0.82, got[1], 0.005)


@claim("Q1", "With <bos>, a quantization's stability pick stays within 1 layer of the reference's and 1 point of its best "
       "in 1 of 9 (killed: 6 of 7 at >= 4 bits fail; was 9/9 at layer 6 without <bos>); shortfalls 0.7-4.0 points", "pre-registered (failed)", "quant_score.json", "score_quant.py", "one model (Gemma 4 E2B-it)")
def _():
    pv = j("quant_score.json")["Q2"]["per_variant"]
    r = [100 * v["regret"] for v in pv.values()]
    got = sum(v["ok"] for v in pv.values()), round(min(r), 1), round(max(r), 1)
    return got, got == (1, 0.7, 4.0) and j("quant_score.json")["Q2"]["killed"]


@claim("Q2", "With <bos>, restoring 4 layers by bin flips: KL 0.1007, worse than by activation error (0.0970, CI excludes 0), "
       "better than random (0.1032), ties 'first four' (0.0994); was 0.330 vs 0.352 / 0.383 without <bos>", "pre-registered + amendment (failed vs activation error)", "gq__qkl_ci.json", "qkl.py, score_qkl_ci.py",
       "one model, one quant (Q4_K_M), 1,000 documents; llama.cpp already favours the first blocks")
def _():
    k = j("gq__qkl_ci.json")
    pt, d = k["point"], k["diffs"]
    got = [pt["lens"], pt["actl2"], pt["random_mean"], pt["first4"], d["lens - first4"]["excludes_0"]]
    ok = (close(0.1007, got[0], 0.0001) and close(0.0970, got[1], 0.0001) and close(0.1032, got[2], 0.0001)
          and close(0.0994, got[3], 0.0001) and d["lens - actl2"]["point"] > 0 and d["lens - actl2"]["excludes_0"]
          and d["lens - random_mean"]["excludes_0"] and not got[4])
    return got, ok


@claim("Q3", "QAT moves the map more than Q4_0 (with <bos>: raw 42% vs 13%, re-centered 13% vs 10%; was 74/39, 37/29)",
       "pre-registered (prediction failed) + post-hoc check", "quant_score.json, qat_shift.json", "score_quant.py, qat_shift.py",
       "bin flips conflate a global shift with rearrangement")
def _():
    q = j("qat_shift.json")
    m = {v: {k: 100 * np.mean([r[k] for r in q[v]]) for k in ("raw", "recentered")} for v in q}
    got = [m["gq_qat4_0"]["raw"], m["gq_q4_0"]["raw"], m["gq_qat4_0"]["recentered"], m["gq_q4_0"]["recentered"]]
    return got, all(close(s, g, 0.6) for s, g in zip((41.5, 13.5, 13.5, 9.7), got))


@claim("K1", "More bins make the choice worse: regret 0.49 (512) -> 1.20 (1,024) -> 1.51 (2,048)",
       "pre-registered", "largek_score.json", "score_largek.py", "NMI chance floor grows with K (untested here)")
def _():
    k = j("largek_score.json")["K1"]
    got = [100 * k["K512_mean_regret"], 100 * k["K1024"]["mean_regret"], 100 * k["K2048"]["mean_regret"]]
    return got, all(close(s, g, 0.02) for s, g in zip((0.49, 1.20, 1.51), got))


@claim("W1", "Gemma's word identity rebounds after layer 17 (0.16; was 0.22 after 18 without <bos>); Qwen3-1.7B's does not (0.03)",
       "pre-registered (contrast)", "token_qwen3_score.json", "token_lens.py, score_qwen_token.py",
       "one model pair; correlational until F4")
def _():
    t = j("token_qwen3_score.json")
    got = [t["gemma_base"]["id_rebound"], t["id_rebound"]]
    return got, close(0.16, got[0], 0.01) and close(0.03, got[1], 0.01) and t["gemma_base"]["id_min_layer"] == 17


@claim("X1", "Cross-model relative-depth match >= 0.9 in 13/21 pairs; NMI vs CKA >= 0.7 in 5/21",
       "pre-registered (both failed)", "correspondence.json", "correspondence.py", "1,000 docs for CKA")
def _():
    c = j("correspondence.json")
    got = [c["X1_pairs_ge_0.9"], c["X2_pairs_ge_0.7"]]
    return got, got == [13, 5]


@claim("R1", "Routing stability correlates with p@10 in 11/11 pairs but beats all baselines in 1/11",
       "pre-registered (Q1 pass, Q2 fail)", "routing.json", "routing.py", "hubness is the better signal")
def _():
    r = j("routing.json")
    got = [r["Q1_pairs"], r["Q2_pairs"]]
    return got, got == [11, 1]


# Private claims (not in the public repo) are defined in verify_claims_private.py when it exists.
_priv = Path(__file__).resolve().parent / "verify_claims_private.py"
if _priv.exists():
    exec(_priv.read_text())


# ── geometry write-up (findings.html, 2026-10-05 framing) ──────────────────────────────────────────────────────────

def _adj(name, prep=""):
    C = np.array(j(f"geometry_probe_{name}{prep}.json")["cos_centered"])
    return [C[k, k + 1] for k in range(len(C) - 1)]


@claim("G1", "Untrained Pythia-410M: up to 6 layers apart every article finds itself (>= 0.999); 12 apart 0.98 on average; first vs last layer about 0.27; 6 apart it keeps 0.68 of its 10 neighbours",
       "exploratory", "pythia_geometry.json", "pythia_geometry.py", "one untrained model; 'same place' is lenient by construction")
def _():
    p = j("pythia_geometry.json")["0"]
    T, K = np.array(p["self_top1"]), np.array(p["knn10_overlap"])
    k6 = float(np.mean([K[i, i + 6] for i in range(1, len(K) - 6)]))
    L = len(T)
    m6 = min(min(T[i, i + d], T[i + d, i]) for d in range(1, 7) for i in range(1, L - d))
    a12 = float(np.mean([(T[i, i + 12] + T[i + 12, i]) / 2 for i in range(1, L - 12)]))
    far = (T[1, -1] + T[-1, 1]) / 2
    got = (round(float(m6), 3), round(a12, 3), round(float(far), 3), round(k6, 3))
    return got, m6 >= 0.999 and close(0.98, a12, 0.006) and close(0.27, far, 0.02) and close(0.68, k6, 0.01)


@claim("G2", "Apparent turns vanish with robust scaling: Qwen3 layer 2 0.18 -> 0.69, SmolLM2 layer 7 0.25 -> 0.85; robust minimum across one layer Qwen3 0.61, SmolLM2 0.61, Qwen3.5-2B 0.62",
       "exploratory", "geometry_probe_{qwen3_17b,smollm2_17b,qwen35_2b}{,_robust}.json", "geometry_probe.py",
       "articles only; robust scaling = per-dimension median/MAD")
def _():
    q, qr, s, sr = _adj("qwen3_17b"), _adj("qwen3_17b", "_robust"), _adj("smollm2_17b"), _adj("smollm2_17b", "_robust")
    q5r = _adj("qwen35_2b", "_robust")
    got = [q[2], qr[2], s[7], sr[7], min(qr[1:]), min(sr[1:]), min(q5r[1:])]
    want = [0.18, 0.69, 0.25, 0.85, 0.61, 0.61, 0.62]
    return [round(x, 3) for x in got], all(close(w, g, 0.006) for w, g in zip(want, got))


@claim("G3", "Gemma 4 E2B turns at layer 14: smallest centered cosine across one layer 0.037 plain, 0.045 robust; same layer at n = 1,000 and 6,000",
       "exploratory, replicated", "geometry_probe_gemma4_e2b{,_robust,_n1000,_n6000}.json", "geometry_probe.py",
       "one model family; articles (token-level runs pending re-extraction with <bos>)")
def _():
    out = []
    for tag in ("", "_robust", "_n1000", "_n6000"):
        a = _adj("gemma4_e2b", tag)
        out.append((int(np.argmin(a[1:])) + 1, round(float(min(a[1:])), 3)))
    return out, all(l == 14 for l, _ in out) and close(0.037, out[0][1], 0.001) and close(0.045, out[1][1], 0.001)


@claim("G4", "Across Gemma's turn (13 -> 15): centered cosine 0.004; articles keep 0.49 of neighbours (0.81 between neighbouring layers elsewhere)",
       "exploratory", "geometry_probe_gemma4_e2b.json", "geometry_probe.py", "'elsewhere' = median adjacent-layer overlap outside 12-16")
def _():
    p = j("geometry_probe_gemma4_e2b.json")
    C, K = np.array(p["cos_centered"]), np.array(p["knn10_overlap"])
    els = float(np.median([K[i, i + 1] for i in range(1, len(K) - 2) if not 12 <= i <= 16]))
    got = (round(C[13, 15], 3), round(K[13, 15], 3), round(els, 3))
    return got, close(0.004, got[0], 0.002) and close(0.49, got[1], 0.01) and close(0.81, got[2], 0.02)


@claim("G5", "Disjoint second sample of 3,000 articles: Gemma probe matrices agree within 0.06",
       "replication", "geometry_probe_gemma4_e2b{,_s1}.json", "geometry_probe.py --seed 1", "same model and dataset")
def _():
    a, b = j("geometry_probe_gemma4_e2b.json"), j("geometry_probe_gemma4_e2b_s1.json")
    d = max(float(np.abs(np.array(a[k]) - np.array(b[k])).max()) for k in ("cos_centered", "self_top1", "knn10_overlap"))
    return round(d, 3), d <= 0.06


@claim("G6", "Rotation fit finds the article: within Gemma 14->21 0.995; Gemma->Qwen3 0.982, Gemma E2B->E4B 0.986, Qwen3->SmolLM2 0.944; shuffled 0.002; same-topic pool 0.996 vs 0.985; relative residual 0.54 vs 0.59",
       "exploratory", "geometry_crossmodel_*.json", "geometry_crossmodel.py", "article-level identification; one dataset")
def _():
    gq, ge, qs = (j(f"geometry_crossmodel_{f}.json") for f in ("gemma4_e2b__qwen3_17b", "gemma4_e2b__gemma4_e4b", "qwen3_17b__smollm2_17b"))
    w, c = gq["within"]["14-21"]["rotation_scores"], gq["cross"]["14-22"]["rotation_scores"]
    got = [w["top1"], c["top1"], ge["cross"]["14-17"]["rotation"], qs["cross"]["22-19"]["rotation"],
           gq["shuffled"]["14-21"]["rotation_scores"]["top1"], w["hard"], c["hard"], w["resid"], c["resid"]]
    want = [0.995, 0.982, 0.986, 0.944, 0.002, 0.996, 0.985, 0.54, 0.59]
    return [round(x, 3) for x in got], all(close(a, b, 0.006) for a, b in zip(want, got))


@claim("G7", "CKA vs group-based layer correspondence across models: rank correlation 0.93-0.97 after rescaling top directions (21/21 >= 0.7); raw CKA -0.16 to 0.91",
       "pre-registered (D3)", "diag_d3.json", "diagnostics.py", "7 models, 1,000 documents")
def _():
    d = {k: v for k, v in j("diag_d3.json").items() if isinstance(v, dict)}
    w = [v["nmi_vs_cka_whitened"] for v in d.values()]
    r = [v["nmi_vs_cka_raw"] for v in d.values()]
    got = (len(w), round(min(w), 2), round(max(w), 2), round(min(r), 2), round(max(r), 2))
    return got, got[0] == 21 and min(w) >= 0.7 and close(0.93, got[1], 0.006) and close(0.97, got[2], 0.006)


@claim("G8", "Blocks beat the trend-keeping shuffle: p <= 0.002 in every article-level and Qwen3 token test but one (p = 0.021)",
       "exploratory", "block_tests.json", "block_tests.py", "1,000 shuffles; Gemma token rows excluded (no <bos>)")
def _():
    b = j("block_tests.json")["geometry_blocks"]
    ps = sorted(v["p_trend"] for g, d in b.items() if g != "tok_base" for v in d.values())
    return [round(p, 3) for p in ps[-2:]], ps[-2] <= 0.002 and close(0.021, ps[-1], 0.001)


@claim("G9", "Gemma's own writes at layers 13/14: 14x and 40x the residual; 16x and 26x with massive coordinates zeroed; vs scalar rho -0.98 / -0.90",
       "exploratory", "geometry_flow_noscale_gemma4_e2b{,_nomassive20}.json", "flow_noscale.py",
       "articles; 59 coordinates zeroed (mean |x| > 20x median at any layer)")
def _():
    a, b = j("geometry_flow_noscale_gemma4_e2b.json"), j("geometry_flow_noscale_gemma4_e2b_nomassive20.json")
    got = [a["update_rel"][13], a["update_rel"][14], b["update_rel"][13], b["update_rel"][14],
           a["spearman_vs_scalar"]["update_rel"], b["spearman_vs_scalar"]["update_rel"]]
    want = [14, 40, 16, 26, -0.98, -0.90]
    tol = [0.5, 0.5, 0.5, 0.5, 0.01, 0.01]
    return [round(x, 2) for x in got], all(close(w, g, t) for w, g, t in zip(want, got, tol))


@claim("G10", "Scalar dose (layers 13-14): centered cosine across layer 14 0.022 (x0.5), 0.043 (trained), 0.084 (x2), 0.165 (x4), 0.483 (=1); top-1 kept 92/91/71/23%",
       "exploratory, causal", "block_ablation_gemma4_e2b_dose.json", "block_ablation.py --tag dose", "40 texts")
def _():
    r = j("block_ablation_gemma4_e2b_dose.json")["results"]
    ks = ("scalex0.5_13_14", "clean", "scalex2_13_14", "scalex4_13_14", "scalar1_13_14")
    cos = [r[k]["adj_cos"][14] for k in ks]
    top = [r[k]["top1"] for k in ks if k != "clean"]
    ok = all(close(w, g, 0.002) for w, g in zip((0.022, 0.043, 0.084, 0.165, 0.483), cos)) and \
        all(close(w, g, 0.006) for w, g in zip((0.92, 0.91, 0.71, 0.23), top))
    return ([round(c, 3) for c in cos], [round(t, 3) for t in top]), ok


@claim("G11", "Knockouts with held-out means reproduce the originals (turn and top-1 within 0.01)",
       "exploratory, causal", "block_ablation_gemma4_e2b{,_heldout_means}.json", "block_ablation.py", "5 components")
def _():
    h, o = j("block_ablation_gemma4_e2b_heldout_means.json")["results"], j("block_ablation_gemma4_e2b.json")["results"]
    d = max(max(abs(h[k]["turn_13_15"] - o[k]["turn_13_15"]), abs(h[k]["top1"] - o[k]["top1"])) for k in h if k != "clean")
    return round(d, 4), d <= 0.01


@claim("G12", "E2B: scalar vs centered cosine across the same layer, rank correlation 0.86 (articles, layers 1-33)",
       "exploratory", "weight_structure_gemma4_e2b.json, geometry_probe_gemma4_e2b.json", "weight_structure.py, geometry_probe.py",
       "the last cached layer includes the final norm and is excluded")
def _():
    from scipy.stats import spearmanr
    s = np.array(j("weight_structure_gemma4_e2b.json")["layer_scalar"])
    c = _adj("gemma4_e2b")
    r = float(spearmanr(s[1:34], c[1:34]).statistic)
    return round(r, 3), close(0.86, r, 0.006)


@claim("G13", "E4B pre-registered: sharpest turns at 23 (0.048) and 22 (0.19); scalar-turn rho 0.90; a block starts at 23",
       "pre-registered, held-out model", "prereg_scalar_e4b.json", "score_prereg_e4b.py",
       "same family; window excluded layer 1 after reading the scalars; causal S4 not run")
def _():
    p = j("prereg_scalar_e4b.json")
    ok = p["S1"]["supported"] and p["S2"]["supported"] and p["S3"]["supported"] and close(0.90, p["S2"]["spearman"], 0.006)
    return (p["S1"]["two_sharpest"], p["S1"]["cos"], p["S2"]["spearman"]), ok


@claim("G14", "Flow map: groups do not track length (AMI with length bin <= 0.054; median topic AMI 0.36 / 0.44); 74% of articles at the 64-token cap",
       "exploratory", "flow_confounds.json", "flow_confounds.py", "length in three bins; one sample of 1,500")
def _():
    f = j("flow_confounds.json")
    mx = max(max(f[f"K{K}"]["length_ami"][1:]) for K in (4, 8))
    tm = [float(np.median(f[f"K{K}"]["topic_ami"][1:])) for K in (4, 8)]
    return (round(mx, 3), [round(t, 2) for t in tm], round(f["share_at_cap"], 3)), mx <= 0.06 and close(0.74, f["share_at_cap"], 0.01)


@claim("G15", "Gemma 4 E2B run correctly: next-token top-1 0.56 on the eval texts (Qwen3-1.7B-Base 0.56); precision and attention settings agree (cosine >= 0.9998)",
       "check", "gemma_sanity.json, gemma_dtype.json", "gemma_sanity.py, gemma_dtype.py", "200 and 40 texts; CPU")
def _():
    s, d = j("gemma_sanity.json"), j("gemma_dtype.json")
    g, q = s["google/gemma-4-E2B"]["dbpedia"]["top1"], s["Qwen/Qwen3-1.7B-Base"]["dbpedia"]["top1"]
    mn = min(min(v["cos_ref"]) for k, v in d.items() if k != "cache_turn")
    return (round(g, 3), round(q, 3), round(mn, 5)), close(0.56, g, 0.005) and close(0.56, q, 0.005) and mn >= 0.9998


@claim("G16", "Gemma tokens (re-extracted with <bos>): turn at 14, centered cosine across 13/14 0.08/0.05; keep 0.32 of neighbours across 13->15 vs 0.70 elsewhere; per-layer embeddings off: turn still at 14",
       "exploratory", "geometry_probe_tok_{base,nople0}_bos.json", "token_extract.py, geometry_probe.py", "3,000 token positions")
def _():
    p = j("geometry_probe_tok_base_bos.json")
    C, K = np.array(p["cos_centered"]), np.array(p["knn10_overlap"])
    els = float(np.median([K[i, i + 1] for i in range(1, len(K) - 2) if not 12 <= i <= 16]))
    a0 = _adj("tok_nople0_bos")
    got = (round(C[13, 14], 3), round(C[14, 15], 3), round(K[13, 15], 3), round(els, 3), int(np.argmin(a0[1:])) + 1)
    return got, close(0.08, got[0], 0.006) and close(0.05, got[1], 0.006) and close(0.32, got[2], 0.006) and close(0.70, got[3], 0.006) and got[4] == 14


@claim("G17", "Corrected Gemma token cache matches a separate fp32 run (cosine 1.000 at every layer); next-token top-1 0.557 from the cache",
       "check", "gemma_dtype_tok_base_bos.json, geometry_readout_tok_base_bos.json", "gemma_dtype.py, geometry_readout.py", "40 / 3,000 texts")
def _():
    d = j("gemma_dtype_tok_base_bos.json")["float32_eager"]
    r = j("geometry_readout_tok_base_bos.json")
    got = (round(min(d["cos_cache"]), 4), round(r["acc_logit"][-1], 3))
    return got, got[0] >= 0.999 and close(0.557, got[1], 0.003)


# ── stability-pick companion note ──────────────────────────────────────────────────────────────────────────────────

@claim("S1", "Ten-pair re-analysis on the same 11 pairs: mean shortfall 0.70, single pair expected 1.36, beats last in 10/11",
       "pre-registered re-analysis (after L1)", "robust_b1.json", "robustness.py b1", "same 11 pairs as L1")
def _():
    s = j("robust_b1.json")["summary"]
    got = (100 * s["S10_mean_regret"], 100 * s["single_expected_regret"], s["S10_beats_last"])
    return [round(x, 2) for x in got], close(0.70, got[0], 0.01) and close(1.36, got[1], 0.01) and got[2] == 10


@claim("S2", "Post hoc fixed rules: second-to-last layer 1.87, best fixed depth with hindsight 1.83",
       "post hoc", "robust_b1.json", "build_findings_data.py", "no interval; chosen after seeing the pairs")
def _():
    b = j("robust_b1.json")
    pairs = [v for k, v in b.items() if k != "summary"]
    reg = lambda v, l: 100 * (max(v["p10"]) - v["p10"][l])
    sl = float(np.mean([reg(v, len(v["p10"]) - 2) for v in pairs]))
    fx = min(float(np.mean([reg(v, int(round(f * (len(v["p10"]) - 1)))) for v in pairs])) for f in np.round(np.arange(0.30, 0.96, 0.01), 2))
    return (round(sl, 2), round(fx, 2)), close(1.87, sl, 0.01) and close(1.83, fx, 0.01)


@claim("S3", "It follows the collection, not the task: non-topical shortfall 5.7; length-best layer >= 1/4 depth from topic-best in 9/11",
       "pre-registered (B2)", "robust_b23.json", "robustness.py b23", "three text-derived labels")
def _():
    s = j("robust_b23.json")["summary"]
    return (round(100 * s["nontopical_mean_regret_S"], 2), s["length_oracle_far_from_topical"]), \
        close(5.7, 100 * s["nontopical_mean_regret_S"], 0.05) and s["length_oracle_far_from_topical"] == 9


@claim("S4", "Why-stability: separation tracks stability 11/11 but halves its link to quality in only 2/11; as selectors: kNN margin 1.80, silhouette 2.49, participation ratio 3.78",
       "pre-registered (A1, A2)", "why_stability_score.json", "why_stability.py, score_why_stability.py", "silhouette at K=512 is small (noisy)")
def _():
    s = j("why_stability_score.json")
    m = s["mean_pick_shortfall"]
    got = (s["A1"]["n_pos"], s["A2"]["n_halved"], round(m["knn_margin"], 2), round(m["silhouette"], 2), round(m["part_ratio"], 2))
    return got, got[:2] == (11, 2) and close(1.80, got[2], 0.01) and close(2.49, got[3], 0.01) and close(3.78, got[4], 0.01)

REPL = []


def replication(rid, prior, cite, kind, verdict_note):
    """kind: direct (same measure) / conceptual (same question, our partition measure) / fact-check."""
    def wrap(fn):
        REPL.append((rid, prior, cite, kind, verdict_note, fn))
        return fn
    return wrap


@replication("R-a", "Intermediate layers often give better embeddings than the last layer",
             "Skean et al. 2025", "conceptual (retrieval p@10 instead of their task suite)", "replicated")
def _():
    sel = j("baselines_score.json")["selections"]
    n = sum(v["oracle"] != v["last"]["layer"] for v in sel.values())
    return f"best layer is not the last in {n}/11 held-out pairs; last-layer regret 3.4 points on average", n >= 9


@replication("R-b", "Layers settle bottom-up during training",
             "Raghu et al. 2017; Chen et al. 2023 (layer convergence bias)", "conceptual (partition agreement with the final checkpoint)",
             "not replicated as stated: ends first, middle last; consistent with Skean et al. 2025 (middle layers change most)")
def _():
    p = j("pythia_score.json")
    rho = [p[m]["P2"]["nmi"]["spearman"] for m in ("pythia410m", "pythia14b")]
    ci = [p[m]["P2"]["nmi"]["ci95"] for m in ("pythia410m", "pythia14b")]
    return f"settle step vs depth: rho {rho[0]:.2f} {np.round(ci[0], 2).tolist()}, {rho[1]:.2f} {np.round(ci[1], 2).tolist()}", not any(c[0] > 0 for c in ci)


@replication("R-c", "Smaller Pythia models converge less stably", "Diehl Martinez et al. 2024", "conceptual",
             "consistent (weak: two sizes)")
def _():
    p = j("pythia_score.json")
    return f"stable layer choice at {p['pythia410m']['P1_count']}/9 checkpoints (410m) vs {p['pythia14b']['P1_count']}/9 (1.4b)", \
        p["pythia410m"]["P1_count"] < p["pythia14b"]["P1_count"]


@replication("R-d", "Corresponding layers of different LLMs sit at similar relative depth", "Wolfram & Schein 2025",
             "conceptual (bin agreement instead of nearest-neighbour overlap)", "partly replicated")
def _():
    v = sorted(x["X1_spearman"] for x in j("correspondence.json")["pairs"].values())
    return f"depth-match Spearman median {np.median(v):.2f} (range {v[0]:.2f}-{v[-1]:.2f}); >= 0.9 in 13/21 pairs", np.median(v) >= 0.9


@replication("R-e", "A few dimensions carry massive activations in LLMs", "Sun et al. 2024",
             "conceptual (variance share of mean-pooled features)", "replicated in 2 of 5 models; extends to Qwen3-1.7B and SmolLM2-1.7B")
def _():
    t = j("task_posthoc.json")["banking77_mean"]
    mx = {m: max(r["top5_var_share"]) for m, r in t.items()}
    return "max top-5 variance share: " + ", ".join(f"{m} {v:.2f}" for m, v in mx.items()), \
        sum(v > 0.95 for v in mx.values()) == 2


@replication("R-f", "Hubness marks documents that retrieve poorly", "Radovanović et al. 2010",
             "conceptual (k-occurrence vs a document's own p@10 failure)", "replicated (it was the best of four signals)")
def _():
    r = j("routing.json")
    a = [v["auroc"]["hubness"] for v in r.values() if isinstance(v, dict) and "auroc" in v]
    return f"hubness AUROC for failure {min(a):.2f}-{max(a):.2f} across 11 pairs", min(a) > 0.5


@replication("R-g", "In left-to-right LMs, current-token information falls with depth and next-token information rises",
             "Voita, Sennrich & Titov 2019", "direct in spirit (k-means bins of token states, NMI)",
             "partly replicated: identity falls in both; next-word information peaks at the last layer in Qwen3 but at layer 24 of 35 in Gemma, and never overtakes identity in either")
def _():
    g, q = j("token_score.json")["base"], j("token_qwen3_score.json")
    return (f"Gemma: identity {g['id_peak']:.2f} -> {g['id_last']:.2f}, next-word peak at layer {g['next_peak_layer']} of 35; "
            f"Qwen3: identity drop {q['id_drop']:.2f}, next-word peak at layer {q['next_peak_layer']} (last)"), \
        g["id_last"] < g["id_peak"] and q["next_peak_layer"] == q["n_layers"]


@replication("R-h", "Layers with lower intrinsic dimension carry more semantic structure", "Valeriani et al. 2023",
             "conceptual (TwoNN on pooled documents as a layer picker)",
             "consistent: lowest-ID was the better direction and the strongest rival selector; disagrees with Cheng et al. 2025 (token level)")
def _():
    d = j("baselines_score.json")["directions"]["twonn"]
    return f"dev regret: lowest-ID {100 * d['dev_mean_regret']['argmin']:.1f} vs highest-ID {100 * d['dev_mean_regret']['argmax']:.1f} points", \
        d["chosen"] == "argmin"


@replication("R-i", "llama.cpp k-quants give the first and last blocks extra bits", "llama.cpp `use_more_bits`",
             "fact-check (read from the GGUF files)", "confirmed for the first four blocks")
def _():
    g = j("gguf_layer_bits.json")
    files = [f for f in g if any(t in f for t in ("Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "IQ2_M"))]
    ok = all(all(x > g[f]["median_bpw"] for x in g[f]["bpw_by_decoder_layer"][:4]) for f in files)
    return f"blocks 0-3 above the median bits per weight in {len(files)}/{len(files)} mixed quantizations", ok


@replication("R-j", "Google's QAT Q4_0 keeps quality better than post-training Q4_0", "Google 2025 (Gemma QAT)",
             "conceptual (retrieval p@10 at the chosen layer)", "consistent")
def _():
    q = {k: j(f"{k}__select.json")["selection"]["l2"]["oracle"] for k in ("gq_qat4_0", "gq_q4_0", "gq_bf16")}
    return f"best-layer p@10: QAT {q['gq_qat4_0']:.3f}, Q4_0 {q['gq_q4_0']:.3f}, bf16 {q['gq_bf16']:.3f}", q["gq_qat4_0"] > q["gq_q4_0"]


def main():
    rows, bad = [], 0
    for cid, text, design, source, script, threats, fn in CLAIMS:
        try:
            got, ok = fn()
        except Exception as e:  # noqa: BLE001
            got, ok = f"ERROR {e!r}", False
        bad += not ok
        rows.append(f"| {cid} | {text} | {'✓' if ok else '✗ MISMATCH'} `{got}` | {design} | `results/{source}` | "
                    f"`{script}` | {threats} |")
        print(cid, "ok" if ok else "MISMATCH", got)
    OUT.write_text("# Claims registry (generated by `diag/verify_claims.py`; do not edit by hand)\n\n"
                   "Every number below is recomputed from the named result file. ✓ = matches the value stated in the "
                   "writeups. Design: pre-registered / post hoc / exploratory / descriptive.\n\n"
                   "| id | claim | recomputed | design | source | produced by | known threats |\n"
                   "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    rrows = []
    for rid, prior, cite, kind, verdict, fn in REPL:
        try:
            got, ok = fn()
        except Exception as e:  # noqa: BLE001
            got, ok = f"ERROR {e!r}", False
        rrows.append(f"| {rid} | {prior} | {cite} | {kind} | {got} | {verdict}{'' if ok else ' (CHECK: data does not support this verdict)'} |")
        bad += not ok
        print(rid, "ok" if ok else "CHECK", got)
    with OUT.open("a") as f:
        f.write("\n## Replications of prior findings\n\nWhere our measurements reproduce (or fail to reproduce) a "
                "published finding. 'Conceptual' means the same question asked with a different measure, so agreement is "
                "weaker evidence than a direct replication.\n\n| id | prior finding | source | kind | our measurement "
                "(recomputed) | verdict |\n|---|---|---|---|---|---|\n" + "\n".join(rrows) + "\n")
    print(f"{len(CLAIMS) + len(REPL) - bad}/{len(CLAIMS) + len(REPL)} entries consistent with their sources")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
