"""research/PREREG_diagnostics.md D1–D4: why did the overnight predictions fail?

python diagnostics.py d1|d1guard|d2|d3|d4   -> results/diag_<part>.json
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

from common import RESULTS, load_cache, set_threads
from lens import frame, layers_of, log
from partition import assign, kmeans
from token_lens import fast_nmi

HELD = ["qwen35_08b", "qwen35_08b_20ng", "qwen35_4b", "qwen35_4b_20ng", "gemma4_e4b", "gemma4_e4b_20ng", "smollm2_17b",
        "smollm2_17b_20ng", "qwen3_17b_20ng", "qwen35_2b_20ng", "gemma4_e2b_20ng"]
CORR_MODELS = ["qwen3_17b", "qwen35_08b", "qwen35_2b", "qwen35_4b", "gemma4_e2b", "gemma4_e4b", "smollm2_17b"]
VARIANTS = [(512, "W"), (64, "W"), (512, "D"), (64, "D")]


def deflated(Xtr, Xte, drop=5, r=64, seed=0):
    """Frame D: center, drop the top `drop` PCs, whiten the next r (PCs fit on <=12k train rows, as in lens.frame)."""
    mu = Xtr.mean(0)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), min(12000, len(Xtr)), replace=False)
    _, S, Vt = np.linalg.svd(Xtr[idx] - mu, full_matrices=False)
    W = Vt[drop:drop + r].T / (S[drop:drop + r] / np.sqrt(len(idx)) + 1e-6)
    return ((Xtr - mu) @ W).astype(np.float32), ((Xte - mu) @ W).astype(np.float32)


def top5_share(Xtr, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), min(12000, len(Xtr)), replace=False)
    S = np.linalg.svd(Xtr[idx] - Xtr.mean(0), compute_uv=False)
    v = S ** 2
    return float(v[:5].sum() / v.sum())


def stability(Ftr, Fte, K):
    c = [assign(Fte, kmeans(Ftr, K, seed=s, nredo=1))[0] for s in (0, 1)]
    return fast_nmi(c[0], c[1])


def layer_stats(name, variants=VARIANTS, share=True):
    Xtr, Xte = load_cache(name, "train"), load_cache(name, "test")
    frames = {"W": frame(Xtr, Xte), "D": deflated(Xtr, Xte)}
    r = {f"{K}{f}": stability(*frames[f], K) for K, f in variants}
    if share:
        r["top5"] = top5_share(Xtr)
    return r


def pick(stab, skip0=False):
    s = np.array(stab, float)
    if skip0:
        s[0] = -np.inf
    return int(np.flatnonzero(s == s.max()).max())


def d1():
    task = json.loads((RESULTS / "task_score.json").read_text())
    post = json.loads((RESULTS / "task_posthoc.json").read_text())
    out = {}
    for t in ("banking77_mean", "stsb_mean"):
        out[t] = {}
        for m, v in task[t].items():
            part = RESULTS / "diag" / "d1_parts" / f"{t}__{m}.json"     # per-model checkpoint (resume after a crash)
            if part.exists():
                rows = json.loads(part.read_text())
            else:
                rows = [layer_stats(n) for n in layers_of(t.replace("_mean", f"_{m}_mean"))]
                part.parent.mkdir(parents=True, exist_ok=True)
                part.write_text(json.dumps(rows))
            perf = {"primary": np.array(v["perf"]), "whiten64": np.array(post[t][m]["whiten64"]["perf"])}
            assert len(rows) == len(perf["primary"]) == len(perf["whiten64"]), (t, m, len(rows), len(perf["primary"]))
            res = {"rows": rows}
            for K, f in VARIANTS:
                st = [r[f"{K}{f}"] for r in rows]
                S = pick(st)
                res[f"{K}{f}"] = {"S": S, "range_ge1": float(np.ptp(st[1:])), "top5_at_S": rows[S]["top5"],
                                  **{f"regret_{o}": float(p.max() - p[S]) for o, p in perf.items()},
                                  **{f"beats_last_{o}": bool(p[S] > p[-1] + 1e-12) for o, p in perf.items()}}
            res["regret_last"] = {o: float(p.max() - p[-1]) for o, p in perf.items()}
            out[t][m] = res
            log("d1", t, m, {k: (res[k]["S"], round(res[k]["regret_primary"], 3)) for k in ("512W", "64W", "512D", "64D")})
    summ = {}
    for K, f in VARIANTS:
        key = f"{K}{f}"
        b = {t: sum(out[t][m][key]["beats_last_primary"] for m in out[t]) for t in out}
        b2 = {t: sum(out[t][m][key]["beats_last_whiten64"] for m in out[t]) for t in out}
        summ[key] = {"beats_last_primary": b, "beats_last_whiten64": b2,
                     "passes_primary": b["banking77_mean"] >= 4 and b["stsb_mean"] >= 3,
                     "mean_regret_primary": {t: float(np.mean([out[t][m][key]["regret_primary"] for m in out[t]])) for t in out}}
    out["summary"] = summ
    out["D1_H1"] = summ["64W"]["passes_primary"]
    out["D1_H2"] = summ["512D"]["passes_primary"]
    (RESULTS / "diag_d1.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summ, indent=1))


def d1guard():
    out, reg = {}, {f"{K}{f}": [] for K, f in VARIANTS}
    for p in HELD:
        lab = "fine" if p.endswith("_20ng") else "l2"
        p10 = np.array([r[f"p@10_{lab}"] for r in json.loads((RESULTS / f"{p}__select.json").read_text())["rows"]])
        rows = [layer_stats(n, share=False) for n in layers_of(p)]
        out[p] = {}
        for K, f in VARIANTS:
            S = pick([r[f"{K}{f}"] for r in rows])
            out[p][f"{K}{f}"] = {"S": S, "regret": float(p10.max() - p10[S])}
            reg[f"{K}{f}"].append(p10.max() - p10[S])
        log("guard", p, {k: (v["S"], round(v["regret"], 4)) for k, v in out[p].items()})
    out["mean_regret"] = {k: float(np.mean(v)) for k, v in reg.items()}
    (RESULTS / "diag_d1guard.json").write_text(json.dumps(out, indent=1))
    print(out["mean_regret"])


def d2():
    ps = json.loads((RESULTS / "pythia_score.json").read_text())
    out, cases = {}, []
    for m in ("pythia410m", "pythia14b"):
        out[m] = []
        for c in ps[m]["checkpoints"]:
            if c["step"] < 1000:
                continue
            rows = [layer_stats(n, variants=[(512, "D")]) for n in layers_of(f"{m}_s{c['step']}")]
            p10 = np.array(c["p10"])
            SD = pick([r["512D"] for r in rows])
            top5 = [r["top5"] for r in rows]
            rec = {"step": c["step"], "S": c["S"], "oracle": c["oracle"], "S_D": SD, "top5": top5,
                   "regret_S": c["regret_S"], "regret_S_D": float(p10.max() - p10[SD]), "stability_D": [r["512D"] for r in rows]}
            if c["step"] >= 4000 and c["S"] != c["oracle"]:
                cases.append(bool(top5[c["S"]] > top5[c["oracle"]]))
            out[m].append(rec)
            log("d2", m, c["step"], c["S"], c["oracle"], SD, round(rec["regret_S"], 3), round(rec["regret_S_D"], 3))
    out["H1_cases"] = cases
    out["D2_H1"] = bool(len(cases) > 0 and np.mean(cases) >= 0.8)
    out["D2_H2_by_model"] = {m: bool(np.mean([r["regret_S_D"] for r in out[m]]) < np.mean([r["regret_S"] for r in out[m]]))
                             for m in ("pythia410m", "pythia14b")}
    out["D2_H2"] = all(out["D2_H2_by_model"].values())
    (RESULTS / "diag_d2.json").write_text(json.dumps(out, indent=1))
    print(out["H1_cases"], out["D2_H1"], out["D2_H2_by_model"])


def gram_vec(X):
    X = X - X.mean(0)
    K = X @ X.T
    H = K - K.mean(0) - K.mean(1)[:, None] + K.mean()
    v = H.ravel()
    return (v / np.linalg.norm(v)).astype(np.float32)


def d3():
    import itertools
    corr = json.loads((RESULTS / "correspondence.json").read_text())
    mats = np.load(RESULTS / "correspondence.npz")
    codes0 = np.load(RESULTS / "codes" / f"{CORR_MODELS[0]}.npz")["c0"]
    sub = np.random.default_rng(0).choice(codes0.shape[1], 1000, replace=False)   # same draw as correspondence.py
    grams = {}
    for m in CORR_MODELS:
        g = []
        for nm in layers_of(m):
            _, Fte = frame(load_cache(nm, "train"), load_cache(nm, "test"))
            g.append(gram_vec(Fte[sub].astype(np.float64)))
        grams[m] = np.stack(g)
        log("d3 gram", m)
    out, hit, higher = {}, 0, 0
    for A, B in itertools.combinations(CORR_MODELS, 2):
        N = mats[f"{A}|{B}|nmi"]
        Cw = grams[A] @ grams[B].T
        r = spearmanr(N.ravel(), Cw.ravel()).statistic
        r0 = corr["pairs"][f"{A}|{B}"]["X2_nmi_vs_cka"]
        La, Lb = Cw.shape
        x1 = spearmanr(np.arange(La) / (La - 1), Cw.argmax(1) / (Lb - 1)).statistic
        out[f"{A}|{B}"] = {"nmi_vs_cka_whitened": float(r), "nmi_vs_cka_raw": r0, "X1_cka_whitened": float(x1)}
        hit += r >= 0.7
        higher += r > r0
        log("d3", A, B, round(float(r), 3), round(r0, 3))
    out["pairs_ge_0.7"], out["pairs_higher_than_raw"] = int(hit), int(higher)
    out["D3_H1"] = bool(hit >= 12 and higher >= 18)
    (RESULTS / "diag_d3.json").write_text(json.dumps(out, indent=1))
    print(hit, higher, out["D3_H1"])


def d4():
    poly = json.loads((RESULTS / "polysemy.json").read_text())
    tok = json.loads((RESULTS / "tok__token_lens.json").read_text())
    gid = [r["id_K512"] for r in sorted((r for r in tok["rows"] if r["model"] == "base"), key=lambda r: r["layer"])]
    qid = json.loads((RESULTS / "token_qwen3_score.json").read_text())["qwen_curves"]["id_K512"]
    out = {}
    for key, ident in (("gemma4_e2b_base", gid), ("qwen3_17b", qid)):
        pc = [r["partial_freq"] for r in poly[key]["layers"]]
        n = min(len(pc), len(ident))
        rho = spearmanr(ident[1:n], pc[1:n]).statistic
        out[key] = {"spearman_identity_vs_polysemy": float(rho), "n_layers": n - 1,
                    "identity": ident[:n], "polysemy_partial": pc[:n], "best_polysemy_layer": int(np.argmax(pc[1:n]) + 1),
                    "best_polysemy_value": float(max(pc[1:n]))}
        log("d4", key, round(float(rho), 3), out[key]["best_polysemy_layer"], round(out[key]["best_polysemy_value"], 3))
    out["D4_H1"] = bool(out["gemma4_e2b_base"]["spearman_identity_vs_polysemy"] <= -0.5)
    (RESULTS / "diag_d4.json").write_text(json.dumps(out, indent=1))
    print(out["D4_H1"])


if __name__ == "__main__":
    set_threads()
    {"d1": d1, "d1guard": d1guard, "d2": d2, "d3": d3, "d4": d4}[sys.argv[1]]()
