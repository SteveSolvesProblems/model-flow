"""research/PREREG_robustness.md B1-B4a.

python robustness.py b1      -> results/robust_b1.json   (selection noise: 10 seed pairs x train subsamples)
python robustness.py b23     -> results/robust_b23.json  (non-topical labels; outcome metric spaces)
python robustness.py b4a     -> results/robust_b4a.json  (bin flips vs matched noise)
"""
import json
import sys

import faiss
import numpy as np

from common import RESULTS, load_cache, set_threads
from lens import frame, layers_of, log
from partition import assign, kmeans
from quant_lens import fit_frame
from token_lens import fast_nmi

HELD = ["qwen35_08b", "qwen35_08b_20ng", "qwen35_4b", "qwen35_4b_20ng", "gemma4_e4b", "gemma4_e4b_20ng", "smollm2_17b",
        "smollm2_17b_20ng", "qwen3_17b_20ng", "qwen35_2b_20ng", "gemma4_e2b_20ng"]
NPAIRS, SUB = 10, 20000


def topical(p):
    lab = "fine" if p.endswith("_20ng") else "l2"
    rows = json.loads((RESULTS / f"{p}__select.json").read_text())["rows"]
    return lab, np.array([r[f"p@10_{lab}"] for r in rows])


def pick(v):
    v = np.asarray(v)
    return int(np.flatnonzero(v == v.max()).max())


def b1():
    out = {}
    for p in HELD:
        _, p10 = topical(p)
        names = layers_of(p)
        S = np.zeros((len(names), NPAIRS))
        for li, nm in enumerate(names):
            Xtr, Xte = load_cache(nm, "train"), load_cache(nm, "test")
            for j in range(NPAIRS):
                idx = np.random.default_rng(j).choice(len(Xtr), min(SUB, len(Xtr)), replace=False)
                Ftr, Fte = frame(Xtr[idx], Xte)
                c = [assign(Fte, kmeans(Ftr, 512, seed=s, nredo=1))[0] for s in (2 * j, 2 * j + 1)]
                S[li, j] = fast_nmi(c[0], c[1])
            log("b1", p, li, np.round(S[li].mean(), 4), np.round(S[li].std(), 4))
        picks = [pick(S[:, j]) for j in range(NPAIRS)]
        s10 = pick(S.mean(1))
        out[p] = {"stability": S.tolist(), "p10": p10.tolist(), "S10": s10, "regret_S10": float(p10.max() - p10[s10]),
                  "single_picks": picks, "single_regrets": [float(p10.max() - p10[k]) for k in picks],
                  "regret_last": float(p10.max() - p10[-1])}
        log("b1", p, "S10", s10, "picks", picks)
    reg10 = [out[p]["regret_S10"] for p in HELD]
    regs = [np.mean(out[p]["single_regrets"]) for p in HELD]
    beats = sum(out[p]["regret_S10"] < out[p]["regret_last"] - 1e-12 for p in HELD)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(10000):
        ps = rng.integers(0, len(HELD), len(HELD))
        boots.append(np.mean([out[HELD[i]]["regret_last"] - out[HELD[i]]["single_regrets"][rng.integers(NPAIRS)]
                              for i in ps]))
    out["summary"] = {"S10_mean_regret": float(np.mean(reg10)), "single_expected_regret": float(np.mean(regs)),
                      "S10_beats_last": int(beats), "last_minus_single_ci95": np.percentile(boots, [2.5, 97.5]).tolist(),
                      "B1a": bool(np.mean(reg10) <= 0.010), "B1b": bool(np.mean(regs) <= 0.015), "B1c": bool(beats >= 9)}
    (RESULTS / "robust_b1.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))


def texts_for(p):
    from extract_hf import dbpedia, newsgroups
    if p.endswith("_20ng"):
        d, _ = newsgroups()
        return [[t if t.strip() else "(empty)" for t in d[s][0]] for s in ("train", "test")]
    d, _ = dbpedia(24000, 6000)
    return [d["train"][0], d["test"][0]]


def nontopical(tr, te):
    def feats(ts):
        L = np.array([len(t) for t in ts], float)
        head = [t[:400] for t in ts]
        dig = np.array([sum(c.isdigit() for c in h) / max(len(h), 1) for h in head])
        cap = np.array([sum(c.isupper() for c in h) / max(len(h), 1) for h in head])
        return {"length": L, "digits": dig, "capitals": cap}
    ftr, fte = feats(tr), feats(te)
    out = {}
    for k in ftr:
        q = np.quantile(ftr[k], [1 / 3, 2 / 3])
        out[k] = (np.digitize(ftr[k], q), np.digitize(fte[k], q))
    return out


def p_at_10(A, B, ytr, yte):
    def unit(X):
        return np.ascontiguousarray(X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12), dtype=np.float32)
    idx = faiss.IndexFlatIP(A.shape[1])
    idx.add(unit(A))
    _, I = idx.search(unit(B), 10)
    return float((ytr[I] == yte[:, None]).mean())


def b23():
    out = {}
    cache_txt = {}
    for p in HELD:
        key = "20ng" if p.endswith("_20ng") else "dbpedia"
        if key not in cache_txt:
            cache_txt[key] = nontopical(*texts_for(p))
        labs = cache_txt[key]
        lab, p10 = topical(p)
        names = layers_of(p)
        ytr_t, yte_t = load_cache(names[0], "train", lab)[1], load_cache(names[0], "test", lab)[1]
        rows = []
        for nm in names:
            Xtr, Xte = load_cache(nm, "train"), load_cache(nm, "test")
            assert len(Xtr) == len(labs["length"][0]) and len(Xte) == len(labs["length"][1])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            Ftr, Fte = frame(Xtr, Xte)
            r = {"centered": p_at_10(Xtr - mu, Xte - mu, ytr_t, yte_t),
                 "zscored": p_at_10((Xtr - mu) / sd, (Xte - mu) / sd, ytr_t, yte_t),
                 "whiten64": p_at_10(Ftr, Fte, ytr_t, yte_t)}
            for k, (a, b) in labs.items():
                r[f"nt_{k}"] = p_at_10(Xtr - mu, Xte - mu, a, b)
            rows.append(r)
            log("b23", p, nm, {k: round(v, 3) for k, v in r.items()})
        sel = json.loads((RESULTS / f"{p}__select.json").read_text())["rows"]
        S = pick([r["stability_K512"] for r in sel])
        res = {"S": S, "rows": rows}
        for k in rows[0]:
            v = np.array([r[k] for r in rows])
            res[k] = {"oracle": int(v.argmax()), "regret_S": float(v.max() - v[S]), "regret_last": float(v.max() - v[-1])}
        out[p] = res
    nt = ["nt_length", "nt_digits", "nt_capitals"]
    depth = {p: len(out[p]["rows"]) - 1 for p in HELD}
    summ = {"nontopical_mean_regret_S": float(np.mean([out[p][k]["regret_S"] for p in HELD for k in nt])),
            "length_oracle_far_from_topical": int(sum(abs(out[p]["nt_length"]["oracle"] - out[p]["centered"]["oracle"])
                                                      >= depth[p] / 4 for p in HELD))}
    summ["B2a"] = bool(summ["nontopical_mean_regret_S"] >= 0.015)
    summ["B2b"] = bool(summ["length_oracle_far_from_topical"] >= 6)
    for sp in ("zscored", "whiten64"):
        m = float(np.mean([out[p][sp]["regret_S"] for p in HELD]))
        b = int(sum(out[p][sp]["regret_S"] < out[p][sp]["regret_last"] - 1e-12 for p in HELD))
        summ[f"{sp}_mean_regret_S"], summ[f"{sp}_S_beats_last"] = m, b
    summ["B3a"] = all(summ[f"{sp}_mean_regret_S"] <= 0.010 and summ[f"{sp}_S_beats_last"] >= 9 for sp in ("zscored", "whiten64"))
    out["summary"] = summ
    (RESULTS / "robust_b23.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summ, indent=1))


def b4a():
    out = {}
    refs = layers_of("gq_bf16")
    for v in ("gq_q4km", "gq_iq2m"):
        rows = []
        for li, ref in enumerate(refs):
            Rtr, Rte = load_cache(ref, "train"), load_cache(ref, "test")
            Q = load_cache(ref.replace("gq_bf16", v), "test")
            mu, W = fit_frame(Rtr)
            C = kmeans((Rtr - mu) @ W, 512, seed=0, nredo=1)
            base = assign((Rte - mu) @ W, C)[0]
            E = Q - Rte
            en = np.linalg.norm(E, axis=1, keepdims=True)
            g = np.random.default_rng(li).standard_normal(E.shape).astype(np.float32)
            iso = Rte + g / np.linalg.norm(g, axis=1, keepdims=True) * en
            shuf = Rte + E[np.random.default_rng(1000 + li).permutation(len(E))]
            f = lambda X: float((assign((X - mu) @ W, C)[0] != base).mean())
            rows.append({"layer": li, "real": f(Q), "isotropic": f(iso), "shuffled": f(shuf),
                         "rel_err": float(np.mean(en[:, 0] / (np.linalg.norm(Rte, axis=1) + 1e-12)))})
            log("b4a", v, rows[-1])
        ratio = [r["real"] / max(r["isotropic"], 1e-9) for r in rows[1:]]
        out[v] = {"rows": rows, "share_layers_real_ge_1.25x_iso": float(np.mean([x >= 1.25 for x in ratio]))}
    out["B4a"] = all(out[v]["share_layers_real_ge_1.25x_iso"] >= 0.5 for v in ("gq_q4km", "gq_iq2m"))
    (RESULTS / "robust_b4a.json").write_text(json.dumps(out, indent=1))
    print({v: out[v]["share_layers_real_ge_1.25x_iso"] for v in ("gq_q4km", "gq_iq2m")}, out["B4a"])


if __name__ == "__main__":
    set_threads()
    {"b1": b1, "b23": b23, "b4a": b4a}[sys.argv[1]]()
