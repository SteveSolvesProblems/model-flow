"""research/PREREG_bins_w1.md: is the stability-quality link a ghost?

  W1-a  untrained null: rho(stability, quality) over layers 1..L, Pythia step 0 vs step 143000 (existing select JSON)
  W1-b  noise ladder: the best layer of each held-out pair with isotropic noise added, s in {0, .5, 1, 2, 4, 8}
  W1-c  bag-of-words reference: TF-IDF (first 64 words) -> SVD 512, stability and quality
Stability and quality exactly as select_layers.py. Writes results/bins_w1.json (saved after each part).

python bins_w1.py [a|b|c|all]
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score as nmi

from common import RESULTS, cache_meta, load_cache, set_threads
from lens import frame, layers_of, log
from partition import assign, kmeans
from robustness import HELD
from select_layers import K_RULE, p_at_k

PYTHIA = ["pythia410m", "pythia410m_seed1", "pythia410m_seed2", "pythia410m_seed3", "pythia410m_seed4",
          "pythia410m_dedup", "pythia14b", "pythia14b_dedup"]
LEVELS = [0, 0.5, 1, 2, 4, 8]
OUT = RESULTS / "bins_w1.json"


def lab(p):
    return "fine" if p.endswith("_20ng") else "l2"


def sel(name):
    return json.loads((RESULTS / f"{name}__select.json").read_text())


def curves(name, lv):
    rows = sel(name)["rows"]
    return np.array([r["stability_K512"] for r in rows]), np.array([r[f"p@10_{lv}"] for r in rows])


def stab_quality(Xtr, Xte, ytr, yte):
    Ftr, Fte = frame(Xtr, Xte, preproc="whiten")
    c = [assign(Fte, kmeans(Ftr, K_RULE, seed=s, nredo=1))[0] for s in (0, 1)]
    return float(nmi(c[0], c[1])), p_at_k(Xtr, ytr, Xte, yte)


def part_a():
    rows, fp = {}, {}
    for r in PYTHIA:
        # Amendment 1: these caches carry no weight fingerprint; check hub revisions and the states themselves
        r0, r1 = cache_meta(f"{r}_s0_L00").get("revision"), cache_meta(f"{r}_s143000_L00").get("revision")
        a0, a1 = (load_cache(f"{r}_{st}_L12", "test")[:500] for st in ("s0", "s143000"))
        rel = float(np.linalg.norm(a0 - a1) / np.linalg.norm(a1))
        fp[r] = {"s0": r0, "s143000": r1, "rel_diff_L12": rel, "differ": bool(r0 and r1 and r0 != r1 and rel > 0.5)}
        if not fp[r]["differ"]:
            log("dropped (fingerprint check)", r, fp[r])
            continue
        res = {}
        for st in ("s0", "s143000"):
            s, q = curves(f"{r}_{st}", "l2")
            res[st] = {"rho": float(spearmanr(s[1:], q[1:])[0]), "q_range": float(q[1:].max() - q[1:].min()),
                       "stab_range": float(s[1:].max() - s[1:].min())}
        rows[r] = res
        log("A", r, res)
    P = list(rows.values())
    unt = [p["s0"]["rho"] for p in P]
    tr = [p["s143000"]["rho"] for p in P]
    a1 = sum(t > u for t, u in zip(tr, unt))
    med_u = float(np.nanmedian(unt))
    tr_hi = sum(t >= 0.70 for t in tr)
    reading = ("ghost" if med_u >= 0.70 else "training-specific" if tr_hi >= 7 else "uninformative (Pythia lacks the link)")
    return {"fingerprints": fp, "pairs": rows, "n_pairs": len(P),
            "W1a1": {"count": a1, "of": len(P), "supported": a1 >= 7},
            "W1a2": {"median_untrained_rho": med_u, "ghost": med_u >= 0.70},
            "trained_rho_ge_0.70": tr_hi, "no_quality_variation": [r for r, p in rows.items() if p["s0"]["q_range"] < 0.01],
            "reading": reading}


def part_b():
    out = {}
    for p in HELD:
        lv = lab(p)
        s_real, q_real = curves(p, lv)
        best = int(q_real.argmax())
        nm = layers_of(p)[best]
        Xtr, ytr = load_cache(nm, "train", lv)
        Xte, yte = load_cache(nm, "test", lv)
        Xtr, Xte = Xtr.astype(np.float32), Xte.astype(np.float32)
        sigma = float(np.sqrt(Xtr.var(0).mean()))
        rng = np.random.default_rng(0)
        Ztr, Zte = rng.standard_normal(Xtr.shape, dtype=np.float32), rng.standard_normal(Xte.shape, dtype=np.float32)
        curve = []
        for s in LEVELS:
            st, q = stab_quality(Xtr + s * sigma * Ztr, Xte + s * sigma * Zte, {lv: ytr}, {lv: yte})
            curve.append((st, q[lv]))
            log("B", p, "layer", best, "s", s, round(st, 4), round(q[lv], 4))
        cs = np.array(curve)
        o = np.argsort(cs[:, 0])
        L = len(q_real) - 1
        lay = np.arange(1, L + 1)
        inr = (s_real[lay] >= cs[:, 0].min()) & (s_real[lay] <= cs[:, 0].max())
        qhat = np.interp(s_real[lay][inr], cs[o, 0], cs[o, 1])
        D = float(np.median(np.abs(q_real[lay][inr] - qhat))) * 100 if inr.any() else float("nan")
        out[p] = {"best_layer": best, "sigma": sigma, "curve": cs.tolist(),
                  "curve_monotone": bool(np.all(np.diff(cs[:, 0]) <= 1e-9) and np.all(np.diff(cs[:, 1]) <= 1e-9)),
                  "s0_diff_vs_select": [float(cs[0, 0] - s_real[best]), float(cs[0, 1] - q_real[best])],   # k-means run noise
                  "D_points": D, "coverage": float(inr.mean())}
        log("B", p, {k: v for k, v in out[p].items() if k != "curve"})
    Ds = [v["D_points"] for v in out.values()]
    cov = [v["coverage"] for v in out.values()]
    mD, mc = float(np.nanmedian(Ds)), float(np.median(cov))
    verdict = ("signal-to-noise explains the link" if mD <= 2 and mc >= 0.6 else
               "not signal-to-noise" if mD > 5 else "inconclusive")
    return {"pairs": out, "median_D_points": mD, "median_coverage": mc, "verdict": verdict}


def bow(data):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from extract_hf import dbpedia, newsgroups
    d, _ = newsgroups() if data == "20ng" else dbpedia(24000, 6000)
    first = lambda ts: [" ".join(t.split()[:64]) for t in ts]
    vec = TfidfVectorizer()
    Ttr = vec.fit_transform(first(d["train"][0]))
    Tte = vec.transform(first(d["test"][0]))
    svd = TruncatedSVD(512, random_state=0).fit(Ttr)
    return svd.transform(Ttr).astype(np.float32), svd.transform(Tte).astype(np.float32), d["train"][1], d["test"][1]


def part_c():
    out = {}
    for data, lv in (("dbpedia", "l2"), ("20ng", "fine")):
        Xtr, Xte, ytr, yte = bow(data)
        st, q = stab_quality(Xtr, Xte, {lv: ytr[lv]}, {lv: yte[lv]})
        out[data] = {"stability": st, "quality": q[lv], "pairs": {}}
        log("C", data, round(st, 4), round(q[lv], 4))
        for p in HELD:
            if (data == "20ng") != p.endswith("_20ng"):
                continue
            s, qq = curves(p, lv)
            s, qq = s[1:], qq[1:]
            out[data]["pairs"][p] = {"n_layers": len(s), "above_stab": int((s > st).sum()),
                                     "above_quality": int((qq > q[lv]).sum()), "above_both": int(((s > st) & (qq > q[lv])).sum())}
    return out


def main():
    set_threads()
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for k, f in (("a", part_a), ("c", part_c), ("b", part_b)):
        if what in (k, "all"):
            res[f"W1{k}"] = f()
            OUT.write_text(json.dumps(res, indent=1))
            log("saved", k)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("pairs", "fingerprints")} for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
