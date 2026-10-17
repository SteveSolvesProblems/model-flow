"""research/PREREG_bins_w2.md W2-a: what the bins carry, layer by layer (articles).

Bins: K = 512 k-means (seeds 0 and 1) in the whiten-64 frame fit on 20,000 train documents (seed 0); 5,000 test
documents (seed 1), as why_stability.py part A. Attributes from the text the model saw (first 64 whitespace words):
topic levels, length (< 16, 16-63, >= 64 words of the full text), symbols (share of non-alphanumeric, non-space
characters; tertiles from train), lines (newlines per word; tertiles from train), format = length x symbols x lines.
Per layer: NMI(bins, attribute); merge / split rates between layers l and l+1, and the same between the seed-0 and
seed-1 partitions of layer l+1 (k-means noise). W2-a1: 20NG pairs, partial Spearman(W1-b deviation, NMI format |
stability). Reference: untrained Pythia-410M (step 0). Writes results/bins_w2a.json (after each model).
"""
import json
import re

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import normalized_mutual_info_score as nmi

from common import RESULTS, load_cache, set_threads
from lens import frame, layers_of, log
from partition import assign, kmeans
from robustness import HELD, SUB

N_TEST, K = 5000, 512
OUT = RESULTS / "bins_w2a.json"


def first64(t):
    m = list(re.finditer(r"\S+", t))
    return t[:m[63].end()] if len(m) > 64 else t


def text_attrs(texts, edges=None):
    f = [first64(t) for t in texts]
    nw = np.array([len(t.split()) for t in texts])
    sym = np.array([sum(not (c.isalnum() or c.isspace()) for c in t) / max(1, len(t)) for t in f])
    lines = np.array([t.count("\n") / max(1, len(t.split())) for t in f])
    if edges is None:
        edges = {"symbols": np.quantile(sym, [1 / 3, 2 / 3]).tolist(), "lines": np.quantile(lines, [1 / 3, 2 / 3]).tolist()}
    a = {"length": np.digitize(nw, [16, 64]), "symbols": np.digitize(sym, edges["symbols"]),
         "lines": np.digitize(lines, edges["lines"])}
    a["format"] = a["length"] * 9 + a["symbols"] * 3 + a["lines"]
    return a, edges


def attributes(pair):
    from extract_hf import dbpedia, newsgroups
    if pair.endswith("_20ng"):
        d, _ = newsgroups()
        levels = ("coarse", "fine")
    else:
        d, _ = dbpedia(24000, 6000)
        levels = ("l1", "l2")
    names = layers_of(pair)
    for lv in levels:                                          # cache order = loader order (checked on labels)
        assert np.array_equal(load_cache(names[0], "test", lv)[1], d["test"][1][lv]), f"label order {pair} {lv}"
    _, edges = text_attrs(d["train"][0])
    a, _ = text_attrs(d["test"][0], edges)
    for lv in levels:
        a[lv] = d["test"][1][lv]
    return a, edges


def pair_rates(a, b):
    """merge = P(same at b | different at a), split = P(different at b | same at a), over unordered item pairs."""
    c2 = lambda n: n * (n - 1) / 2
    n = len(a)
    same_a = c2(np.bincount(a)).sum()
    same_b = c2(np.bincount(b)).sum()
    _, cnt = np.unique(a.astype(np.int64) * 100000 + b, return_counts=True)
    both = c2(cnt).sum()
    tot = c2(n)
    return float((same_b - both) / (tot - same_a)), float((same_a - both) / same_a)


def run(pair):
    attrs, edges = attributes(pair)
    names = layers_of(pair)
    tidx = None
    codes = []
    rows = []
    for li, nm in enumerate(names):
        Xtr, Xte = load_cache(nm, "train"), load_cache(nm, "test")
        idx = np.random.default_rng(0).choice(len(Xtr), min(SUB, len(Xtr)), replace=False)
        if tidx is None:
            tidx = np.sort(np.random.default_rng(1).choice(len(Xte), min(N_TEST, len(Xte)), replace=False))
        Ftr, Fte = frame(Xtr[idx], Xte[tidx])
        c = [assign(Fte, kmeans(Ftr, K, seed=s, nredo=1))[0] for s in (0, 1)]
        codes.append(c)
        r = {f"nmi_{k}": float(nmi(c[0], v[tidx])) for k, v in attrs.items()}
        r["stability_here"] = float(nmi(c[0], c[1]))
        if li > 0:
            m, s = pair_rates(codes[li - 1][0], c[0])
            m0, s0 = pair_rates(c[1], c[0])
            r.update(merge=m, split=s, merge_noise=m0, split_noise=s0, merge_excess=m - m0, split_excess=s - s0)
        rows.append(r)
        log(pair, li, {k: round(v, 3) for k, v in r.items()})
    return {"edges": edges, "attr_counts": {k: np.bincount(v[tidx]).tolist() for k, v in attrs.items()},
            "rows": rows}


def partial_spearman(x, y, z):
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    ex, ey = res(rx), res(ry)
    return float((ex * ey).sum() / np.sqrt((ex ** 2).sum() * (ey ** 2).sum()))


def score(out):
    w1 = json.loads((RESULTS / "bins_w1.json").read_text())["W1b"]["pairs"]
    per = {}
    for p in [h for h in HELD if h.endswith("_20ng")]:
        sel = json.loads((RESULTS / f"{p}__select.json").read_text())["rows"]
        s = np.array([r["stability_K512"] for r in sel])
        q = np.array([r["p@10_fine"] for r in sel])
        cs = np.array(w1[p]["curve"])
        o = np.argsort(cs[:, 0])
        lay = np.arange(1, len(q))
        inr = (s[lay] >= cs[:, 0].min()) & (s[lay] <= cs[:, 0].max())
        lay = lay[inr]
        dev = q[lay] - np.interp(s[lay], cs[o, 0], cs[o, 1])
        fmt = np.array([out[p]["rows"][l]["nmi_format"] for l in lay])
        rho = partial_spearman(dev, fmt, s[lay])
        per[p] = {"n_layers": int(len(lay)), "partial_rho": rho}
    neg = sum(v["partial_rho"] < 0 for v in per.values())
    return {"W2a1": {"pairs": per, "negative": neg, "of": len(per), "supported": neg >= 5, "killed": neg <= 3}}


def main():
    set_threads()
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for p in [h for h in HELD if h.endswith("_20ng")] + [h for h in HELD if not h.endswith("_20ng")] + ["pythia410m_s0"]:
        if p in out:
            continue
        out[p] = run(p)
        OUT.write_text(json.dumps(out))
        if all(h in out for h in HELD if h.endswith("_20ng")) and "score" not in out:
            out["score"] = score(out)
            OUT.write_text(json.dumps(out))
            log("W2-a1", json.dumps(out["score"]))
    out["score"] = score(out)
    OUT.write_text(json.dumps(out))
    print(json.dumps(out["score"], indent=1))


if __name__ == "__main__":
    main()
