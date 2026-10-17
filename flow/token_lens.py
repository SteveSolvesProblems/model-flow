"""research/PREREG_token_lens.md: analysis of token-level partitions (pilot).

Per layer: whiten-64 frame and k-means (K in 32/512/2048, seeds 0/1) fit on each model's fit tokens; evaluation tokens
assigned. The 2-bit model is assigned to the tuned model's frame and bins.
Measures: stability, identity NMI, next-word NMI, context effect (topic NMI within word minus a permutation null),
2-bit flip rates by frequency band and token type. Also writes a compact file for the viewer.

python token_lens.py
"""
import json

import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi

from common import CACHE, RESULTS, load_cache, save_result, set_threads
from lens import layers_of, log
from partition import assign, kmeans
from quant_lens import fit_frame

MODELS = {"base": "tok_base", "it": "tok_it", "it2bit": "tok_it2bit"}
KS = [32, 512]
N_VIEW_DOCS = 150


def fast_nmi(a, b):
    """NMI with arithmetic-mean normalization (sklearn's default), for small integer label arrays."""
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    n = len(a)
    C = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(C, (a, b), 1)
    pa, pb, P = C.sum(1) / n, C.sum(0) / n, C / n
    nz = P > 0
    mi = float((P[nz] * np.log(P[nz] / np.outer(pa, pb)[nz])).sum())
    ha = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum())
    hb = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum())
    if ha == 0 and hb == 0:
        return 1.0
    return mi / ((ha + hb) / 2) if (ha + hb) > 0 else 0.0


def context_effect(cells, words, topic, rng, n_perm=5):
    """Occurrence-weighted mean over words of NMI(bin, topic) minus the within-word permutation null."""
    per = []
    for idx in words:
        c, t = cells[idx], topic[idx]
        obs = fast_nmi(t, c)
        null = np.mean([fast_nmi(rng.permutation(t), c) for _ in range(n_perm)])
        per.append((obs - null, len(idx)))
    d = np.array([p[0] for p in per]); w = np.array([p[1] for p in per], float)
    return float(np.average(d, weights=w)), d, w


def boot_ci(d, w, rng, n=500):
    k = len(d)
    s = [np.average(d[i], weights=w[i]) for i in (rng.integers(0, k, k) for _ in range(n))]
    return np.quantile(s, [0.025, 0.975]).tolist()


def describe(cells, K, words, nxt, vocab, freq_all, only=None):
    """Per bin: most over-represented words and next words (count >= 3), from fit tokens."""
    out = {}
    base_w = np.bincount(words, minlength=len(vocab)) / len(words)
    ok = nxt >= 0
    base_n = np.bincount(nxt[ok], minlength=len(vocab)) / ok.sum()
    for c in (range(K) if only is None else sorted(only)):
        m = cells == c
        n = int(m.sum())
        if n == 0:
            continue
        def top(ids, base, total):
            cnt = np.bincount(ids, minlength=len(vocab))
            cand = np.flatnonzero(cnt >= 3)
            lift = (cnt[cand] / total) / (base[cand] + 1e-12)
            o = cand[np.argsort(-lift)][:3]
            return [[str(vocab[i]), round(float(cnt[i] / total), 3), round(float((cnt[i] / total) / base[i]), 1)] for i in o]
        mn = m & ok
        out[c] = {"n": n, "w": top(words[m], base_w, n), "next": top(nxt[mn], base_n, max(1, int(mn.sum())))}
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="tok_meta.npz")
    ap.add_argument("--models", nargs="+", default=["base=tok_base", "it=tok_it", "it2bit=tok_it2bit"])
    ap.add_argument("--tag", default="tok")
    a = ap.parse_args()
    global MODELS
    MODELS = dict(m.split("=") for m in a.models)
    set_threads()
    meta = dict(np.load(CACHE / a.meta, allow_pickle=True))
    fit, ev = meta["fit"], meta["eval"]
    vocab, freq = meta["vocab"], meta["freq"]
    w_ev, nx_ev, l1 = ev[:, 3], ev[:, 4], meta["eval_l1"]
    rank = np.argsort(np.argsort(-freq))
    band = np.where(rank[w_ev] < 100, "top100", np.where(rank[w_ev] < 1000, "101-1000", "rarer"))
    ttype = meta["eval_type"]
    has_next = nx_ev >= 0
    # words for the context measure: >= 20 eval occurrences over >= 3 top-level topics
    ctx_words = []
    for c in np.unique(w_ev):
        idx = np.flatnonzero(w_ev == c)
        if len(idx) >= 20 and len(np.unique(l1[idx])) >= 3:
            ctx_words.append(idx)
    log(f"eval tokens {len(ev)}, fit tokens {len(fit)}, context words {len(ctx_words)}")
    view_rows = np.flatnonzero(ev[:, 0] < N_VIEW_DOCS)
    L = len(layers_of(next(iter(MODELS.values()))))
    rows, view = [], {m: {str(K): [] for K in (32, 512)} for m in MODELS}
    desc = {m: {str(K): [] for K in (32, 512)} for m in MODELS if m != "it2bit"}
    rng = np.random.default_rng(0)
    for li in range(L):
        fits = {}
        # it2bit reuses the it frame, so it must come after it; other names (e.g. ablations) run in the given order
        for key in [k for k in MODELS if k != "it2bit"] + [k for k in MODELS if k == "it2bit"]:
            name = layers_of(MODELS[key])[li]
            Xe = load_cache(name, "eval")
            if key == "it2bit":
                mu, W, C = fits["it"]
            else:
                Xf = load_cache(name, "fit")
                mu, W = fit_frame(Xf)
                Zf = (Xf - mu) @ W
                C = {(K, s): kmeans(Zf, K, seed=s, nredo=1) for K in KS for s in (0, 1)}
                fits[key] = (mu, W, C)
            Ze = (Xe - mu) @ W
            cells = {k: assign(Ze, c)[0] for k, c in C.items()}
            r = {"model": key, "layer": li}
            for K in KS:
                e0, e1 = cells[(K, 0)], cells[(K, 1)]
                r[f"stab_K{K}"] = float(nmi(e0, e1))
                r[f"id_K{K}"] = float(nmi(w_ev, e0))
                r[f"next_K{K}"] = float(nmi(nx_ev[has_next], e0[has_next]))
                eff, d, w = context_effect(e0, ctx_words, l1, rng)
                r[f"ctx_K{K}"] = eff
                if K == 512:
                    r["ctx_K512_ci"] = boot_ci(d, w, rng)
            if key == "it2bit":
                for K in (512,):
                    it_cells = fits["it_cells"]
                    flip = np.mean([it_cells[(K, s)] != cells[(K, s)] for s in (0, 1)], axis=0)
                    ceil = nmi(it_cells[(K, 0)], it_cells[(K, 1)])
                    r["flip_norm_K512"] = float(1 - np.mean([nmi(it_cells[(K, s)], cells[(K, s)]) for s in (0, 1)]) / ceil)
                    r["flip_K512"] = float(flip.mean())
                    r["flip_by_band"] = {b: float(flip[band == b].mean()) for b in ("top100", "101-1000", "rarer")}
                    r["flip_by_type"] = {t: float(flip[ttype == t].mean()) for t in np.unique(ttype)}
            if key == "it":
                fits["it_cells"] = cells
            for K in (32, 512):
                view[key][str(K)].append(cells[(K, 0)][view_rows].tolist())
            if key != "it2bit":
                fc = {K: assign(Zf, C[(K, 0)])[0] for K in (32, 512)}
                used512 = set(cells[(512, 0)][view_rows].tolist())
                d32 = describe(fc[32], 32, fit[:, 3], fit[:, 4], vocab, freq)
                d512 = describe(fc[512], 512, fit[:, 3], fit[:, 4], vocab, freq, only=used512)
                desc[key]["32"].append(d32)
                desc[key]["512"].append({c: v for c, v in d512.items() if c in used512})
            rows.append(r)
            log(f"L{li:02d} {key:6s} id={r['id_K512']:.3f} next={r['next_K512']:.3f} ctx={r['ctx_K512']:.4f} "
                f"stab={r['stab_K512']:.3f}" + (f" flip={r['flip_K512']:.3f}" if "flip_K512" in r else ""))
    save_result(a.tag, "token_lens", {"rows": rows, "n_eval": int(len(ev)), "n_fit": int(len(fit)),
                                      "n_ctx_words": len(ctx_words)})
    vr = ev[view_rows]
    out = {"n_layers": L, "docs": [str(t) for t in meta["eval_texts"][:N_VIEW_DOCS]],
           "tokens": {"doc": vr[:, 0].tolist(), "pos": vr[:, 1].tolist(), "text": [str(t) for t in meta["token_text"][view_rows]],
                      "l2": [str(meta["l2_names"][i]) for i in meta["eval_l2"][view_rows]]},
           "codes": view, "desc": desc, "rows": rows}
    (RESULTS / f"{a.tag}_viewer.json").write_text(json.dumps(out, separators=(",", ":")))
    log("saved")


if __name__ == "__main__":
    main()
