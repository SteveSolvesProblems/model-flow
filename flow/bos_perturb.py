"""research/PREREG_null_followups.md P1: <bos> on vs off as a perturbation of Gemma 4 E2B's coordinate system.

Tokens: tok_base (no <bos>, tok_meta.npz) vs tok_base_bos (tok_meta_bos.npz), eval rows aligned by document and
text token (no-<bos> position p = <bos> position p + 1, token ids must agree). Articles: gemma4_e2b_it test states
without <bos> (cache_nobos/) vs with, 5,000 test documents (seed 1, as why_stability.py).
Per layer: cos (raw), ccos (each condition centered), knn (kNN10 overlap across conditions, centered cosine),
tokens also ccos by position band; huge coordinates (mean |x| > 20x median) per condition (exploratory).
Writes results/bos_perturb.json.
"""
import json
import os
from pathlib import Path

import numpy as np

from common import CACHE, CACHE_DIRS, RESULTS, load_cache, set_threads
from lens import layers_of, log

NOBOS = Path(os.environ.get("DIAG_CACHE_NOBOS", "data/cache_nobos"))
N_TEST, K, BIG = 5000, 10, 20.0


def meta(f):
    return np.load(next(c / f for c in CACHE_DIRS if (c / f).exists()), allow_pickle=True)


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def knn(X, k=K, chunk=2048):
    U = unit(X - X.mean(0)).astype(np.float32)
    out = np.empty((len(U), k), np.int64)
    for s in range(0, len(U), chunk):
        S = U[s:s + chunk] @ U.T
        np.fill_diagonal(S[:, s:s + chunk], -np.inf)
        out[s:s + chunk] = np.argpartition(-S, k, axis=1)[:, :k]
    return out


def overlap(A, B):
    return float(np.mean([len(set(a) & set(b)) / len(a) for a, b in zip(A, B)]))


def huge(X):
    m = np.abs(X).mean(0)
    return set(np.flatnonzero(m > BIG * np.median(m)).tolist())


def compare(Xa, Xb, knn_rows=None):
    Xa, Xb = Xa.astype(np.float32), Xb.astype(np.float32)
    cos = float((unit(Xa) * unit(Xb)).sum(1).mean())
    ca, cb = unit(Xa - Xa.mean(0)), unit(Xb - Xb.mean(0))
    cc = (ca * cb).sum(1)
    r = knn_rows if knn_rows is not None else np.arange(len(Xa))
    return {"cos": cos, "ccos": float(cc.mean()), "knn": overlap(knn(Xa[r]), knn(Xb[r])),
            "huge_a": sorted(huge(Xa)), "huge_b": sorted(huge(Xb))}, cc


def tokens():
    a, b = meta("tok_meta.npz")["eval"], meta("tok_meta_bos.npz")["eval"]
    ib = {(int(d), int(p)): i for i, (d, p) in enumerate(b[:, :2])}
    pairs = [(i, ib[(int(d), int(p) + 1)]) for i, (d, p) in enumerate(a[:, :2]) if (int(d), int(p) + 1) in ib]
    ia, jb = np.array(pairs).T
    same = float(np.mean(a[ia, 2] == b[jb, 2]))
    log("aligned", len(ia), "of", min(len(a), len(b)), "token ids agree", same)
    assert len(ia) >= 0.99 * min(len(a), len(b)) and same == 1.0, "alignment failed"
    pos = a[ia, 1]
    early, late = (pos >= 1) & (pos <= 4), pos >= 16
    sub = np.sort(np.random.default_rng(0).choice(len(ia), min(20000, len(ia)), replace=False))   # kNN subset
    rows = []
    for na, nb in zip(layers_of("tok_base"), layers_of("tok_base_bos")):
        r, cc = compare(load_cache(na, "eval")[ia], load_cache(nb, "eval")[jb], sub)
        r.update(ccos_pos1_4=float(cc[early].mean()), ccos_pos16=float(cc[late].mean()))
        rows.append(r)
        log("tok", na, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)})
    return {"n_aligned": int(len(ia)), "ids_agree": same, "n_early": int(early.sum()), "n_late": int(late.sum()),
            "n_knn": int(len(sub)), "rows": rows}


def articles():
    names = layers_of("gemma4_e2b_it")
    n = np.load(CACHE / names[0] / "test_X.npy", mmap_mode="r").shape[0] if (CACHE / names[0]).exists() else None
    rows = []
    for nm in names:
        Xb = load_cache(nm, "test")
        Xa = np.load(NOBOS / nm / "test_X.npy")
        assert Xa.shape == Xb.shape
        t = np.sort(np.random.default_rng(1).choice(len(Xb), min(N_TEST, len(Xb)), replace=False))
        r, _ = compare(Xa[t], Xb[t])
        rows.append(r)
        log("doc", nm, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)})
    return {"n_test": N_TEST, "n_docs": n, "rows": rows}


def score(out):
    s = {}
    for kind in ("tokens", "articles"):
        k = np.array([r["knn"] for r in out[kind]["rows"]])
        s[f"P1a_{kind}"] = {"early_1_13": float(k[1:14].mean()), "late_15_35": float(k[15:36].mean()),
                            "supported": bool(k[1:14].mean() > k[15:36].mean())}
    t = out["tokens"]["rows"]
    wins = sum(t[l]["ccos_pos16"] > t[l]["ccos_pos1_4"] for l in range(2, 36))
    s["P1b"] = {"layers_late_higher": int(wins), "of": 34, "supported": bool(wins >= 30)}
    s["P1a_killed"] = not (s["P1a_tokens"]["supported"] or s["P1a_articles"]["supported"])
    for kind in ("tokens", "articles"):
        ha = set().union(*[set(r["huge_a"]) for r in out[kind]["rows"]])
        hb = set().union(*[set(r["huge_b"]) for r in out[kind]["rows"]])
        s[f"P1c_{kind}"] = {"n_nobos": len(ha), "n_bos": len(hb), "jaccard": len(ha & hb) / max(1, len(ha | hb))}
    return s


def main():
    set_threads()
    out = {"tokens": tokens(), "articles": articles()}
    out["score"] = score(out)
    (RESULTS / "bos_perturb.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["score"], indent=1))


if __name__ == "__main__":
    main()
