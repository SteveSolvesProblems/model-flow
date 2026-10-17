"""Data for the explainer's latent-geometry section (exploratory, not pre-registered).

For Gemma 4 E2B (base) and Qwen3-1.7B: ~300 tokens (repeated occurrences of a few common words plus a random fill)
and ~300 documents (stratified by DBpedia top-level class), with 2D positions at every layer in two views:
  shared   one 2D principal-component map fit on all layers at once (each layer centered, vectors unit length):
           the residual stream's common coordinates; an item that keeps its direction stays in place
  aligned  each layer's own 2D principal-component map, rotated/reflected to best match the previous layer
           (2D Procrustes): shows the arrangement layer by layer, with the coordinate change removed
Heatmaps come from geometry_probe.py (3,000-item samples). Writes results/geometry_viewer.json.

python geometry_viewer.py
"""
import json

import numpy as np

from common import CACHE_DIRS, RESULTS, cache_dir, set_threads
from lens import layers_of, log

WORDS = ["of", "the", "in", "is", "and", "born", "species", "album", "river", "school", "film", "american",
         "family", "village", "university", "football"]
PER_WORD, N_ITEMS = 15, 300
MODEL_NAMES = {"gemma": "Gemma 4 E2B", "qwen": "Qwen3-1.7B", "mbert": "ModernBERT-large", "laya": "Laya (fine-tuned ModernBERT)"}


def rows(name, split, idx):
    return np.load(cache_dir(name) / f"{split}_X.npy", mmap_mode="r")[np.sort(idx)][np.argsort(np.argsort(idx))].astype(np.float32)


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def views(states):
    """states: list over layers of (n, d) arrays for the same items."""
    cen = [unit(X - X.mean(0)) for X in states]
    allc = np.concatenate(cen)
    _, _, Vt = np.linalg.svd(allc - allc.mean(0), full_matrices=False)
    shared = [c @ Vt[:2].T for c in cen]
    aligned, prev = [], None
    for c in cen:
        _, _, V = np.linalg.svd(c - c.mean(0), full_matrices=False)
        P = (c - c.mean(0)) @ V[:2].T
        P /= np.sqrt((P ** 2).sum(1).mean()) + 1e-12
        if prev is not None:
            U, _, Wt = np.linalg.svd(P.T @ prev)                    # orthogonal R minimizing |P R - prev|
            P = P @ (U @ Wt)
        aligned.append(P)
        prev = P
    r = lambda A: np.round(np.stack(A), 3).tolist()
    return {"shared": r(shared), "aligned": r(aligned)}


def tokens(cache, meta_file):
    m = np.load(next(d / meta_file for d in CACHE_DIRS if (d / meta_file).exists()), allow_pickle=True)
    ev, vocab, text = m["eval"], m["vocab"], m["token_text"]
    rng = np.random.default_rng(0)
    pick, labels = [], []
    for w in WORDS:
        hits = np.flatnonzero(vocab[ev[:, 3]] == w)
        if len(hits) >= 10:
            sel = rng.choice(hits, min(PER_WORD, len(hits)), replace=False)
            pick += sel.tolist()
            labels += [w] * len(sel)
    rest = rng.choice(np.setdiff1d(np.arange(len(ev)), pick), N_ITEMS - len(pick), replace=False)
    pick += rest.tolist()
    labels += ["(other)"] * len(rest)
    pick = np.array(pick)
    states = [rows(nm, "eval", pick) for nm in layers_of(cache)]
    docs = m["eval_texts"]
    items = [{"label": labels[k], "tok": str(text[i]), "doc": str(docs[ev[i, 0]])[:110]} for k, i in enumerate(pick)]
    return items, states


def documents(name):
    from extract_hf import dbpedia
    names = layers_of(name)
    y = np.load(cache_dir(names[0]) / "test_y_l1.npy")
    data, maps = dbpedia(1, 6000)                                   # same test draw as extraction (seed 0)
    l1 = {v: k for k, v in maps["l1"].items()}
    rng = np.random.default_rng(0)
    per = N_ITEMS // len(np.unique(y))
    pick = np.concatenate([rng.choice(np.flatnonzero(y == c), min(per, int((y == c).sum())), replace=False) for c in np.unique(y)])
    states = [rows(nm, "test", pick) for nm in names]
    items = [{"label": l1[int(y[i])], "doc": data["test"][0][i][:110]} for i in pick]
    return items, states


def main():
    set_threads()
    out = {}
    for key, kind, src, extra, probe in (("gemma_tok", "token", "tok_base", "tok_meta.npz", "tok_base"),
                                          ("qwen_tok", "token", "tok_qwen3", "tok_meta_qwen3.npz", "tok_qwen3"),
                                          ("gemma_doc", "document", "gemma4_e2b", None, "gemma4_e2b"),
                                          ("qwen_doc", "document", "qwen3_17b", None, "qwen3_17b"),
                                          ("mbert_doc", "document", "mbert_large", None, "mbert_large"),
                                          ("laya_doc", "document", "laya", None, "laya")):
        items, states = tokens(src, extra) if kind == "token" else documents(src)
        pr = json.loads((RESULTS / f"geometry_probe_{probe}.json").read_text())
        out[key] = {"kind": kind, "model": MODEL_NAMES[key.split("_")[0]],
                    "items": items, **views(states),
                    "heat": {k: np.round(np.array(pr[k]), 3).tolist() for k in ("self_top1", "cos_centered", "knn10_overlap")},
                    "chance": {"self_top1": pr["chance_top1"], "knn10_overlap": pr["chance_knn10"]}}
        log(key, len(items), "items", len(states), "layers")
    (RESULTS / "geometry_viewer.json").write_text(json.dumps(out, separators=(",", ":")))
    log("saved", (RESULTS / "geometry_viewer.json").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
