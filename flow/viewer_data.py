"""Data for the attractor-lens viewer (research page, private).

Three variants of Gemma 4 E2B on the same DBpedia test documents:
  base   = gemma4_e2b (official base model, fp16)
  it     = gq_bf16    (instruction-tuned, bf16 GGUF reference)
  it2bit = gq_iq2m    (instruction-tuned, IQ2_M), assigned to the *it* frame and attractors so cells correspond
Per model, layer and K in {8, 32}: cell of each sampled test document, plus a description of every cell
(size, top l1/l2 classes and their shares, from the train split). Also per-layer stability/retrieval
(select JSONs) and the quantization flip ladder (quantlens_all).

python viewer_data.py --out results/viewer_data.json
"""
import argparse
import json

import numpy as np

from common import RESULTS, load_cache, set_threads
from extract_hf import dbpedia
from lens import layers_of, log
from partition import assign, kmeans
from quant_lens import fit_frame

MODELS = {"base": "gemma4_e2b", "it": "gq_bf16", "it2bit": "gq_iq2m"}
KS = [8, 32]


def describe(tr_cells, K, y1, y2, names1, names2):
    out = []
    for c in range(K):
        m = tr_cells == c
        n = int(m.sum())
        if n == 0:
            out.append({"n": 0})
            continue
        b1, b2 = np.bincount(y1[m], minlength=len(names1)), np.bincount(y2[m], minlength=len(names2))
        t2 = np.argsort(-b2)[:3]
        out.append({"n": n, "l1": names1[int(b1.argmax())], "l1p": round(float(b1.max() / n), 3),
                    "l2": [[names2[int(i)], round(float(b2[i] / n), 3)] for i in t2 if b2[i] > 0],
                    # full counts (sparse) so the page can rank classes by how unusual they are in this cell
                    "c1": {int(i): int(b1[i]) for i in np.flatnonzero(b1)},
                    "c2": {int(i): int(b2[i]) for i in np.flatnonzero(b2)}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=1500)
    ap.add_argument("--out", default=str(RESULTS / "viewer_data.json"))
    a = ap.parse_args()
    set_threads()
    data, maps = dbpedia(24000, 6000)
    names = {lv: [k for k, _ in sorted(maps[lv].items(), key=lambda kv: kv[1])] for lv in ("l1", "l2")}
    texts, lab = data["test"]
    first = layers_of(MODELS["base"])[0]
    ytr1, ytr2 = load_cache(first, "train", "l1")[1], load_cache(first, "train", "l2")[1]
    yte1, yte2 = load_cache(first, "test", "l1")[1], load_cache(first, "test", "l2")[1]
    assert (yte2 == lab["l2"]).all() and (yte1 == lab["l1"]).all(), "cache labels don't match the dataset order"
    for m in MODELS.values():
        f0 = layers_of(m)[0]
        assert (load_cache(f0, "test", "l2")[1] == yte2).all(), f"{m}: different test order"
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(yte2), a.n_docs, replace=False))
    L = len(layers_of(MODELS["base"]))
    codes = {m: {K: [] for K in KS} for m in MODELS}
    cells = {m: {K: [] for K in KS} for m in MODELS}
    for li in range(L):
        fits = {}
        for key in ("base", "it"):
            n = layers_of(MODELS[key])[li]
            Xtr, Xte = load_cache(n, "train"), load_cache(n, "test")
            mu, W = fit_frame(Xtr)
            Ztr, Zte = (Xtr - mu) @ W, (Xte[idx] - mu) @ W
            for K in KS:
                C = kmeans(Ztr, K, seed=0, nredo=3)
                tr = assign(Ztr, C)[0]
                codes[key][K].append(assign(Zte, C)[0].tolist())
                cells[key][K].append(describe(tr, K, ytr1, ytr2, names["l1"], names["l2"]))
                fits[(key, K)] = (mu, W, C, cells[key][K][-1])
        n = layers_of(MODELS["it2bit"])[li]
        Xte = load_cache(n, "test")[idx]
        for K in KS:
            mu, W, C, desc = fits[("it", K)]
            codes["it2bit"][K].append(assign((Xte - mu) @ W, C)[0].tolist())
            cells["it2bit"][K].append(desc)   # same attractors as it
        log(f"layer {li} done")
    sel = {}
    for key, m in MODELS.items():
        d = json.loads((RESULTS / f"{m}__select.json").read_text())
        sel[key] = {"stability": [r["stability_K512"] for r in d["rows"]], "p10": [r["p@10_l2"] for r in d["rows"]],
                    "S": d["selection"]["l2"]["S"]["layer"], "oracle": d["selection"]["l2"]["oracle_layer"]}
    q = json.loads((RESULTS / "gq_bf16__quantlens_all.json").read_text())
    ladder = {}
    for r in q["rows"]:
        ladder.setdefault(r["variant"].replace("gq_", ""), []).append(
            {"layer": r["layer"], "changed": round(r["changed_K512"], 4), "flip": round(r["flip_norm_K512"], 4),
             "rel_l2": round(r["rel_l2"], 4)})
    out = {"n_layers": L, "ks": KS, "docs": [{"t": texts[i][:220], "l1": names["l1"][yte1[i]], "l2": names["l2"][yte2[i]]}
                                           for i in idx],
           "codes": {m: {str(K): v for K, v in d.items()} for m, d in codes.items()},
           "cells": {m: {str(K): v for K, v in d.items()} for m, d in cells.items()},
           "select": sel, "ladder": ladder, "names": names,
           "base": {"l1": np.bincount(ytr1, minlength=len(names["l1"])).tolist(),
                    "l2": np.bincount(ytr2, minlength=len(names["l2"])).tolist()},
           "models": {"base": "Gemma 4 E2B (base)", "it": "Gemma 4 E2B-it (bf16)", "it2bit": "Gemma 4 E2B-it (2-bit IQ2_M)"}}
    with open(a.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    log("saved", a.out)


if __name__ == "__main__":
    main()
