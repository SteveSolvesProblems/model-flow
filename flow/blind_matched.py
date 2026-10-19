"""N2: are Gemma 4 E2B's turn neurons more readable than equally active ones? (plan in
research/NOTE_neurons_vs_cells.md, "N2"; follow-up to the blind labelling of 2026-10-18)

Units at layers 13 and 14 (neuron = input to down_proj):
  turn_var  the 8 neurons per layer with the largest write variance, var(a) |w|^2, as in blind_key.json
  matched   for each turn neuron, the neuron of the same layer with the nearest activation spread (std of a on tok_meta
            eval texts 100-129), outside the top 64 by write variance, without reuse
  random    the 8 random neurons per layer of the first blind set (blind_key.json)
Poles: the 6 tokens where a unit is most positive and the 6 where it is most negative, on the same token sample as
neuron_cells.py --tokens (first 1,500 DBpedia test articles, 14 tokens each). All 48 units are recomputed in this run
and shuffled with a fixed seed. Writes results/<tag>_units.md (for raters), results/<tag>_key.json (kept from them).
--exclude (added after the first run): the planned top-64 exclusion left no control within 2x of most turn neurons'
spread; --exclude 8 takes controls from outside the 8 selected per layer only.

DIAG_ATTN=eager python blind_matched.py --dtype float16     # P100
"""
import argparse
import json
import re

import numpy as np
import torch

from common import CACHE_DIRS, RESULTS, set_threads
from extract_hf import dbpedia, load_causal_lm
from lens import log

N_ART, MAX_LEN, BATCH = 6000, 64, 16
RANK_TEXTS, N_TOK_ART, TOK_PER_ART, LAYERS, TOP_EXCL = (100, 130), 1500, 14, (13, 14), 64


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--exclude", type=int, default=TOP_EXCL, help="controls come from outside the top N by write variance")
    ap.add_argument("--tag", default="blind2")
    a = ap.parse_args()
    set_threads()
    from transformers import AutoTokenizer
    MODEL = "google/gemma-4-E2B"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = load_causal_lm(MODEL, getattr(torch, a.dtype), "cuda")
    layers = {int(r.group(1)): m for n, m in model.named_modules() if (r := re.search(r"language_model\.layers\.(\d+)$", n))}
    cur = {}

    def grab(L):
        def f(mod, inp):
            cur[L] = inp[0]
        return f
    for L in LAYERS:
        layers[L].mlp.down_proj.register_forward_pre_hook(grab(L))

    # ranking and activation spread on the rank texts, as in neuron_cells.py
    hidden = layers[LAYERS[0]].mlp.down_proj.out_features
    meta = np.load(next(d / "tok_meta.npz" for d in CACHE_DIRS if (d / "tok_meta.npz").exists()), allow_pickle=True)
    s1, s2, col2, ntok = {L: 0 for L in LAYERS}, {L: 0 for L in LAYERS}, {}, 0
    for L in LAYERS:
        gw = layers[L].post_feedforward_layernorm(torch.ones(1, hidden, device="cuda", dtype=model.dtype)).float()[0]
        col2[L] = ((layers[L].mlp.down_proj.weight.float().T * gw) ** 2).sum(1)
    for t in meta["eval_texts"][slice(*RANK_TEXTS)]:
        model(**tok(str(t), truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda"), use_cache=False)
        for L in LAYERS:
            x = cur[L][0].float()
            s1[L], s2[L] = s1[L] + x.sum(0), s2[L] + (x ** 2).sum(0)
        ntok += x.shape[0]
    key_old = json.loads((RESULTS / "blind_key.json").read_text())
    picks, check = [], {}
    for L in LAYERS:
        var = s2[L] / ntok - (s1[L] / ntok) ** 2
        std = var.clamp(min=0).sqrt().cpu().numpy()
        vorder = torch.argsort(var * col2[L], descending=True).cpu().numpy()
        turn = [u["neuron"] for u in key_old if u["layer"] == str(L) and u["kind"] == "turn_var"]
        check[L] = {"turn_in_top8_this_run": int(len(set(turn) & set(vorder[:8].tolist()))),
                    "turn_std": [float(std[n]) for n in turn]}
        used = set(vorder[:a.exclude].tolist())
        for n in turn:
            cand = [m for m in np.argsort(np.abs(std - std[n])) if m not in used]
            used.add(int(cand[0]))
            picks += [("turn_var", L, n, f"{L}-{len(picks)}"), ("matched", L, int(cand[0]), f"{L}-{len(picks)}")]
            check[L].setdefault("matched_std_ratio", []).append(float(std[cand[0]] / std[n]))
        picks += [("random", L, u["neuron"], None) for u in key_old if u["layer"] == str(L) and u["kind"] == "random"]
    log("check", check)

    # token sample of neuron_cells.py --tokens
    texts = dbpedia(10, N_ART)[0]["test"][0]
    trng, where, acts = np.random.default_rng(0), [], {L: [] for L in LAYERS}
    for s in range(0, N_TOK_ART, BATCH):
        enc = tok(texts[s:s + BATCH], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
        model(**enc, use_cache=False)
        for b in range(enc["input_ids"].shape[0]):
            n = int(enc["attention_mask"][b].sum())
            ps = np.sort(trng.choice(np.arange(1, n), min(TOK_PER_ART, n - 1), replace=False))
            where += [(s + b, int(p)) for p in ps]
            for L in LAYERS:
                acts[L].append(cur[L][b, ps].float().cpu().numpy())
    acts = {L: np.concatenate(v) for L, v in acts.items()}
    ids = {}

    def context(j, p):
        if j not in ids:
            ids[j] = tok(texts[j], truncation=True, max_length=MAX_LEN)["input_ids"]
        t = ids[j]
        return (tok.decode(t[max(1, p - 8):p]) + " [[" + tok.decode(t[p:p + 1]) + "]] " + tok.decode(t[p + 1:p + 4])).replace("\n", " ")
    order = np.random.default_rng(19).permutation(len(picks))
    key, md = [], []
    for uid, i in enumerate(order):
        kind, L, n, pair = picks[i]
        v = acts[L][:, n]
        hi, lo = np.argsort(-v)[:6], np.argsort(v)[:6]
        key.append({"id": uid, "layer": str(L), "kind": kind, "neuron": int(n), "pair": pair})
        md += [f"## Unit {uid}", "Positive pole (top 6 tokens, marked [[ ]]):"] + ["  " + context(*where[k]) for k in hi] + \
              ["Negative pole:"] + ["  " + context(*where[k]) for k in lo] + [""]
    (RESULTS / f"{a.tag}_units.md").write_text("\n".join(md))
    (RESULTS / f"{a.tag}_key.json").write_text(json.dumps({"dtype": a.dtype, "exclude": a.exclude, "n_tokens": int(len(where)), "check": check,
                                                         "units": key}, indent=1))
    log("saved", len(key), "units")


if __name__ == "__main__":
    main()
