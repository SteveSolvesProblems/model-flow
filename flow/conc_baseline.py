"""How concentrated is an MLP write at every layer? Baseline for neuron_cells.py test B (review follow-up; plan in
research/NOTE_neurons_vs_cells.md, "S"). Writes results/conc_baseline_<tag>.json.

Per token (8 random tokens from each of tok_meta eval texts 0-9, <bos> skipped): the write w = sum_i a_i W_i (after the post-norm weight where the
model has one), neurons added largest |a_i| |W_i| first; k50 / k90 = fewest neurons leaving <= 50% / 10% of ||w||^2
unexplained (vector residual, cross terms included). Also the participation ratio of the per-neuron energies.

python conc_baseline.py --model google/gemma-4-E2B --tag e2b
"""
import argparse
import json
import re

import numpy as np
import torch

from common import CACHE_DIRS, RESULTS, set_threads
from extract_hf import load_causal_lm
from lens import log

N_TEXTS, PER_TEXT, MAX_LEN = 10, 8, 64


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B")
    ap.add_argument("--tag", default="e2b")
    a = ap.parse_args()
    set_threads()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_causal_lm(a.model, torch.bfloat16, "cuda")
    layers = {int(r.group(1)): m for n, m in model.named_modules() if (r := re.search(r"(?:^|\.)(?:language_model|model)\.layers\.(\d+)$", n))}
    cur = {}
    for L, m in layers.items():
        m.mlp.down_proj.register_forward_pre_hook(lambda mod, inp, L=L: cur.__setitem__(L, inp[0]))
    hidden = layers[0].mlp.down_proj.out_features
    def wd(L):                                                         # built per layer: all in fp32 does not fit
        m = layers[L]
        W = m.mlp.down_proj.weight.float().T
        if hasattr(m, "post_feedforward_layernorm"):
            W = W * m.post_feedforward_layernorm(torch.ones(1, hidden, device="cuda", dtype=torch.bfloat16)).float()[0]
        return W
    rng = np.random.default_rng(0)
    width = {}
    meta = np.load(next(d / "tok_meta.npz" for d in CACHE_DIRS if (d / "tok_meta.npz").exists()), allow_pickle=True)
    stats = {L: {"k50": [], "k90": [], "pr": []} for L in layers}
    acts = {L: [] for L in layers}
    for s in meta["eval_texts"][:N_TEXTS]:
        model(**tok(str(s), truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda"), use_cache=False)
        n = cur[0].shape[1]
        ps = torch.as_tensor(np.sort(rng.choice(np.arange(1, n), min(PER_TEXT, n - 1), replace=False)), device="cuda")
        for L in layers:
            acts[L].append(cur[L][0, ps].float())
    for L in layers:
        W = wd(L)
        width[L] = W.shape[0]
        for x in torch.cat(acts[L]):
            C = x[:, None] * W
            w = C.sum(0)
            tot = (w ** 2).sum()
            en = (C ** 2).sum(1)
            idx = torch.argsort(en, descending=True)
            left = ((w - torch.cumsum(C[idx], 0)) ** 2).sum(1) / tot
            stats[L]["k50"].append(int(torch.nonzero(left <= 0.5)[0]) + 1)
            stats[L]["k90"].append(int(torch.nonzero(left <= 0.1)[0]) + 1)
            stats[L]["pr"].append(float(en.sum() ** 2 / (en ** 2).sum()))
        del W
        log("layer", L)
    res = {"model": a.model, "n_texts": N_TEXTS, "layers": {}}
    for L in sorted(layers):
        wdt = width[L]
        res["layers"][L] = {"width": wdt, **{k: float(np.median(v)) for k, v in stats[L].items()},
                            **{f"{k}_frac": float(np.median(v) / wdt) for k, v in stats[L].items()}}
        log(L, res["layers"][L])
    (RESULTS / f"conc_baseline_{a.tag}.json").write_text(json.dumps(res, indent=1))
    log("saved")


if __name__ == "__main__":
    main()
