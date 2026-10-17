"""Are blocks a shared language? (exploratory; stated before running)
Skip one layer at a time (its contribution dropped: the layer's output is replaced by its input; the layer still runs,
so any keys/values it shares with later layers are unchanged) and, for models without shared keys/values, swap
adjacent layers k and k+1. Measure KL(clean || modified) and top-1 agreement of next-token predictions on n texts.
Predictions: (a) skipping hurts more where the geometry turns (1 - centered cosine across the layer), but block
membership should matter beyond the size of the layer's update; (b) swapping two layers inside a block hurts less than
swapping across a block boundary. The analysis (score_layer_skip.py) controls for update size.
Writes results/layer_skip_<name>.json.

python layer_skip.py --name qwen3_17b --model Qwen/Qwen3-1.7B --swap
"""
import argparse
import json
import re

import numpy as np
import torch

from common import CACHE_DIRS, RESULTS
from lens import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).eval()
    parent, layers = None, None
    for n, m in model.named_modules():
        if re.search(r"(language_model\.|^model\.)layers$", n) and isinstance(m, torch.nn.ModuleList):
            parent, layers = n, m
    L = len(layers)
    log(a.name, "layers module", parent, L)
    meta = np.load(next(d / "tok_meta.npz" for d in CACHE_DIRS if (d / "tok_meta.npz").exists()), allow_pickle=True)
    encs = [tok(str(t), truncation=True, max_length=64, return_tensors="pt") for t in meta["eval_texts"][: a.n]]

    skip = {"k": None}
    def hook(k):
        def f(mod, args, kwargs, out):
            if skip["k"] != k:
                return None
            h = args[0] if args else kwargs["hidden_states"]
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return f
    hooks = [layers[k].register_forward_hook(hook(k), with_kwargs=True) for k in range(L)]

    conds = [("skip", k) for k in range(L)] + ([("swap", k) for k in range(L - 1)] if a.swap else [])
    kl = {f"{c}_{k}": [] for c, k in conds}
    t1 = {f"{c}_{k}": [] for c, k in conds}
    with torch.no_grad():
        for i, e in enumerate(encs):
            skip["k"] = None
            clean = torch.log_softmax(model(**e).logits[0].float(), -1)
            for c, k in conds:
                if c == "skip":
                    skip["k"] = k
                    lp = torch.log_softmax(model(**e).logits[0].float(), -1)
                    skip["k"] = None
                else:
                    layers[k], layers[k + 1] = layers[k + 1], layers[k]
                    try:
                        lp = torch.log_softmax(model(**e).logits[0].float(), -1)
                    finally:
                        layers[k], layers[k + 1] = layers[k + 1], layers[k]
                key = f"{c}_{k}"
                kl[key].append((clean.exp() * (clean - lp)).sum(-1).numpy())
                t1[key].append((clean.argmax(-1) == lp.argmax(-1)).numpy())
            if i % 10 == 0:
                log(a.name, "text", i)
    for h in hooks:
        h.remove()
    res = {k: {"kl": float(np.concatenate(v).mean()), "top1": float(np.concatenate(t1[k]).mean())} for k, v in kl.items()}
    (RESULTS / f"layer_skip_{a.name}.json").write_text(json.dumps({"name": a.name, "model": a.model, "n_layers": L,
                                                                    "n_texts": len(encs), "results": res}))
    log("skip KL:", [round(res[f"skip_{k}"]["kl"], 3) for k in range(L)])
    if a.swap:
        log("swap KL:", [round(res[f"swap_{k}"]["kl"], 3) for k in range(L - 1)])
    log("saved")


if __name__ == "__main__":
    main()
