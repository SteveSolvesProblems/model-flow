"""Where does Gemma 4 E2B's next-token accuracy go? gemma_sanity.py found 56% top-1 on the DBpedia eval texts with a
fresh float32 / eager-attention CPU run, against 17.9% from the cached (float16, P100, SDPA, padded batches) states.
Here, on the first 40 eval texts of tok_meta.npz, batch 1, CPU, for four settings (float32 or bfloat16 x eager or
SDPA attention):
  top1          accuracy on the true next token (positions 1..n-1, as in gemma_sanity.py)
  cos_ref[l]    mean cosine, per layer, between the setting's token states and the float32-eager states
  cos_cache[l]  mean cosine between the setting's token states and the cached tok_base states (same doc, position)
  turn[l]       centered cosine across layer l (all tokens of the 40 texts), to see whether the turn at 14 is there
Writes results/gemma_dtype.json.
"""
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import CACHE_DIRS, RESULTS, load_cache, set_threads
from lens import layers_of, log

MODEL, N, MAX_LEN = "google/gemma-4-E2B", 40, 64


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)


@torch.no_grad()
def run(dtype, attn, texts, tok):
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, attn_implementation=attn).eval()
    hit = n = 0
    states = []                                     # per doc: (layers, tokens, d)
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_LEN)["input_ids"]
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        lg = out.logits[0].float()
        tgt = ids[0, 2:]
        hit += float((lg[1:-1].argmax(-1) == tgt).sum()); n += len(tgt)
        states.append(np.stack([h[0].float().numpy() for h in out.hidden_states]))
    del model
    return hit / n, states


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="tok_base")
    ap.add_argument("--meta", default="tok_meta.npz")
    ap.add_argument("--only-fp32", action="store_true")
    a = ap.parse_args()
    set_threads()
    meta = np.load(next(c / a.meta for c in CACHE_DIRS if (c / a.meta).exists()), allow_pickle=True)
    texts = [str(t) for t in meta["eval_texts"][:N]]
    ev = meta["eval"]
    rows = np.flatnonzero(ev[:, 0] < N)             # cached token rows of these docs (positions >= 1)
    names = layers_of(a.cache)
    cache = [load_cache(nm, "eval")[rows] for nm in names]
    tok = AutoTokenizer.from_pretrained(MODEL)
    res, ref = {}, None
    settings = ((torch.float32, "eager"), (torch.float32, "sdpa"), (torch.bfloat16, "eager"), (torch.bfloat16, "sdpa"))
    out_name = "gemma_dtype.json" if a.cache == "tok_base" else f"gemma_dtype_{a.cache}.json"
    for dtype, attn in (settings[:1] if a.only_fp32 else settings):
        key = f"{str(dtype).split('.')[-1]}_{attn}"
        top1, st = run(dtype, attn, texts, tok)
        L = st[0].shape[0]
        # token states at the cached rows, layer by layer
        H = np.stack([np.stack([st[d][l, p] for d, p in ev[rows, :2]]) for l in range(L)])
        if ref is None:
            ref = H
        r = {"top1": top1,
             "cos_ref": [float((unit(H[l]) * unit(ref[l])).sum(1).mean()) for l in range(L)],
             "cos_cache": [float((unit(H[l]) * unit(cache[l].astype(np.float32))).sum(1).mean()) for l in range(min(L, len(cache)))],
             "turn": [float((unit(H[l] - H[l].mean(0)) * unit(H[l + 1] - H[l + 1].mean(0))).sum(1).mean()) for l in range(L - 1)]}
        res[key] = r
        log(key, "top1", round(top1, 3), "cos_cache L1/14/20/last", [round(r["cos_cache"][l], 3) for l in (1, 14, 20, -1)],
            "turn 13/14", [round(r["turn"][l], 3) for l in (13, 14)])
        (RESULTS / out_name).write_text(json.dumps(res))
    cturn = [float((unit(c - c.mean(0)) * unit(d - d.mean(0))).sum(1).mean())
             for c, d in zip([x.astype(np.float32) for x in cache[:-1]], [x.astype(np.float32) for x in cache[1:]])]
    res["cache_turn"] = cturn
    (RESULTS / out_name).write_text(json.dumps(res))
    log("cache turn 13/14", [round(cturn[l], 3) for l in (13, 14)], "saved")


if __name__ == "__main__":
    main()
