"""Is Gemma 4 E2B's low next-token accuracy on our texts (17.9%, Qwen3 52.9%) real model behaviour or a pipeline
artifact? Independent of extract_hf.py / token_extract.py: fresh AutoModelForCausalLM loads, float32, eager attention,
CPU, default tokenizer settings (BOS as the tokenizer adds it). Per model and text set: top-1 accuracy on the true next
token and mean NLL (nats/token), over positions 1..n-1 (the first prediction, from BOS alone, is skipped).
Text sets: the first 200 DBpedia eval texts of tok_meta.npz, and 200 WikiText-103 test paragraphs (>= 200 chars),
both cut to 64 tokens. Writes results/gemma_sanity.json.

python gemma_sanity.py --models google/gemma-4-E2B google/gemma-4-E2B-it Qwen/Qwen3-1.7B-Base
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import CACHE_DIRS, RESULTS, set_threads
from lens import log

N, MAX_LEN = 200, 64


def texts():
    meta = np.load(next(c / "tok_meta.npz" for c in CACHE_DIRS if (c / "tok_meta.npz").exists()), allow_pickle=True)
    dbp = [str(t) for t in meta["eval_texts"][:N]]
    from datasets import load_dataset
    wt = [t.strip() for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"]
          if len(t.strip()) >= 200 and not t.strip().startswith("=")][:N]
    return {"dbpedia": dbp, "wikitext": wt}


@torch.no_grad()
def score(model, tok, txts):
    hit = n = 0
    nll = 0.0
    per_doc = []
    for t in txts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_LEN)["input_ids"]
        logits = model(input_ids=ids).logits[0].float()
        tgt = ids[0, 2:]                      # skip the prediction made from BOS alone
        lg = logits[1:-1]
        lp = torch.log_softmax(lg, -1)
        h = (lg.argmax(-1) == tgt).float()
        hit += float(h.sum()); n += len(tgt)
        nll += float(-lp[torch.arange(len(tgt)), tgt].sum())
        per_doc.append(float(h.mean()) if len(tgt) else float("nan"))
    return {"top1": hit / n, "nll": nll / n, "n_tokens": n, "first_ids": tok(txts[0])["input_ids"][:5]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    set_threads()
    T = texts()
    f = RESULTS / "gemma_sanity.json"
    out = json.loads(f.read_text()) if f.exists() else {}
    for m in a.models:
        tok = AutoTokenizer.from_pretrained(m)
        model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float32, attn_implementation="eager").eval()
        out[m] = {k: score(model, tok, v) for k, v in T.items()}
        log(m, {k: (round(v["top1"], 3), round(v["nll"], 3)) for k, v in out[m].items()})
        f.write_text(json.dumps(out, indent=1))
        del model
    log("saved")


if __name__ == "__main__":
    main()
