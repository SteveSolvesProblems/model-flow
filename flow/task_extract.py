"""research/PREREG_overnight.md E2: Banking77 and STS-B features from every layer, last-token and mean pooling.

python task_extract.py --model Qwen/Qwen3-1.7B --name qwen3_17b --data stsb --dtype float32
Writes <cache>/<data>_<name>_{last,mean}_L##/{train,test}_X.npy and labels:
  banking77: {split}_y_intent.npy
  stsb: train = unique train sentences; test = the test pairs' sentence1 then sentence2, test_y_score.npy (gold, per pair)
"""
import argparse
import json

import numpy as np
import torch

from common import CACHE
from extract_hf import load_causal_lm, log


def data(name):
    from datasets import load_dataset
    if name == "banking77":
        ds = load_dataset("mteb/banking77")
        return {s: (list(ds[s]["text"]), {"intent": np.array(ds[s]["label"])}) for s in ("train", "test")}
    ds = load_dataset("sentence-transformers/stsb")
    tr = sorted(set(ds["train"]["sentence1"]) | set(ds["train"]["sentence2"]))
    te = ds["test"]
    return {"train": (tr, {}), "test": (list(te["sentence1"]) + list(te["sentence2"]), {"score": np.array(te["score"])})}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True, choices=["banking77", "stsb"])
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=64)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    m = load_causal_lm(a.model, getattr(torch, a.dtype), "cuda")
    for split, (texts, labels) in data(a.data).items():
        out = None
        for s in range(0, len(texts), a.batch):
            enc = tok(texts[s:s + a.batch], padding=True, truncation=True, max_length=a.max_len, return_tensors="pt").to("cuda")
            hs = m(**enc, output_hidden_states=True, use_cache=False).hidden_states
            mask = enc["attention_mask"]
            last = mask.sum(1) - 1
            ar = torch.arange(len(last), device=last.device)
            if out is None:
                out = {p: [] for p in ("last", "mean")}
                for p in out:
                    for li in range(len(hs)):
                        d = CACHE / f"{a.data}_{a.name}_{p}_L{li:02d}"
                        d.mkdir(parents=True, exist_ok=True)
                        out[p].append(np.lib.format.open_memmap(d / f"{split}_X.npy", "w+", np.float16, (len(texts), hs[0].shape[-1])))
            mk = mask.unsqueeze(-1).float()
            for li, h in enumerate(hs):
                hl = h[ar, last].float()
                hm = (h.float() * mk).sum(1) / mk.sum(1)
                if not (torch.isfinite(hl).all() and torch.isfinite(hm).all()):
                    raise RuntimeError(f"non-finite at layer {li}; rerun in float32")
                out["last"][li][s:s + len(hl)] = hl.cpu().numpy().astype(np.float16)
                out["mean"][li][s:s + len(hm)] = hm.cpu().numpy().astype(np.float16)
            if s % (a.batch * 100) == 0:
                log(a.data, split, s, len(texts))
        for p in out:
            for li, arr in enumerate(out[p]):
                arr.flush()
                d = CACHE / f"{a.data}_{a.name}_{p}_L{li:02d}"
                for k, v in labels.items():
                    np.save(d / f"{split}_y_{k}.npy", v)
                (d / "meta.json").write_text(json.dumps({"model": a.model, "layer": li, "pool": p, "dtype": a.dtype,
                                                         "data": a.data, "max_len": a.max_len}))
    log("done")


if __name__ == "__main__":
    main()
