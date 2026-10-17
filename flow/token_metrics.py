"""research/PREREG_baselines.md phase B: Skean et al.'s token-level layer scores, computed during the forward pass.
Per layer, averaged over the first N train documents (same shuffle as the caches), position 0 dropped:
  prompt entropy: α=1 matrix entropy of the trace-normalized Gram of a document's token representations;
  curvature: mean angle between consecutive difference vectors of the token trajectory.
Both datasets in one process, so the model loads once. Writes results/<name>__token_metrics.json (name, name_20ng).

python token_metrics.py --model Qwen/Qwen3-1.7B --name qwen3_17b --dtype float32
"""
import argparse
import json

import numpy as np
import torch

from common import RESULTS
from extract_hf import dbpedia, load_causal_lm, log, newsgroups


@torch.no_grad()
def scores(h, mask):
    """h: (B, T, d) one layer; mask: (B, T) valid tokens, position 0 already excluded. Returns per-doc
    (entropy, curvature) with NaN where a document is too short."""
    h = h.double() * mask.unsqueeze(-1)
    n = mask.sum(1)
    G = h @ h.transpose(1, 2)                                  # padded rows give zero eigenvalues
    tr = G.diagonal(dim1=1, dim2=2).sum(1).clamp_min(1e-30)
    lam = torch.linalg.eigvalsh(G / tr[:, None, None]).clamp_min(0)
    ent = -(torch.where(lam > 0, lam * lam.clamp_min(1e-300).log(), torch.zeros_like(lam))).sum(1)
    ent = torch.where(n >= 2, ent, torch.full_like(ent, float("nan")))
    v = h[:, 1:] - h[:, :-1]                                   # difference vectors
    vm = mask[:, 1:] * mask[:, :-1]
    cos = torch.nn.functional.cosine_similarity(v[:, 1:], v[:, :-1], dim=-1).clamp(-1, 1)
    am = vm[:, 1:] * vm[:, :-1]
    ang = (torch.arccos(cos) * am).sum(1) / am.sum(1).clamp_min(1)
    ang = torch.where(am.sum(1) >= 1, ang, torch.full_like(ang, float("nan")))
    return ent.cpu().numpy(), ang.cpu().numpy()


@torch.no_grad()
def run(model, tok, texts, bs, max_len, dev):
    ent, curv = None, None
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        mask = enc["attention_mask"].clone()
        mask[:, 0] = 0                                           # drop position 0 (attention-sink norms)
        if ent is None:
            ent, curv = [[] for _ in hs], [[] for _ in hs]
        mask = mask.to(dev)
        for i, h in enumerate(hs):
            h = h.to(dev)
            if not torch.isfinite(h[mask.bool()]).all():
                raise RuntimeError(f"non-finite hidden state at layer {i}; rerun in float32")
            e, c = scores(h, mask)
            ent[i].append(e)
            curv[i].append(c)
    E = [np.concatenate(x) for x in ent]
    C = [np.concatenate(x) for x in curv]
    return [{"layer": i, "prompt_entropy": float(np.nanmean(E[i])), "curvature": float(np.nanmean(C[i])),
             "n_entropy": int(np.isfinite(E[i]).sum()), "n_curvature": int(np.isfinite(C[i]).sum())}
            for i in range(len(E))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--gpu-mem", default="14GiB")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = load_causal_lm(a.model, getattr(torch, a.dtype), "cuda", a.offload, a.gpu_mem)
    for suffix, texts in (("", dbpedia(a.n, 0)[0]["train"][0]), ("_20ng", newsgroups()[0]["train"][0][:a.n])):
        rows = run(model, tok, texts[:a.n], a.batch, a.max_len, "cuda")
        out = {"cache": a.name + suffix, "model": a.model, "dtype": a.dtype, "n_docs": a.n, "rows": rows}
        (RESULTS / f"{a.name}{suffix}__token_metrics.json").write_text(json.dumps(out, indent=1))
        log(a.name + suffix, "done", len(rows), "layers")


if __name__ == "__main__":
    main()
