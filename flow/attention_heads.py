"""Per-head attention behaviour (exploratory, not pre-registered), to connect the residual geometry blocks to
classic mechanistic-interpretability head types. CPU forward passes with eager attention.
On n DBpedia test texts (<= 64 tokens, the same texts as the token caches), per layer and head, averaged over query
positions q >= 4:
  entropy   attention entropy / log(number of visible keys) (0 = one key, 1 = uniform)
  sink      attention to position 0 (Gemma: <bos>; Qwen3: first text token; ModernBERT: [CLS])
  sink_end  ModernBERT only: attention to the final [SEP]
  prev      attention to q-1          self   attention to q
  dist      mean |q - k| over keys, divided by q
On n_ind random token sequences of length 32 repeated twice, per head:
  induction attention from a position in the second copy to the position just after its earlier occurrence
            (for the bidirectional ModernBERT this is closer to a duplicate-token score)
Note: with <= 64 tokens, Gemma's sliding window (512) and ModernBERT's local window (128) never bind, so local/sliding
and global/full layers see the same keys here.
Writes results/attention_heads_<name>.json.

python attention_heads.py --name gemma4_e2b --model google/gemma-4-E2B --meta tok_meta.npz
"""
import argparse
import json

import numpy as np
import torch

from common import CACHE_DIRS, RESULTS
from lens import log


def head_stats(att, causal, q0=4):
    """att: (H, T, T) attention for one text. Returns dict of (H,) arrays."""
    H, T, _ = att.shape
    qs = np.arange(q0, T)
    A = att[:, qs, :]                                                    # H x Q x T
    nk = (qs + 1) if causal else np.full(len(qs), T)
    ent = -(A * np.log(A + 1e-12)).sum(-1) / np.log(nk)[None, :]
    k = np.arange(T)
    dist = (A * np.abs(qs[:, None] - k[None, :])[None]).sum(-1) / qs[None, :]
    out = {"entropy": ent.mean(1), "sink": A[:, :, 0].mean(1), "prev": A[:, np.arange(len(qs)), qs - 1].mean(1),
           "self": A[:, np.arange(len(qs)), qs].mean(1), "dist": dist.mean(1)}
    if not causal:
        out["sink_end"] = A[:, :, T - 1].mean(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--meta", default="tok_meta.npz")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-ind", type=int, default=50)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    causal = "modernbert" not in a.model.lower()
    tok = AutoTokenizer.from_pretrained(a.model)
    load = AutoModelForCausalLM if causal else AutoModel
    model = load.from_pretrained(a.model, dtype=torch.bfloat16 if causal else torch.float32,
                                 attn_implementation="eager").eval()
    meta = np.load(next(d / a.meta for d in CACHE_DIRS if (d / a.meta).exists()), allow_pickle=True)
    texts = [str(t) for t in meta["eval_texts"][: a.n]]
    acc = None
    with torch.no_grad():
        for n, t in enumerate(texts):
            enc = tok(t, truncation=True, max_length=64, return_tensors="pt")
            if enc["input_ids"].shape[1] < 12:
                continue
            atts = model(**enc, output_attentions=True).attentions
            st = [head_stats(x[0].float().numpy(), causal) for x in atts]
            if acc is None:
                acc = [{k: np.zeros_like(v) for k, v in s.items()} for s in st]
                cnt = 0
            for A, s in zip(acc, st):
                for k, v in s.items():
                    A[k] += v
            cnt += 1
            if n % 50 == 0:
                log(a.name, "text", n)
        stats = [{k: (v / cnt).tolist() for k, v in A.items()} for A in acc]
        # induction: random tokens drawn from the texts' own vocabulary, repeated twice
        rng = np.random.default_rng(0)
        pool = np.unique(np.concatenate([tok(t)["input_ids"] for t in texts[:100]]))
        pool = pool[~np.isin(pool, tok.all_special_ids)]
        ind = None
        for r in range(a.n_ind):
            seq = rng.choice(pool, 32, replace=False)
            if not causal:
                pre = [tok.cls_token_id]
            else:
                pre = [tok.bos_token_id] if tok("x")["input_ids"][0] == tok.bos_token_id else []
            post = [tok.sep_token_id] if not causal else []
            ids = torch.tensor([list(pre) + list(seq) + list(seq) + post])
            atts = model(input_ids=ids, output_attentions=True).attentions
            o = len(pre)
            qpos = np.arange(o + 32 + 1, o + 64)                          # second copy, from its 2nd token
            kpos = qpos - 32 + 1                                          # token after the earlier occurrence
            sc = [x[0].float().numpy()[:, qpos, kpos].mean(1) for x in atts]
            ind = sc if ind is None else [p + s for p, s in zip(ind, sc)]
            log(a.name, "induction", r) if r % 25 == 0 else None
        for s, v in zip(stats, ind):
            s["induction"] = (v / a.n_ind).tolist()
    out = {"name": a.name, "model": a.model, "causal": causal, "n_texts": cnt, "n_induction": a.n_ind,
           "n_layers": len(stats), "n_heads": len(stats[0]["entropy"]), "layers": stats}
    (RESULTS / f"attention_heads_{a.name}.json").write_text(json.dumps(out))
    for li, s in enumerate(stats):
        log(a.name, li, {k: round(float(np.mean(v)), 3) for k, v in s.items()},
            "max induction", round(max(s["induction"]), 2), "max prev", round(max(s["prev"]), 2))
    log("saved")


if __name__ == "__main__":
    main()
