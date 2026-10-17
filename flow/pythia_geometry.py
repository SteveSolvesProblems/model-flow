"""When do blocks form during training? (exploratory) Pythia-410M (seed 0) at the F2-d checkpoints: run the same
3,000 DBpedia test articles as geometry_probe.py (mean-pooled, <= 64 tokens), and compute the geometry_probe matrices
(self_top1, cos_centered, knn10_overlap over layers) directly, without caching hidden states. Records each checkpoint's
weight fingerprint (Pythia hub branches have duplicated weights before; see PREREG_flow.md Amendment 3).
Writes results/pythia_geometry.json (one entry per step, saved after each).

HF_HUB_CACHE=data/hf python pythia_geometry.py
"""
import argparse
import json

import numpy as np
import torch

from common import RESULTS
from extract_hf import dbpedia, weights_fingerprint
from lens import log

STEPS = [0, 128, 256, 512, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 16000, 32000, 64000, 143000]


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def probe(states):
    cen = [unit(X - X.mean(0)) for X in states]
    L = len(cen)
    knn = [np.argsort(-(c @ c.T), axis=1)[:, 1:11] for c in cen]
    T, C, K = np.zeros((L, L)), np.zeros((L, L)), np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            C[i, j] = float((cen[i] * cen[j]).sum(1).mean())
            T[i, j] = float(((cen[i] @ cen[j].T).argmax(1) == np.arange(len(cen[i]))).mean())
            if j >= i:
                K[i, j] = K[j, i] = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(knn[i], knn[j])])
    return T, C, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoTokenizer, GPTNeoXForCausalLM
    data, _ = dbpedia(1, 6000)
    texts = data["test"][0]
    idx = np.sort(np.random.default_rng(0).choice(len(texts), a.n, replace=False))
    texts = [texts[i] for i in idx]
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    path = RESULTS / "pythia_geometry.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for st in STEPS:
        if str(st) in out:
            continue
        model = GPTNeoXForCausalLM.from_pretrained(a.model, revision=f"step{st}", dtype=torch.float32).eval()
        fp = weights_fingerprint(model)
        pooled = None
        with torch.no_grad():
            for s in range(0, len(texts), a.batch):
                enc = tok(texts[s:s + a.batch], padding=True, truncation=True, max_length=64, return_tensors="pt")
                hs = model(**enc, output_hidden_states=True).hidden_states
                m = enc["attention_mask"].unsqueeze(-1).float()
                p = [((h * m).sum(1) / m.sum(1)).numpy() for h in hs]
                pooled = p if pooled is None else [np.concatenate([x, y]) for x, y in zip(pooled, p)]
        T, C, K = probe(pooled)
        out[str(st)] = {"fingerprint": fp, "self_top1": np.round(T, 4).tolist(), "cos_centered": np.round(C, 4).tolist(),
                        "knn10_overlap": np.round(K, 4).tolist()}
        path.write_text(json.dumps(out))
        Lh = len(T)
        log("step", st, fp[:10], "adjacent self-match median", round(float(np.median([T[i, i + 1] for i in range(1, Lh - 1)])), 3),
            "+6 layers", round(float(np.mean([T[i, i + 6] for i in range(1, Lh - 6)])), 3))
        del model
    log("done")


if __name__ == "__main__":
    main()
