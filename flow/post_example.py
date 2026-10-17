"""Worked example for the Model Flow post: one article's 10 nearest neighbours before and after Gemma 4 E2B's turn.

Same sample and neighbour definition as geometry_probe.py (3,000 test articles, seed 0, centered cosine, 10 nearest
within the sample). Layers 13 and 15 straddle the turn (layer 14). The article shown is chosen by a fixed rule, not by
eye: the first article in sample order whose neighbour overlap equals the sample median and whose title is at most 40
characters.
Writes results/post_example.json.

HF_HOME=... python post_example.py
"""
import json

import numpy as np

from common import RESULTS, load_cache, sample_idx
from extract_hf import dbpedia

A, B, N = 13, 15, 3000


def knn(X):
    C = X - X.mean(0)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    return C, np.argsort(-(C @ C.T), axis=1)[:, 1:11]


def title(text):
    for sep in (" (", " is ", " was ", " are ", ", ", ": "):
        text = text.split(sep)[0]
    return text.strip().rstrip(",:;")[:80]


def main():
    data, _ = dbpedia(24000, 6000)
    texts, labels = data["test"]
    XA, XB = load_cache(f"gemma4_e2b_L{A:02d}", "test"), load_cache(f"gemma4_e2b_L{B:02d}", "test")
    idx = sample_idx(len(XA), N, 0)
    CA, ka = knn(XA[idx])
    CB, kb = knn(XB[idx])
    ov = np.array([len(set(a) & set(b)) for a, b in zip(ka, kb)])
    med = int(np.median(ov))
    pick = next(i for i in range(N) if ov[i] == med and len(title(texts[idx[i]])) <= 40)
    t = lambda j: {"title": title(texts[idx[j]]), "l1": labels["l1"][idx[j]].item() if hasattr(labels["l1"][idx[j]], "item") else int(labels["l1"][idx[j]]),
                   "l3": int(labels["l3"][idx[j]])}
    out = {"model": "gemma4_e2b", "layers": [A, B], "n": N, "median_overlap": med, "mean_overlap": float(ov.mean()) / 10,
           "article": {**t(pick), "text": texts[idx[pick]][:300]},
           "centered_cos_self": float(CA[pick] @ CB[pick]),
           "before": [{**t(j), "kept": bool(j in set(kb[pick]))} for j in ka[pick]],
           "after": [{**t(j), "kept": bool(j in set(ka[pick]))} for j in kb[pick]]}
    (RESULTS / "post_example.json").write_text(json.dumps(out, indent=1))
    print(out["article"]["title"], "| overlap", med, "| self cos", round(out["centered_cos_self"], 3))
    for side in ("before", "after"):
        print(side, [(r["title"], r["kept"]) for r in out[side]])


if __name__ == "__main__":
    main()
