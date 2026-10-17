"""Seed-0 and seed-1 K=512 test cells for every layer of a cache (the same frame and k-means as rule S), plus seed-0
centroid distances. Shared by research/PREREG_overnight.md E1 (P2), E3 and E5. Writes results/codes/<model>.npz:
  c0, c1   (L, N_test) int16   test cells for seeds 0 and 1
  d1, d2   (L, N_test) float16 distances to the nearest and second-nearest seed-0 centroid (whiten-64 frame)

python partition_codes.py --model qwen35_4b [--layers 25]
"""
import argparse

import numpy as np

from common import RESULTS, load_cache, set_threads
from lens import frame, layers_of, log
from partition import assign, kmeans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="only these layers (others left as -1)")
    ap.add_argument("--k", type=int, default=512)
    a = ap.parse_args()
    set_threads()
    names = layers_of(a.model)
    L = len(names)
    n = len(load_cache(names[0], "test"))
    c0 = np.full((L, n), -1, np.int16)
    c1 = np.full((L, n), -1, np.int16)
    d1 = np.zeros((L, n), np.float16)
    d2 = np.zeros((L, n), np.float16)
    for li, name in enumerate(names):
        if a.layers is not None and li not in a.layers:
            continue
        Ftr, Fte = frame(load_cache(name, "train"), load_cache(name, "test"))
        I0, D0 = assign(Fte, kmeans(Ftr, a.k, seed=0, nredo=1))
        I1, _ = assign(Fte, kmeans(Ftr, a.k, seed=1, nredo=1))
        c0[li], c1[li] = I0, I1
        d1[li], d2[li] = D0[:, 0], D0[:, 1]
        log(a.model, li)
    out = RESULTS / "codes"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{a.model}.npz", c0=c0, c1=c1, d1=d1, d2=d2, layers=np.arange(L))


if __name__ == "__main__":
    main()
