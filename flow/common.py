"""Shared utilities: paths, threading, caches, result records."""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("DIAG_CACHE", ROOT / "cache"))
RESULTS = Path(os.environ.get("DIAG_RESULTS", ROOT / "results"))
# Extra cache roots searched after CACHE (colon-separated), e.g. the internal disk when CACHE is the external drive.
CACHE_DIRS = [CACHE] + [Path(p) for p in os.environ.get("DIAG_CACHE_EXTRA", "").split(":") if p]


def cache_dir(name):
    for c in CACHE_DIRS:
        if (c / name).exists():
            return c / name
    return CACHE / name
SEEDS = [0, 1, 2, 3, 4]


def set_threads(n=None):
    n = n or int(os.environ.get("DIAG_THREADS", "14"))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, str(n))
    try:
        import torch
        torch.set_num_threads(n)
    except ImportError:
        pass
    try:
        import faiss
        faiss.omp_set_num_threads(n)
    except ImportError:
        pass


def code_hash():
    h = hashlib.sha1()
    for p in sorted(ROOT.glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:10]


# ── feature caches: CACHE/<name>/{split}_X.npy, {split}_y_<label>.npy, meta.json ──

def save_cache(name, split, X, labels, meta=None):
    d = CACHE / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{split}_X.npy", X.astype(np.float16 if X.dtype == np.float16 else np.float32))
    for lname, y in labels.items():
        np.save(d / f"{split}_y_{lname}.npy", np.asarray(y))
    if meta is not None:
        (d / "meta.json").write_text(json.dumps(meta, indent=2))


def load_cache(name, split, label=None):
    d = cache_dir(name)
    X = np.load(d / f"{split}_X.npy", mmap_mode="r").astype(np.float32)
    if label is None:
        return X
    return X, np.load(d / f"{split}_y_{label}.npy")


def cache_meta(name):
    return json.loads((cache_dir(name) / "meta.json").read_text())


def save_result(name, exp, payload):
    RESULTS.mkdir(parents=True, exist_ok=True)
    rec = {"cache": name, "exp": exp, "code_hash": code_hash(),
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    for n in (name, f"{name}_L00"):          # results are usually named after a layer-cache prefix
        try:
            rec["cache_meta"] = cache_meta(n)
            break
        except FileNotFoundError:
            pass
    out = RESULTS / f"{name}__{exp}.json"
    out.write_text(json.dumps(rec, indent=2, default=_json_default))
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def mean_sd(xs):
    xs = np.asarray(xs, dtype=float)
    return {"mean": float(xs.mean()), "sd": float(xs.std(ddof=1)) if len(xs) > 1 else 0.0, "n": len(xs)}


def sample_idx(total, n, seed=0):
    """Sorted sample of n item indices. seed 0 reproduces the original sample; any other seed draws from the items the
    seed-0 sample left out (a disjoint replication sample) while enough remain."""
    rng0 = np.random.default_rng(0)
    base = np.sort(rng0.choice(total, min(n, total), replace=False))
    if seed == 0:
        return base
    rest = np.setdiff1d(np.arange(total), base)
    if len(rest) >= n:
        return np.sort(np.random.default_rng(seed).choice(rest, n, replace=False))
    return np.sort(np.random.default_rng(seed).choice(total, min(n, total), replace=False))
