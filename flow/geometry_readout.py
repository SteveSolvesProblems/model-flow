"""Early readout (exploratory, not pre-registered): how early does each layer already give the model's final
next-token prediction? Token caches only. For 3,000 held-out token positions (the geometry_probe sample), at every layer:
  logit     final norm + the model's own output layer applied directly (the "logit lens")
  mapped    a linear map from layer l to the last layer (ridge, fit on the fit split, like a tuned-lens translator)
            followed by the output layer
Scores per layer: top-1 agreement with the final layer, KL(final || layer), top-1 accuracy on the true next token.
Early-exit estimates for the mapped readout: stop at the first layer whose top probability >= threshold; report the
mean share of layers run and agreement with the full model (an upper bound on savings: no exit overhead counted).
The last cached layer already carries the final norm; that is checked, not assumed.
Writes results/geometry_readout_<name>.json.

python geometry_readout.py --name tok_base --meta tok_meta.npz --model google/gemma-4-E2B
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from common import CACHE_DIRS, RESULTS, cache_dir, sample_idx, set_threads
from lens import layers_of, log

THRESH = [0.5, 0.7, 0.9, 0.95]


def weights(model):
    from safetensors import safe_open
    snap = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots/*/")
    cfg = json.loads(open(glob.glob(snap + "config.json")[0]).read())
    tc = cfg.get("text_config", cfg)
    want = {}
    for f in sorted(glob.glob(snap + "*.safetensors")):
        with safe_open(f, "pt") as h:                   # torch: the weights are bfloat16
            for k in h.keys():
                base = k.replace("model.language_model.", "model.")
                if base in ("model.norm.weight", "model.embed_tokens.weight", "lm_head.weight"):
                    want[base] = h.get_tensor(k).float().numpy()
    W = want.get("lm_head.weight", want["model.embed_tokens.weight"])
    return want["model.norm.weight"], W, tc.get("rms_norm_eps", 1e-6), tc.get("final_logit_softcapping")


def logprobs(H, W, cap):
    Z = H @ W.T
    if cap:
        Z = np.tanh(Z / cap) * cap
    Z -= Z.max(1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--n-fit", type=int, default=10000)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=250)
    a = ap.parse_args()
    set_threads()
    names = layers_of(a.name)
    g, W, eps, cap = weights(a.model)
    log(a.model, "output layer", W.shape, "softcap", cap)
    m = np.load(next(d / a.meta for d in CACHE_DIRS if (d / a.meta).exists()))
    ev = m["eval"]
    ne, nf = len(ev), np.load(cache_dir(names[0]) / "fit_X.npy", mmap_mode="r").shape[0]
    eidx = sample_idx(ne, a.n, a.seed)
    fidx = np.sort(np.random.default_rng(a.seed + 2).choice(nf, min(a.n_fit, nf), replace=False))
    nxt = np.full(ne, -1)
    ok = (ev[1:, 0] == ev[:-1, 0]) & (ev[1:, 1] == ev[:-1, 1] + 1)
    nxt[:-1][ok] = ev[1:, 2][ok]
    tgt = nxt[eidx]
    norm = lambda X: X / np.sqrt((X ** 2).mean(1, keepdims=True) + eps) * g
    ld = lambda nm, sp, ix: np.load(cache_dir(nm) / f"{sp}_X.npy", mmap_mode="r")[ix].astype(np.float32)
    Yf, Ye = ld(names[-1], "fit", fidx), ld(names[-1], "eval", eidx)
    # the cached last layer should already be normalized: dividing out the norm weight leaves unit RMS
    rms_check = float(np.sqrt(((Ye / g) ** 2).mean(1)).mean())
    log("last layer RMS after dividing out the norm weight (1.0 = already normalized):", round(rms_check, 3))
    ymu = Yf.mean(0)

    def run(H):
        out = []
        for s in range(0, len(H), a.chunk):
            out.append(logprobs(H[s:s + a.chunk], W, cap))
        return out

    final = run(Ye)
    fin_top = np.concatenate([lp.argmax(1) for lp in final])
    res = {k: [] for k in ("agree_logit", "agree_mapped", "kl_logit", "kl_mapped", "acc_logit", "acc_mapped", "maxp_mapped")}
    top_m = []
    t0 = time.time()
    for li, nm in enumerate(names):
        He = ld(nm, "eval", eidx)
        if li < len(names) - 1:
            Hf = ld(nm, "fit", fidx)
            mu = Hf.mean(0)
            Xc = Hf - mu
            G = Xc.T @ Xc
            G[np.diag_indices(len(G))] += a.lam * np.trace(G) / len(G)
            M = np.linalg.solve(G, Xc.T @ (Yf - ymu))
            mapped, direct = (He - mu) @ M + ymu, norm(He)
        else:
            mapped = direct = He
        for tag, H in (("logit", direct), ("mapped", mapped)):
            lps = run(H)
            top = np.concatenate([lp.argmax(1) for lp in lps])
            kl = np.concatenate([(np.exp(f) * (f - lp)).sum(1) for f, lp in zip(final, lps)])
            res[f"agree_{tag}"].append(float((top == fin_top).mean()))
            res[f"kl_{tag}"].append(float(kl.mean()))
            res[f"acc_{tag}"].append(float((top[tgt >= 0] == tgt[tgt >= 0]).mean()))
            if tag == "mapped":
                top_m.append(top)
                res["maxp_mapped"].append(np.concatenate([np.exp(lp.max(1)) for lp in lps]).tolist())
        log(a.name, li, f"agree logit {res['agree_logit'][-1]:.3f} mapped {res['agree_mapped'][-1]:.3f}  "
            f"KL logit {res['kl_logit'][-1]:.2f} mapped {res['kl_mapped'][-1]:.2f}  ({time.time() - t0:.0f}s)")
    L = len(names)
    T, P = np.array(top_m), np.array(res.pop("maxp_mapped"))
    stable = np.array([(T[l:] == fin_top).all(0).mean() for l in range(L)])     # top-1 never changes again from l on
    exits = {}
    for th in THRESH:
        hit = P >= th
        hit[-1] = True
        first = hit.argmax(0)
        exits[str(th)] = {"mean_layers_frac": float((first + 1).mean() / L),
                          "agree_with_final": float((T[first, np.arange(T.shape[1])] == fin_top).mean())}
    out = {"name": a.name, "model": a.model, "n": int(len(eidx)), "n_next_known": int((tgt >= 0).sum()),
           "layers": names, "last_layer_rms_check": rms_check, **res, "stable_from_layer": stable.tolist(),
           "early_exit_mapped": exits}
    (RESULTS / f"geometry_readout_{a.name}{'' if a.seed == 0 else f'_s{a.seed}'}.json").write_text(json.dumps(out))
    log("stable from layer (share):", [round(x, 2) for x in stable])
    log("early exit:", exits)


if __name__ == "__main__":
    main()
