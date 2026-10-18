"""Macro -> micro links on Gemma 4 E2B (exploratory; plan and predictions in research/NOTE_neurons_vs_cells.md, M1-M3).

Neuron = input to down_proj at a decoder layer. One GPU pass collects every layer's neurons (mean over tokens) for the
first 1,500 test articles (the flow map's set) and runs M3; M1 and M2 then run on the CPU from the cached residuals.
  M1  flow-map splits (consensus k-means bins, as sankey_validity.py) vs the best single neuron separating the children
  M2  neurons matched across layers by activation: are their read vectors aligned within a block, across the turn,
      and across the turn after the Procrustes rotation between the residual clouds?
  M3  single-neuron mean ablations (KL) vs size-based predictions, with and without the residual's norm
Writes results/macro_micro.json.

python macro_micro.py
"""
import json
import re

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from common import CACHE_DIRS, RESULTS, load_cache, set_threads
from extract_hf import dbpedia, load_causal_lm
from lens import log
from neuron_cells import ranks, unit
from partition import assign, kmeans

MODEL, NAME, N_DOCS, N_FIT, MAX_LEN, BATCH = "google/gemma-4-E2B", "gemma4_e2b", 1500, 10000, 64, 16
N_ROT, N_CONS = 5, 20
M2_PAIRS = {"block_3_12": [(4, 6), (6, 8), (8, 10), (10, 12)], "block_15_34": [(17, 19), (20, 22), (25, 27), (30, 32)],
            "turn": [(11, 15), (12, 15), (12, 16)]}
M3_LAYERS, M3_N, M3_TEXTS, M3_RANK, M3_B = (8, 13, 14, 22), 200, 50, (100, 130), 4


def max_auc(Z, y):
    """Best single column of Z separating y (bool) from ~y, two-sided AUC."""
    R = ranks(Z)
    n1, n0 = y.sum(), (~y).sum()
    auc = (R[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(np.maximum(auc, 1 - auc).max())


def pca64(Xtr, Xte):
    mu = Xtr.mean(0)
    A, B = Xtr - mu, Xte - mu
    m = A.mean(0)
    W = np.linalg.svd(A - m, full_matrices=False)[2][:64].T
    return (A - m) @ W, (B - m) @ W


def consensus(Ztr, Zte, K):
    n = len(Zte)
    co = np.zeros((n, n), np.float32)
    for s in range(N_CONS):
        c = assign(Zte, kmeans(Ztr, K, seed=s, nredo=1))[0]
        co += c[:, None] == c[None, :]
    D = 1 - co / N_CONS
    np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform(D, checks=False), "average"), K, "maxclust") - 1


@torch.no_grad()
def gpu_pass():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = load_causal_lm(MODEL, torch.bfloat16, "cuda")
    layers = {int(r.group(1)): m for n, m in model.named_modules() if (r := re.search(r"language_model\.layers\.(\d+)$", n))}
    nL = len(layers)
    cur, sel = {}, {}

    def hook(L):
        def f(mod, inp):
            if L in sel:                                              # M3: row b of the batch ablates neuron sel[L][b]
                x = inp[0].clone()
                ix, val = sel[L]
                x[torch.arange(len(ix)), :, ix] = val.to(x.dtype)[:, None]
                return (x,)
            cur[L] = inp[0]
        return f
    hooks = [layers[L].mlp.down_proj.register_forward_pre_hook(hook(L)) for L in range(nL)]
    out = {"width": [layers[L].mlp.down_proj.in_features for L in range(nL)]}
    # read vectors for M2: gate row times the pre-feedforward norm weight
    out["read"] = {}
    for L in {l for ps in M2_PAIRS.values() for p in ps for l in p}:
        g = layers[L].pre_feedforward_layernorm(torch.ones(1, 1536, device="cuda", dtype=torch.bfloat16)).float()[0]
        out["read"][L] = (layers[L].mlp.gate_proj.weight.float() * g).cpu().numpy()

    # articles: every layer's neurons, mean over tokens
    texts = dbpedia(10, 6000)[0]["test"][0][:N_DOCS]
    acts = [[] for _ in range(nL)]
    with torch.no_grad():
        for s in range(0, len(texts), BATCH):
            enc = tok(texts[s:s + BATCH], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
            model(**enc, use_cache=False)
            m = enc["attention_mask"].unsqueeze(-1).float()
            for L in range(nL):
                acts[L].append(((cur[L].float() * m).sum(1) / m.sum(1)).cpu().numpy())
    out["acts"] = [np.concatenate(a) for a in acts]
    log("articles done")

    # M3: single-neuron mean ablations, batched (row b ablates one neuron), and size predictors on the same tokens
    meta = np.load(next(d / "tok_meta.npz" for d in CACHE_DIRS if (d / "tok_meta.npz").exists()), allow_pickle=True)
    enc_t = lambda ts: [tok(str(t), truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda") for t in ts]
    evals, rank_enc = enc_t(meta["eval_texts"][:M3_TEXTS]), enc_t(meta["eval_texts"][slice(*M3_RANK)])
    mean_a, ntok = {L: 0 for L in M3_LAYERS}, 0
    with torch.no_grad():
        for e in rank_enc:
            model(**e, use_cache=False)
            for L in M3_LAYERS:
                mean_a[L] = mean_a[L] + cur[L][0].float().sum(0)
            ntok += e["input_ids"].shape[1]
    mean_a = {L: v / ntok for L, v in mean_a.items()}
    rng = np.random.default_rng(0)
    neurons = {L: np.sort(rng.choice(out["width"][L], M3_N, replace=False)) for L in M3_LAYERS}
    Wd, gw, eps = {}, {}, {}
    for L in M3_LAYERS:
        gw[L] = layers[L].post_feedforward_layernorm(torch.ones(1, 1536, device="cuda", dtype=torch.bfloat16)).float()[0]
        Wd[L] = layers[L].mlp.down_proj.weight.float()                   # 1536 x width
        eps[L] = getattr(layers[L].post_feedforward_layernorm, "eps", 1e-6)
    pred = {L: {k: np.zeros(M3_N) for k in ("act", "write", "write_rel")} for L in M3_LAYERS}
    kl = {L: np.zeros(M3_N) for L in M3_LAYERS}
    npos, null = 0, 0.0
    with torch.no_grad():
        for t, e in enumerate(evals):
            o = model(**e, output_hidden_states=True, use_cache=False)
            sel.clear()
            batch = {k: v.expand(M3_B, -1) for k, v in e.items()}
            clean = torch.log_softmax(model(**batch, use_cache=False).logits[:, 1:].float(), -1)   # same batch shape
            pc = clean.exp()
            null += float((pc[0] * (clean[0] - clean[1])).sum())
            T = clean.shape[1]
            for L in M3_LAYERS:
                a = cur[L][0, 1:].float()                                     # tokens x width (skip <bos>)
                raw = a @ Wd[L].T
                rms = torch.sqrt((raw ** 2).mean(-1) + eps[L])                # post-norm divides the MLP output by this
                n = torch.as_tensor(neurons[L], device="cuda")
                size = a[:, n].abs() * (Wd[L][:, n] * gw[L][:, None]).norm(dim=0) / rms[:, None]
                hn = o.hidden_states[L][0, 1:].float().norm(dim=-1)           # residual entering decoder layer L
                pred[L]["act"] += a[:, n].abs().sum(0).cpu().numpy()
                pred[L]["write"] += size.sum(0).cpu().numpy()
                pred[L]["write_rel"] += (size / hn[:, None]).sum(0).cpu().numpy()
            npos += T
            for L in M3_LAYERS:
                for b in range(0, M3_N, M3_B):
                    ix = torch.as_tensor(neurons[L][b:b + M3_B], device="cuda")
                    sel.clear()
                    sel[L] = (ix, mean_a[L][ix])
                    lp = torch.log_softmax(model(**{k: v[:len(ix)] for k, v in batch.items()}, use_cache=False).logits[:, 1:].float(), -1)
                    kl[L][b:b + M3_B] += (pc[:len(ix)] * (clean[:len(ix)] - lp)).sum(-1).sum(1).cpu().numpy()
            sel.clear()
            if t % 10 == 0:
                log("M3 text", t)
    for h in hooks:
        h.remove()
    out["m3"] = {"neurons": {L: v.tolist() for L, v in neurons.items()}, "kl": {L: (v / npos).tolist() for L, v in kl.items()},
                 "pred": {L: {k: (v / npos).tolist() for k, v in d.items()} for L, d in pred.items()}, "n_pos": npos, "kl_null_identical_rows": null / npos}
    del model
    torch.cuda.empty_cache()
    return out


def m1(acts, width):
    nL = len(acts)
    rng = np.random.default_rng(0)
    rots = {w: [np.linalg.qr(rng.standard_normal((w, w)).astype(np.float32))[0] for _ in range(N_ROT)] for w in set(width)}
    log("rotations ready")
    res = {}
    for K in (8, 4):
        groups = {}
        for i in range(1, nL + 1):                                         # hidden i = residual entering decoder layer i
            Ztr, Zte = pca64(load_cache(f"{NAME}_L{i:02d}", "train")[:N_FIT], load_cache(f"{NAME}_L{i:02d}", "test")[:N_DOCS])
            groups[i] = consensus(Ztr, Zte, K)
        splits, olds = [], []

        def score(l, sub, y, **kw):
            A = acts[l][sub] - acts[l][sub].mean(0)
            return {"layer": l, "n": [int(y.sum()), int((~y).sum())], "neurons": max_auc(A, y),
                    "rotations": float(np.mean([max_auc(A @ Q, y) for Q in rots[width[l]]])),
                    "neurons_l_minus_2": max_auc(acts[l - 2][sub], y), "width_change": width[l] != width[l - 2], **kw}
        for l in range(2, nL):
            gin, gout = groups[l], groups[l + 1]
            pairs = set()
            for p in np.unique(gin):
                mem = np.flatnonzero(gin == p)
                kids, cnt = np.unique(gout[mem], return_counts=True)
                ok = cnt >= max(30, 0.2 * len(mem))
                if ok.sum() < 2:
                    continue
                c1, c2 = kids[np.argsort(-cnt)[:2]]
                pairs.add(frozenset((int(c1), int(c2))))
                sub = mem[np.isin(gout[mem], [c1, c2])]
                splits.append(score(l, sub, gout[sub] == c1, parent=int(p)))
            # control (added in run 2): output groups whose members mostly came from different input groups
            sizes = {int(c): int((gout == c).sum()) for c in np.unique(gout)}
            major = {c: np.bincount(gin[gout == c]).argmax() for c in sizes if sizes[c] >= 30}
            cand = sorted(((min(sizes[a], sizes[b]), a, b) for a in major for b in major
                           if a < b and major[a] != major[b] and frozenset((a, b)) not in pairs), reverse=True)[:2]
            for _, c1, c2 in cand:
                sub = np.flatnonzero(np.isin(gout, [c1, c2]))
                olds.append(score(l, sub, gout[sub] == c1))
            log("M1", K, "layer", l, "splits", len(splits), "old pairs", len(olds))
        f = lambda c: float(np.mean([c(s) for s in splits])) if splits else None
        same = [s for s in splits if not s["width_change"]]
        g = lambda ss, c: float(np.mean([c(x) for x in ss])) if ss else None
        res[K] = {"splits": splits, "n_splits": len(splits), "old_pairs": olds, "n_old": len(olds),
                  "control_old_frac_beat_rot": g(olds, lambda x: x["neurons"] > x["rotations"]),
                  "control_old_frac_beat_prev": g(olds, lambda x: x["neurons"] > x["neurons_l_minus_2"]),
                  "gain_vs_prev_splits": g(splits, lambda x: x["neurons"] - x["neurons_l_minus_2"]),
                  "gain_vs_prev_old": g(olds, lambda x: x["neurons"] - x["neurons_l_minus_2"]),
                  "M1a_frac_beat_rot": f(lambda s: s["neurons"] > s["rotations"]),
                  "M1b_frac_beat_prev": f(lambda s: s["neurons"] > s["neurons_l_minus_2"]),
                  "M1b_frac_beat_prev_same_width": float(np.mean([s["neurons"] > s["neurons_l_minus_2"] for s in same])) if same else None,
                  "splits_per_layer": np.bincount([s["layer"] for s in splits], minlength=nL).tolist()}
        log("M1", K, {k: v for k, v in res[K].items() if k not in ("splits", "old_pairs", "splits_per_layer")})
    return res


def m2(acts, read):
    rng = np.random.default_rng(0)
    std = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-8)
    res = {}
    for kind, pairs in M2_PAIRS.items():
        for l, m in pairs:
            Zl, Zm = std(acts[l]), std(acts[m])
            S = rng.choice(Zl.shape[1], 500, replace=False)
            C = Zl[:, S].T @ Zm / len(Zl)                                   # 500 x width_m
            j = np.abs(C).argmax(1)
            sgn = np.sign(C[np.arange(len(S)), j])
            Hl = unit(load_cache(f"{NAME}_L{l:02d}", "train")[:N_FIT]); Hl -= Hl.mean(0)
            Hm = unit(load_cache(f"{NAME}_L{m:02d}", "train")[:N_FIT]); Hm -= Hm.mean(0)
            U, _, Vt = np.linalg.svd(Hl.T @ Hm)
            Q = U @ Vt                                                      # h_m ~ h_l Q, so neuron j reads Q r_j on h_l
            Rl, Rm = read[l] - read[l].mean(0), read[m] - read[m].mean(0)   # run 2: shared component removed
            ri, rj = unit(Rl[S] * sgn[:, None]), unit(Rm[j])
            rand = unit(Rm[rng.choice(len(Rm), len(S))])
            res[f"{l}-{m}"] = {"kind": kind, "match_corr_median": float(np.median(np.abs(C[np.arange(len(S)), j]))),
                               "cos_raw_median": float(np.median(np.abs((ri * rj).sum(1)))),
                               "cos_mapped_median": float(np.median(np.abs((ri * unit(rj @ Q.T)).sum(1)))),
                               "cos_random_median": float(np.median(np.abs((ri * rand).sum(1)))),
                               "cos_random_mapped_median": float(np.median(np.abs((ri * unit(rand @ Q.T)).sum(1))))}
            log("M2", l, m, res[f"{l}-{m}"])
    return res


def m3(d):
    pooled = {k: [] for k in ("act", "write", "write_rel")}
    klp, out = [], {"per_layer": {}}
    for L in M3_LAYERS:
        kl = np.array(d["kl"][L])
        out["per_layer"][L] = {k: float(spearmanr(d["pred"][L][k], kl)[0]) for k in pooled}
        out["per_layer"][L]["kl_median"] = float(np.median(kl))
        for k in pooled:
            pooled[k] += d["pred"][L][k]
        klp += list(kl)
    out["pooled"] = {k: float(spearmanr(v, klp)[0]) for k, v in pooled.items()}
    out["M3_pass"] = out["pooled"]["write_rel"] >= 0.5 and out["pooled"]["write_rel"] > out["pooled"]["write"]
    log("M3", out)
    return out


def main():
    set_threads()
    g = gpu_pass()
    res = {"model": MODEL, "n_docs": N_DOCS, "width": g["width"], "m3_raw": g["m3"]}
    res["M3"] = m3(g["m3"])
    res["M2"] = m2(g["acts"], g["read"])
    (RESULTS / "macro_micro.json").write_text(json.dumps(res, indent=1, default=str))
    res["M1"] = m1(g["acts"], g["width"])
    (RESULTS / "macro_micro.json").write_text(json.dumps(res, indent=1, default=str))
    log("saved")


if __name__ == "__main__":
    main()
