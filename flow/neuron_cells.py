"""Do Gemma 4's MLP neurons draw the turn? (exploratory; plan and predictions in research/NOTE_neurons_vs_cells.md)

Neuron = input to down_proj (gelu(gate x) * up x) at the turn layers (E2B: 13, 14; E4B: 22, 23).
  A  best single axis (two-sided AUC) for each k-means cell / topic, neuron axes vs random rotations of the same
     activations; on mean-pooled articles, and with --tokens also on single tokens
  B  per-token concentration of the MLP write (after post_feedforward_layernorm) over neurons
  C  ablate the top-K turn neurons (by write energy, or by varying write energy), K random, or all, at the turn
     layers together; mean, zero or resampled (from other texts' tokens) values; turn and prediction vs clean
  D  what sits at each pole of the top turn neurons (and, with --tokens, of random neurons, token by token)
Writes results/neuron_cells[_<tag>].json.

python neuron_cells.py                                                          # E2B, first run
python neuron_cells.py --modes mean zero resample --tokens --tag e2b_v2
DIAG_LOW_VRAM=1 python neuron_cells.py --model google/gemma-4-E4B --cache gemma4_e4b --layers 22 23 \
    --modes mean zero resample --tag e4b
"""
import argparse
import json
import re

import numpy as np
import torch

from common import CACHE_DIRS, RESULTS, load_cache, set_threads
from extract_hf import dbpedia, load_causal_lm
from lens import frame, log
from partition import assign, kmeans

N_ART, MAX_LEN, BATCH = 6000, 64, 16
N_EVAL, RANK_TEXTS, N_POS, N_ROT = 100, (100, 130), 2000, 5
KS = (16, 64, 256, 1024)
N_TOK_ART, TOK_PER_ART = 1500, 14          # --tokens: about 20,000 tokens from the first 1,500 articles
SUBSET = ("top64", "top1024", "vtop64", "vtop1024", "rand1024", "all")   # conditions repeated for zero / resample


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def ranks(Z):
    return np.argsort(np.argsort(Z, 0, kind="stable"), 0, kind="stable").astype(np.float64) + 1


def best_auc(R, groups):
    """For each group g: max over columns of the two-sided AUC of g vs rest (R = column ranks)."""
    out = []
    for g in np.unique(groups):
        m = groups == g
        n1, n0 = m.sum(), (~m).sum()
        auc = (R[m].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
        out.append(float(np.maximum(auc, 1 - auc).max()))
    return out


def test_a(A, H_out, H_in, topics):
    rng = np.random.default_rng(0)
    Ac = A - A.mean(0)
    targets = {"topics": topics}
    for side, H in (("out", H_out), ("in", H_in)):
        F = frame(H, H)[0]
        for K in (8, 64):
            targets[f"cells{K}_{side}"] = assign(F, kmeans(F, K, seed=0))[0]
    R = ranks(Ac)
    res = {n: {"neurons_per_group": best_auc(R, g), "rotations": []} for n, g in targets.items()}
    for _ in range(N_ROT):
        R = ranks(Ac @ np.linalg.qr(rng.standard_normal((A.shape[1], A.shape[1])))[0])
        for n, g in targets.items():
            res[n]["rotations"].append(float(np.mean(best_auc(R, g))))
    for n, r in res.items():
        r["neurons"] = float(np.mean(r["neurons_per_group"]))
        log("A", n, f"neurons {r['neurons']:.3f}  rotations {np.mean(r['rotations']):.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B")
    ap.add_argument("--cache", default="gemma4_e2b", help="cached residuals for the sanity check")
    ap.add_argument("--layers", type=int, nargs=2, default=[13, 14])
    ap.add_argument("--modes", nargs="+", default=["mean"], choices=["mean", "zero", "resample"])
    ap.add_argument("--tokens", action="store_true", help="also run A and D on single tokens")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    LAYERS = tuple(a.layers)
    L0, L1 = LAYERS
    set_threads()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_causal_lm(a.model, torch.bfloat16, "cuda")
    layers = {int(r.group(1)): m for n, m in model.named_modules() if (r := re.search(r"language_model\.layers\.(\d+)$", n))}
    width = {L: layers[L].mlp.down_proj.in_features for L in LAYERS}
    hidden = layers[L0].mlp.down_proj.out_features
    cur = {}

    def grab(L):
        def f(mod, inp):
            cur[L] = inp[0]
        return f
    hooks = [layers[L].mlp.down_proj.register_forward_pre_hook(grab(L)) for L in LAYERS]

    # ── articles: mean-pooled neurons and residuals (and, with --tokens, a sample of single tokens) ──
    data, maps = dbpedia(10, N_ART)
    texts, labels = data["test"]
    acts = {L: [] for L in LAYERS}
    hs = {i: [] for i in range(L0 - 1, L1 + 3)}
    tk = {"a": {L: [] for L in LAYERS}, "h_in": {L: [] for L in LAYERS}, "h_out": {L: [] for L in LAYERS}, "where": []}
    trng = np.random.default_rng(0)
    with torch.no_grad():
        for s in range(0, len(texts), BATCH):
            enc = tok(texts[s:s + BATCH], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
            out = model(**enc, output_hidden_states=True, use_cache=False)
            m = enc["attention_mask"].unsqueeze(-1).float()
            pool = lambda h: ((h.float() * m).sum(1) / m.sum(1)).cpu().numpy()
            for L in LAYERS:
                acts[L].append(pool(cur[L]))
            for i in hs:
                hs[i].append(pool(out.hidden_states[i]))
            if a.tokens and s < N_TOK_ART:
                for b in range(enc["input_ids"].shape[0]):
                    n = int(enc["attention_mask"][b].sum())
                    ps = np.sort(trng.choice(np.arange(1, n), min(TOK_PER_ART, n - 1), replace=False))
                    tk["where"] += [(s + b, int(p)) for p in ps]
                    for L in LAYERS:
                        tk["a"][L].append(cur[L][b, ps].float().cpu().numpy())
                        tk["h_in"][L].append(out.hidden_states[L][b, ps].float().cpu().numpy())
                        tk["h_out"][L].append(out.hidden_states[L + 1][b, ps].float().cpu().numpy())
            if s % (BATCH * 80) == 0:
                log("articles", s)
    acts = {L: np.concatenate(v) for L, v in acts.items()}
    hs = {i: np.concatenate(v) for i, v in hs.items()}
    res = {"model": a.model, "layers": LAYERS, "width": width, "n_articles": len(texts), "max_len": MAX_LEN,
           "post_hoc": ["vtop*: variance-ranked ablations, added after the first run",
                        "zero / resample modes, --tokens and E4B: added after the first run (approved follow-up)"],
           "sanity": {}}
    for i in (L0, L1, L1 + 1):
        try:
            C = load_cache(f"{a.cache}_L{i:02d}", "test")[:len(texts)]
            res["sanity"][f"cache_cos_L{i}"] = float((unit(C) * unit(hs[i])).sum(1).mean())
        except Exception as e:
            res["sanity"][f"cache_cos_L{i}"] = f"unavailable: {type(e).__name__}"
    log("sanity", res["sanity"])

    # ── B/C setup: token texts, per-neuron write directions, ranking on separate texts ──
    meta = np.load(next(d / "tok_meta.npz" for d in CACHE_DIRS if (d / "tok_meta.npz").exists()), allow_pickle=True)
    enc_t = lambda ts: [tok(str(t), truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda") for t in ts]
    evals, rank_enc = enc_t(meta["eval_texts"][:N_EVAL]), enc_t(meta["eval_texts"][slice(*RANK_TEXTS)])
    Wd, col2 = {}, {}
    for L in LAYERS:
        dev = layers[L].mlp.down_proj.weight.device
        gw = layers[L].post_feedforward_layernorm(torch.ones(1, hidden, device=dev, dtype=torch.bfloat16)).float()[0]
        Wd[L] = (layers[L].mlp.down_proj.weight.float().T * gw).cuda()   # neuron i writes a_i * Wd[i] (up to the rms)
        col2[L] = (Wd[L] ** 2).sum(1)
    energy, mean_a, sq_a = ({L: 0 for L in LAYERS} for _ in range(3))
    pool_a = {L: [] for L in LAYERS}
    ntok = 0
    with torch.no_grad():
        for e in rank_enc:
            model(**e, use_cache=False)
            for L in LAYERS:
                x = cur[L][0].float().cuda()
                energy[L] = energy[L] + ((x ** 2) * col2[L]).sum(0)
                mean_a[L] = mean_a[L] + x.sum(0)
                sq_a[L] = sq_a[L] + (x ** 2).sum(0)
                pool_a[L].append(x[1:])                                 # resample pool: non-<bos> tokens
            ntok += e["input_ids"].shape[1]
    mean_a = {L: v / ntok for L, v in mean_a.items()}
    pool_a = {L: torch.cat(v) for L, v in pool_a.items()}
    order = {L: torch.argsort(energy[L], descending=True) for L in LAYERS}
    # variance-ranked (post hoc): var(a_i) |w_i|^2; the energy ranking picks always-on neurons whose mean ablation
    # keeps most of their write
    vorder = {L: torch.argsort((sq_a[L] / ntok - mean_a[L] ** 2) * col2[L], descending=True) for L in LAYERS}

    # ── B: concentration on the eval tokens ──────────────────────────────────
    B = {}
    with torch.no_grad():
        for L in LAYERS:
            k50, k90, top64 = [], [], []
            for e in evals:
                model(**e, use_cache=False)
                for x in cur[L][0].float().cuda()[1:]:                      # one token at a time; skip <bos>
                    C = x[:, None] * Wd[L]                                  # neurons x hidden
                    w = C.sum(0)
                    tot = (w ** 2).sum()
                    idx = torch.argsort(x.abs() * col2[L].sqrt(), descending=True)
                    left = ((w - torch.cumsum(C[idx], 0)) ** 2).sum(1) / tot  # unexplained energy after the top k
                    k50.append(int(torch.nonzero(left <= 0.5)[0]) + 1)
                    k90.append(int(torch.nonzero(left <= 0.1)[0]) + 1)
                    top64.append(float(1 - ((w - C[order[L][:64]].sum(0)) ** 2).sum() / tot))
            k50, k90, top64 = map(np.array, (k50, k90, top64))
            B[L] = {"k50_median": float(np.median(k50)), "k90_median": float(np.median(k90)),
                    "global_top64_share_median": float(np.median(top64)), "n_tokens": int(len(k50)),
                    "energy_share_top": {k: float(energy[L][order[L][:k]].sum() / energy[L].sum()) for k in KS}}
            log("B", L, B[L])
    res["B"] = B

    # ── C: causal ablations at the turn layers together ───────────────────────
    sel, mode, gen = {}, ["mean"], torch.Generator(device="cuda")

    def patch(L):
        def f(mod, inp):
            if sel.get(L) is None:
                return None
            x = inp[0].clone()
            ix = sel[L].to(x.device)
            if mode[0] == "mean":
                x[..., ix] = mean_a[L][ix].to(x.device, x.dtype)
            elif mode[0] == "zero":
                x[..., ix] = 0
            else:                                                       # resample: other texts' tokens, row by row
                rows = torch.randint(0, len(pool_a[L]), x.shape[:-1], generator=gen, device="cuda")
                x[..., ix] = pool_a[L][rows][..., sel[L].cuda()].to(x.device, x.dtype)
            return (x,)
        return f
    for h in hooks:
        h.remove()
    hooks = [layers[L].mlp.down_proj.register_forward_pre_hook(patch(L)) for L in LAYERS]
    g = torch.Generator().manual_seed(0)
    base = {**{f"top{k}": {L: order[L][:k] for L in LAYERS} for k in KS},
            **{f"rand{k}": {L: torch.randperm(width[L], generator=g)[:k] for L in LAYERS} for k in KS},
            **{f"vtop{k}": {L: vorder[L][:k] for L in LAYERS} for k in KS},
            "all": {L: torch.arange(width[L]) for L in LAYERS}}
    conds = {"clean": ("mean", None)}
    for md in a.modes:
        for c, s in base.items():
            if md == "mean" or c in SUBSET:
                conds[c if md == "mean" else f"{md}:{c}"] = (md, s)
    rng = np.random.default_rng(0)
    allpos = [(i, p) for i, e in enumerate(evals) for p in range(1, e["input_ids"].shape[1])]
    by_text = {}
    for k in sorted(rng.choice(len(allpos), min(N_POS, len(allpos)), replace=False).tolist()):
        i, p = allpos[k]
        by_text.setdefault(i, []).append(p)
    states, kl, t1 = {c: [] for c in conds}, {c: [] for c in conds}, {c: [] for c in conds}
    with torch.no_grad():
        for i, e in enumerate(evals):
            clean_lp = None
            for c, (md, s) in conds.items():
                sel.clear()
                sel.update(s or {})
                mode[0] = md
                gen.manual_seed(i)
                out = model(**e, output_hidden_states=True, use_cache=False)
                lp = torch.log_softmax(out.logits[0].float(), -1)
                if clean_lp is None:
                    clean_lp = lp
                kl[c].append((clean_lp.exp() * (clean_lp - lp)).sum(-1).cpu().numpy())
                t1[c].append((clean_lp.argmax(-1) == lp.argmax(-1)).cpu().numpy())
                if i in by_text:
                    states[c].append(torch.stack([h[0, by_text[i]] for h in out.hidden_states], 0).float().cpu().numpy())
            if i % 25 == 0:
                log("C text", i)
    sel.clear()
    Cres = {}
    for c in conds:
        st = np.concatenate(states[c], 1)
        U = [unit(X - X.mean(0)) for X in st]
        cc = lambda x, y: float((U[x] * U[y]).sum(1).mean())
        Cres[c] = {"turn": cc(L0, L1 + 1), "turn_wide": cc(L0 - 1, L1 + 2), "adj_cos": [cc(k, k + 1) for k in range(len(U) - 1)],
                   "kl": float(np.concatenate(kl[c]).mean()), "top1": float(np.concatenate(t1[c]).mean())}
        log("C", c, {k: round(v, 4) for k, v in Cres[c].items() if isinstance(v, float)},
            f"adj {L0 - 1}-{L1 + 3}:", [round(x, 3) for x in Cres[c]["adj_cos"][L0 - 1:L1 + 3]])
    res["C"] = Cres
    for h in hooks:
        h.remove()

    # ── D: poles of the top turn neurons (articles; tokens with --tokens) ──────
    l1 = {v: k for k, v in maps["l1"].items()}
    purity = lambda ix: float(np.bincount(labels["l1"][ix[:50]]).max() / 50)
    topic = lambda ix: l1[int(np.bincount(labels["l1"][ix[:50]]).argmax())]
    res["topic_base_rate"] = {l1[int(k)]: float(c / len(labels["l1"])) for k, c in zip(*np.unique(labels["l1"], return_counts=True))}
    D = {}
    for L in LAYERS:
        D[L] = []
        for n in order[L][:8].tolist():
            v = acts[L][:, n]
            hi, lo = np.argsort(-v), np.argsort(v)
            D[L].append({"neuron": n, "energy_share": float(energy[L][n] / energy[L].sum()),
                         "pos": {"topic": topic(hi), "purity": purity(hi), "texts": [texts[j][:90] for j in hi[:5]]},
                         "neg": {"topic": topic(lo), "purity": purity(lo), "texts": [texts[j][:90] for j in lo[:5]]},
                         "frac_negative": float((v < 0).mean())})
    res["D"] = D

    if a.tokens:
        where = tk["where"]
        ids = {}

        def context(j, p):
            if j not in ids:
                ids[j] = tok(texts[j], truncation=True, max_length=MAX_LEN)["input_ids"]
            t = ids[j]
            return (tok.decode(t[max(1, p - 8):p]) + " [[" + tok.decode(t[p:p + 1]) + "]] " + tok.decode(t[p + 1:p + 4])).replace("\n", " ")
        rng_n = torch.Generator().manual_seed(1)
        res["D_tokens"], res["A_tokens"] = {}, {}
        for L in LAYERS:
            At = np.concatenate(tk["a"][L])
            tlab = labels["l1"][np.array([j for j, _ in where])]
            picks = [("turn_var", n) for n in vorder[L][:8].tolist()] + \
                    [("random", n) for n in torch.randperm(width[L], generator=rng_n)[:8].tolist()]
            res["D_tokens"][L] = []
            for kind, n in picks:
                v = At[:, n]
                hi, lo = np.argsort(-v), np.argsort(v)
                res["D_tokens"][L].append({"kind": kind, "neuron": n, "frac_negative": float((v < 0).mean()),
                                           "pos": [context(*where[k]) for k in hi[:6]],
                                           "neg": [context(*where[k]) for k in lo[:6]],
                                           "pos_val": float(v[hi[0]]), "neg_val": float(v[lo[0]])})
            res["A_tokens"][L] = test_a(At, np.concatenate(tk["h_out"][L]), np.concatenate(tk["h_in"][L]), tlab)
            res["A_tokens"][L]["n_tokens"] = int(len(At))
    del model
    torch.cuda.empty_cache()

    # ── A: neurons vs rotations on articles (CPU) ─────────────────────────────
    res["A"] = {L: test_a(acts[L], hs[L + 1], hs[L], labels["l1"]) for L in LAYERS}
    (RESULTS / f"neuron_cells{'_' + a.tag if a.tag else ''}.json").write_text(json.dumps(res, indent=1, default=str))
    log("saved")


if __name__ == "__main__":
    main()
