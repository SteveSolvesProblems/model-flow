"""Head knockouts (exploratory, not pre-registered; hypotheses in research/NOTE_geometry.md, "Attention heads"):
  T1 parked   mean-ablate all late-stretch heads with first-token attention > 0.5 vs the same number of random heads
              from the same layers. Prediction: parked heads change geometry and prediction less than random heads.
  T2 (Gemma)  mean-ablate the top induction head of layer 14, of 24, of 4, and the lowest-induction head of 14.
              Prediction: the layer-14 induction head weakens the 13-15 coordinate turn more than the others.
Mean ablation: a head's slice of the attention output (input to o_proj) is replaced by its mean over 50 texts.
Measured per condition, against the clean run on the same inputs:
  kl, top1    KL(clean || ablated) of next-token distributions and top-1 agreement, all positions of n texts
  layer_cos   per cached layer: mean cosine between clean and ablated states (2,000 fixed token positions)
  turn        centered cosine between cached layers a and b over the 2,000 positions (Gemma: 13->15 and 14->15)
  adj_cos     centered cosine between adjacent cached layers (the geometry probe's turn profile)
  ind_loss    next-token loss on the second copy of 50 random 32-token sequences repeated twice
Writes results/head_ablation_<name>.json.

python head_ablation.py --name gemma4_e2b --model google/gemma-4-E2B --late 21 33
"""
import argparse
import json
import re

import numpy as np
import torch

from common import CACHE_DIRS, RESULTS
from lens import log


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--late", type=int, nargs=2, required=True, help="first and last layer of the quiet late stretch")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-mean", type=int, default=50)
    ap.add_argument("--n-pos", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).eval()
    heads = json.loads((RESULTS / f"attention_heads_{a.name}.json").read_text())
    H = heads["n_heads"]
    oproj = {}
    for n, m in model.named_modules():
        r = re.search(r"layers\.(\d+)\.self_attn\.o_proj$", n)
        if r and "vision" not in n and "audio" not in n:
            oproj[int(r.group(1))] = m
    L = len(oproj)
    dh = {l: m.in_features // H for l, m in oproj.items()}

    # conditions: name -> list of (layer, head)
    rng = np.random.default_rng(0)
    sink = np.array([lay["sink"] for lay in heads["layers"]])
    ind = np.array([lay["induction"] for lay in heads["layers"]])
    lo, hi = a.late
    parked = [(l, h) for l in range(lo, hi + 1) for h in range(H) if sink[l, h] > 0.5]
    pool = [(l, h) for l in range(lo, hi + 1) for h in range(H)]
    rand = [pool[i] for i in rng.choice(len(pool), len(parked), replace=False)]
    conds = {"parked": parked, "random_same_layers": rand}
    if "gemma" in a.model.lower():
        for l in (14, 24, 4):
            conds[f"induction_L{l}"] = [(l, int(ind[l].argmax()))]
        conds["weakest_L14"] = [(14, int(ind[14].argmin()))]
    log({k: (len(v), v[:4]) for k, v in conds.items()})

    meta = np.load(next(d / "tok_meta.npz" for d in CACHE_DIRS if (d / "tok_meta.npz").exists()), allow_pickle=True)
    texts = [str(t) for t in meta["eval_texts"][: a.n]]
    encs = [tok(t, truncation=True, max_length=64, return_tensors="pt") for t in texts]

    # per-head mean output (input to o_proj), over n_mean texts
    sums, cnt, hooks = {l: 0 for l in oproj}, 0, []
    def grab(l):
        def f(mod, inp):
            sums[l] = sums[l] + inp[0][0].float().sum(0)
        return f
    hooks = [m.register_forward_pre_hook(grab(l)) for l, m in oproj.items()]
    with torch.no_grad():
        for e in encs[: a.n_mean]:
            model(**e)
            cnt += e["input_ids"].shape[1]
    for hk in hooks:
        hk.remove()
    means = {l: (s / cnt) for l, s in sums.items()}

    active = {}
    def ablate(l):
        def f(mod, inp):
            x = inp[0].clone()
            for h in active.get(l, []):
                x[..., h * dh[l]:(h + 1) * dh[l]] = means[l][h * dh[l]:(h + 1) * dh[l]].to(x.dtype)
            return (x,) + tuple(inp[1:])
        return f
    hooks = [m.register_forward_pre_hook(ablate(l)) for l, m in oproj.items()]

    # fixed token positions for geometry: (text, position >= 1)
    allpos = [(i, p) for i, e in enumerate(encs) for p in range(1, e["input_ids"].shape[1])]
    sel = sorted(rng.choice(len(allpos), min(a.n_pos, len(allpos)), replace=False).tolist())
    by_text = {}
    for k in sel:
        i, p = allpos[k]
        by_text.setdefault(i, []).append(p)
    seqs = []
    pool_ids = np.unique(np.concatenate([e["input_ids"][0].numpy() for e in encs[:100]]))
    pool_ids = pool_ids[~np.isin(pool_ids, tok.all_special_ids)]
    bos = [tok.bos_token_id] if tok("x")["input_ids"][0] == tok.bos_token_id else []
    for r in range(50):
        s = rng.choice(pool_ids, 32, replace=False)
        seqs.append(torch.tensor([bos + list(s) + list(s)]))

    names = ["clean"] + list(conds)
    def set_active(name):
        active.clear()
        for l, h in conds.get(name, []):
            active.setdefault(l, []).append(h)

    states = {n: [] for n in names}
    kl = {n: [] for n in conds}
    t1 = {n: [] for n in conds}
    with torch.no_grad():
        for i, e in enumerate(encs):
            clean_lp = None
            for name in names:
                set_active(name)
                out = model(**e, output_hidden_states=True)
                lp = torch.log_softmax(out.logits[0].float(), -1)
                if name == "clean":
                    clean_lp = lp
                else:
                    kl[name].append((clean_lp.exp() * (clean_lp - lp)).sum(-1).numpy())
                    t1[name].append((clean_lp.argmax(-1) == lp.argmax(-1)).numpy())
                if i in by_text:
                    states[name].append(torch.stack([hs[0, by_text[i]] for hs in out.hidden_states], 0).float().numpy())
            if i % 50 == 0:
                log(a.name, "text", i)
        ind_loss = {}
        for name in names:
            set_active(name)
            il = []
            for sq in seqs:
                lg = torch.log_softmax(model(input_ids=sq).logits[0].float(), -1)
                o = len(bos)
                q = torch.arange(o + 32, o + 63)
                il.append(-lg[q, sq[0, q + 1]].mean().item())
            ind_loss[name] = float(np.mean(il))
    cen = lambda S: [unit(X - X.mean(0)) for X in S]
    adj = lambda C: [float((C[k] * C[k + 1]).sum(1).mean()) for k in range(len(C) - 1)]
    clean_st = np.concatenate(states["clean"], 1)
    Lc = clean_st.shape[0]
    res = {}
    for name in names:
        st = np.concatenate(states[name], 1)
        C = cen(st)
        r = {"ind_loss": ind_loss[name], "adj_cos": adj(C)}
        if name != "clean":
            r.update({"heads": conds[name], "kl": float(np.concatenate(kl[name]).mean()),
                      "top1": float(np.concatenate(t1[name]).mean()),
                      "layer_cos": [float((unit(st[k]) * unit(clean_st[k])).sum(1).mean()) for k in range(Lc)]})
        if "gemma" in a.model.lower():
            r["turn_13_15"] = float((C[13] * C[15]).sum(1).mean())
            r["turn_14_15"] = float((C[14] * C[15]).sum(1).mean())
        res[name] = r
        log(name, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)},
            "last-layer cos", round(r["layer_cos"][-1], 4) if "layer_cos" in r else "")
    for hk in hooks:
        hk.remove()
    (RESULTS / f"head_ablation_{a.name}.json").write_text(json.dumps({"name": a.name, "model": a.model, "late": a.late,
                                                                       "n_texts": len(encs), "results": res}))
    log("saved")


if __name__ == "__main__":
    main()
