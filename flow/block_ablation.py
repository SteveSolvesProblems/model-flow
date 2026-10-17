"""What drives Gemma's sharp coordinate turn at layers 13-15? (exploratory, not pre-registered)
Knock out one component at a time (mean ablation: its output replaced by its mean over 30 texts) and measure the
turn and the prediction against the clean run:
  attn_L / mlp_L   attention block / MLP output of layer L, for L in 12-15 (turn) and 8, 22 (controls)
  ple_13_14        per-layer-embedding projection output of layers 13 and 14
  scalar1_13_14    the learned per-layer scalars of layers 13 and 14 set to 1 (causal test of the scalar link)
  scalex<f>_13_14  those scalars multiplied by f (dose-response, after the 2026-10-04 audit: setting them to 1 is a
                   13-20x change far outside normal operation); run with --only scalex0.5_13_14 ... --tag dose
  --mean-offset    take the ablation means from texts [offset, offset + n_mean) instead of the first n_mean evaluated
Prediction stated before running: the turn (centered cosine between cached layers 13 and 15, clean ~0.006) rises
clearly only when a turn-layer component (13 or 14) is knocked out; control-layer knockouts leave it near clean.
Measured: centered cosine for adjacent cached layers and for 12->16 and 13->15 over 2,000 fixed token positions;
KL(clean || ablated) and top-1 agreement over all positions of n texts.
Writes results/block_ablation_<name>.json.

python block_ablation.py --name gemma4_e2b --model google/gemma-4-E2B
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
    ap.add_argument("--name", default="gemma4_e2b")
    ap.add_argument("--model", default="google/gemma-4-E2B")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-mean", type=int, default=30)
    ap.add_argument("--n-pos", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None, help="run only these conditions")
    ap.add_argument("--tag", default="", help="suffix for the output file")
    ap.add_argument("--mean-offset", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).eval()
    layers = {}
    for n, m in model.named_modules():
        r = re.search(r"language_model\.layers\.(\d+)$", n)
        if r:
            layers[int(r.group(1))] = m
    conds = {}
    for L in (12, 13, 14, 15, 8, 22):
        conds[f"attn_{L}"] = [("attn", L)]
        conds[f"mlp_{L}"] = [("mlp", L)]
    conds["ple_13_14"] = [("ple", 13), ("ple", 14)]
    conds["scalar1_13_14"] = [("scalar", 13), ("scalar", 14)]
    for f in (0.5, 2.0, 4.0):
        conds[f"scalex{f:g}_13_14"] = [(f"scale:{f}", 13), (f"scale:{f}", 14)]
    if a.only is not None:
        conds = {k: v for k, v in conds.items() if k in a.only}
        assert len(conds) == len(a.only), f"unknown condition in {a.only}"
    module = lambda kind, L: {"attn": layers[L].self_attn, "mlp": layers[L].mlp,
                              "ple": layers[L].per_layer_projection}[kind]

    meta = np.load(next(d / "tok_meta.npz" for d in CACHE_DIRS if (d / "tok_meta.npz").exists()), allow_pickle=True)
    encs = [tok(str(t), truncation=True, max_length=64, return_tensors="pt") for t in meta["eval_texts"][: a.n]]

    # mean outputs of every targeted module
    targets = sorted({(k, L) for c in conds.values() for k, L in c if k != "scalar" and not k.startswith("scale:")})
    sums, cnt = {t: 0 for t in targets}, 0
    def grab(t):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            sums[t] = sums[t] + o[0].float().sum(0)
        return f
    hooks = [module(*t).register_forward_hook(grab(t)) for t in targets]
    with torch.no_grad():
        for e in ([tok(str(t), truncation=True, max_length=64, return_tensors="pt") for t in
                   meta["eval_texts"][a.mean_offset: a.mean_offset + a.n_mean]] if targets else []):
            model(**e)
            cnt += e["input_ids"].shape[1]
    for h in hooks:
        h.remove()
    means = {t: s / max(cnt, 1) for t, s in sums.items()}

    active = set()
    def repl(t):
        def f(mod, inp, out):
            if t not in active:
                return None
            o = out[0] if isinstance(out, tuple) else out
            new = means[t].to(o.dtype).expand_as(o).clone()
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return f
    hooks = [module(*t).register_forward_hook(repl(t)) for t in targets]
    scal0 = {L: layers[L].layer_scalar.detach().clone() for L in (13, 14)}

    def set_cond(name):
        active.clear()
        for L, v in scal0.items():
            layers[L].layer_scalar.data.copy_(v)
        for k, L in conds.get(name, []):
            if k == "scalar":
                layers[L].layer_scalar.data.fill_(1.0)
            elif k.startswith("scale:"):
                layers[L].layer_scalar.data.copy_(scal0[L] * float(k[6:]))
            else:
                active.add((k, L))

    rng = np.random.default_rng(0)
    allpos = [(i, p) for i, e in enumerate(encs) for p in range(1, e["input_ids"].shape[1])]
    by_text = {}
    for k in sorted(rng.choice(len(allpos), min(a.n_pos, len(allpos)), replace=False).tolist()):
        i, p = allpos[k]
        by_text.setdefault(i, []).append(p)
    names = ["clean"] + list(conds)
    states = {n: [] for n in names}
    kl, t1 = {n: [] for n in conds}, {n: [] for n in conds}
    with torch.no_grad():
        for i, e in enumerate(encs):
            clean_lp = None
            for name in names:
                set_cond(name)
                out = model(**e, output_hidden_states=True)
                lp = torch.log_softmax(out.logits[0].float(), -1)
                if name == "clean":
                    clean_lp = lp
                else:
                    kl[name].append((clean_lp.exp() * (clean_lp - lp)).sum(-1).numpy())
                    t1[name].append((clean_lp.argmax(-1) == lp.argmax(-1)).numpy())
                if i in by_text:
                    states[name].append(torch.stack([hs[0, by_text[i]] for hs in out.hidden_states], 0).float().numpy())
            if i % 20 == 0:
                log(a.name, "text", i)
    set_cond("clean")
    res = {}
    for name in names:
        st = np.concatenate(states[name], 1)
        C = [unit(X - X.mean(0)) for X in st]
        cc = lambda x, y: float((C[x] * C[y]).sum(1).mean())
        r = {"adj_cos": [cc(k, k + 1) for k in range(len(C) - 1)], "turn_13_15": cc(13, 15), "turn_12_16": cc(12, 16)}
        if name != "clean":
            r.update({"parts": conds[name], "kl": float(np.concatenate(kl[name]).mean()),
                      "top1": float(np.concatenate(t1[name]).mean())})
        res[name] = r
        log(name, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)},
            "adj 12-15:", [round(x, 3) for x in r["adj_cos"][12:16]])
    for h in hooks:
        h.remove()
    (RESULTS / f"block_ablation_{a.name}{'_' + a.tag if a.tag else ''}.json").write_text(json.dumps({"name": a.name, "model": a.model,
                                                                        "n_texts": len(encs), "mean_offset": a.mean_offset, "results": res}))
    log("saved")


if __name__ == "__main__":
    main()
