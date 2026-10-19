"""Do the layers after Gemma 4 E2B's turn read its new axes, and is that special to the turn? (exploratory, reviewer
follow-ups; plan in research/NOTE_neurons_vs_cells.md, "U" and "U2"). Writes results/turn_undo[_L<l>].json.

At the input of decoder layer L (hidden L), every token but the first is changed in its centered coordinates and the
layer mean added back; each token's centered norm is kept, so only directions move:
  undo@t    mix (1 - t) c + t c Q, renormalized, with Q the orthogonal Procrustes map from hidden L to hidden L - 2 (for
            L = 15, back to the pre-turn axes), fit on the tokens of tok_meta eval texts 300-599; t = 0.25 ... 1
  mlp       the same with a small nonlinear map (one hidden layer) fit to the same pairs, as a check on fit quality
  random@t  the same mix with a random orthogonal rotation (3 seeds at t = 1, 1 seed below)
  identity  the hook with no change (must give KL 0)
Measured on all positions of eval texts 0-99: KL(clean || condition), top-1 agreement, and the displacement, the mean
cosine between the changed and the original centered state (1 = unchanged). Fit quality: mean cosine between the
mapped hidden L and hidden L - 2 on held-out tokens.

python turn_undo.py [--layer 15] [--random-only] [--model google/gemma-4-E4B --tag e4b]
DIAG_ATTN=eager python turn_undo.py --random-only --clean-kv --dtype float16 --tag kv1_fp16 --layer 13   # P100 (no bf16)
(the first run, 2026-10-18, was layer 15 with undo@1 and random@1 only; --random-only and --model added for the
profile across layers and the E4B check; --clean-kv and --dtype added 2026-10-19 for KV1: with --clean-kv every
random rotation is also run with each layer's own key and value projections, at or after the site, replaced by their
output in the unrotated run, so only queries, MLPs and the stream see the rotation)
"""
import argparse
import json
import re

import numpy as np
import torch

from common import CACHE_DIRS, RESULTS, set_threads
from extract_hf import load_causal_lm
from lens import log

GAP, MAX_LEN, FIT, N_EVAL, N_RAND = 2, 64, (300, 600), 100, 3
TS = (0.25, 0.5, 0.75, 1.0)


def fit_mlp(X, Y, Xe, Ye, steps=600, hidden=3072, seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, Y.shape[1])).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    unit = lambda Z: Z / Z.norm(dim=-1, keepdim=True)
    Xu, Yu = unit(X), unit(Y)
    with torch.enable_grad():
        for _ in range(steps):
            idx = torch.randint(0, len(Xu), (4096,), device="cuda")
            loss = -(unit(net(Xu[idx])) * Yu[idx]).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return net, float((unit(net(unit(Xe))) * unit(Ye)).sum(-1).mean())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--random-only", action="store_true", help="identity and 3 random rotations at t = 1, no fits")
    ap.add_argument("--model", default="google/gemma-4-E2B")
    ap.add_argument("--tag", default="")
    ap.add_argument("--clean-kv", action="store_true", help="with --random-only: add rotations with clean keys and values")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--keep-massive", action="store_true",
                    help="with --random-only: hold coordinates with mean |x| > 20x median fixed, rotate only the rest")
    a = ap.parse_args()
    L_IN, MODEL = a.layer, a.model
    L_REF = L_IN - GAP
    set_threads()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = load_causal_lm(MODEL, getattr(torch, a.dtype), "cuda")
    layers = {int(r.group(1)): m for n, m in model.named_modules() if (r := re.search(r"language_model\.layers\.(\d+)$", n))}
    st = {"f": None, "disp": [0.0, 0]}

    def pre(mod, args, kwargs):
        if st["f"] is None:
            return None
        h = args[0] if args else kwargs["hidden_states"]
        x = h.float()
        c = x[:, 1:] - mu
        c2 = st["f"](c)
        c2 = c2 * (c.norm(dim=-1, keepdim=True) / c2.norm(dim=-1, keepdim=True))
        st["disp"][0] += float(((c2 * c).sum(-1) / (c.norm(dim=-1) ** 2)).sum())
        st["disp"][1] += c.shape[1]
        x[:, 1:] = c2 + mu
        h2 = x.to(h.dtype)
        if args:
            return (h2,) + tuple(args[1:]), kwargs
        return args, {**kwargs, "hidden_states": h2}
    layers[L_IN].register_forward_pre_hook(pre, with_kwargs=True)

    kv = {"mode": None, "store": {}, "i": 0}                         # mode: None, "record" or "replay"
    projs = [(L, nm, getattr(layers[L].self_attn, nm)) for L in sorted(layers) if L >= L_IN
             and not layers[L].self_attn.is_kv_shared_layer for nm in ("k_proj", "v_proj")
             if getattr(layers[L].self_attn, nm, None) is not None]

    def kv_hook(key):
        def f(mod, inp, out):
            if kv["mode"] == "record":
                kv["store"][(kv["i"],) + key] = out.detach()
            elif kv["mode"] == "replay":
                return kv["store"][(kv["i"],) + key]
        return f
    if a.clean_kv:
        for L, nm, mod in projs:
            mod.register_forward_hook(kv_hook((L, nm)))

    meta = np.load(next(d / "tok_meta.npz" for d in CACHE_DIRS if (d / "tok_meta.npz").exists()), allow_pickle=True)
    enc_t = lambda ts: [tok(str(t), truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda") for t in ts]
    fit, evals = enc_t(meta["eval_texts"][slice(*FIT)]), enc_t(meta["eval_texts"][:N_EVAL])

    def states(encs):
        A, B = [], []
        for e in encs:
            hs = model(**e, output_hidden_states=True, use_cache=False).hidden_states
            A.append(hs[L_IN][0, 1:].float())
            B.append(hs[L_REF][0, 1:].float())
        return torch.cat(A), torch.cat(B)
    A, B = states(fit[:40] if a.random_only else fit)
    mu, muB = A.mean(0), B.mean(0)
    U, _, Vt = torch.linalg.svd((A - mu).T @ (B - muB))
    Q = U @ Vt
    unit = lambda X: X / X.norm(dim=-1, keepdim=True)
    Ae, Be = states(evals)
    net, mlp_cos = (None, None) if a.random_only else fit_mlp(A - mu, B - muB, Ae - mu, Be - muB)
    fitq = {"n_fit_tokens": int(len(A)), "d": int(A.shape[1]),
            "cos_mapped_eval": float((unit((Ae - mu) @ Q) * unit(Be - muB)).sum(-1).mean()),
            "cos_mlp_eval": mlp_cos,
            "cos_unmapped_eval": float((unit(Ae - mu) * unit(Be - muB)).sum(-1).mean())}
    log("layer", L_IN, "fit", fitq)

    g = torch.Generator(device="cuda")
    rots = []
    m = A.abs().mean(0)
    massive = torch.nonzero(m > 20 * m.median()).flatten() if a.keep_massive else torch.zeros(0, dtype=torch.long, device="cuda")
    rest = torch.tensor(sorted(set(range(A.shape[1])) - set(massive.tolist())), device="cuda")
    fitq["massive_dims_held"] = massive.tolist()
    for s in range(N_RAND):
        g.manual_seed(s)
        R = torch.linalg.qr(torch.randn(len(rest), len(rest), generator=g, device="cuda"))[0]
        if a.keep_massive:                                             # identity on the massive coordinates
            F = torch.eye(A.shape[1], device="cuda")
            F[rest[:, None], rest[None, :]] = R
            R = F
        rots.append(R)
    mix = lambda M, t: (lambda c: (1 - t) * c + t * M(c))
    conds = {"identity": lambda c: c} if a.random_only else {"identity": lambda c: c, "mlp": lambda c: net(c)}
    for t in (() if a.random_only else TS):
        conds[f"undo@{t:g}"] = mix(lambda c: c @ Q, t)
        conds[f"random0@{t:g}"] = mix(lambda c, R=rots[0]: c @ R, t)
    for s in range(0 if a.random_only else 1, N_RAND):
        conds[f"random{s}@1"] = mix(lambda c, R=rots[s]: c @ R, 1.0)

    if a.clean_kv:
        conds["identity_cleankv"] = conds["identity"]                # replay check: must give KL 0
        for s in range(N_RAND):
            conds[f"random{s}_cleankv@1"] = conds[f"random{s}@1"]
    kv["mode"] = "record" if a.clean_kv else None
    clean = []
    for i, e in enumerate(evals):
        kv["i"] = i
        clean.append(torch.log_softmax(model(**e, use_cache=False).logits[0, 1:].float(), -1))
    res = {"model": MODEL, "dtype": a.dtype, "layer_input": L_IN, "reference_layer": L_REF, "fit": fitq,
           "clean_kv_projections": [f"{L}.{nm}" for L, nm, _ in projs] if a.clean_kv else None, "conds": {}}
    for c, f in conds.items():
        st["f"], st["disp"] = f, [0.0, 0]
        kv["mode"] = "replay" if "cleankv" in c else None
        kl = top1 = n = 0.0
        for i, (e, ref) in enumerate(zip(evals, clean)):
            kv["i"] = i
            lp = torch.log_softmax(model(**e, use_cache=False).logits[0, 1:].float(), -1)
            kl += float((ref.exp() * (ref - lp)).sum())
            top1 += float((ref.argmax(-1) == lp.argmax(-1)).sum())
            n += ref.shape[0]
        res["conds"][c] = {"kl": kl / n, "top1": top1 / n, "cos_to_original": st["disp"][0] / max(st["disp"][1], 1)}
        log(c, {k: round(v, 4) for k, v in res["conds"][c].items()})
    st["f"] = None
    kv["mode"] = None
    r = [res["conds"][k]["kl"] for k in res["conds"] if k.startswith("random") and k.endswith("@1") and "cleankv" not in k]
    res["random_kl_at_1"] = {"mean": float(np.mean(r)), "min": float(min(r)), "max": float(max(r))}
    if a.clean_kv:
        r = [res["conds"][k]["kl"] for k in res["conds"] if k.startswith("random") and "cleankv" in k]
        res["random_cleankv_kl_at_1"] = {"mean": float(np.mean(r)), "min": float(min(r)), "max": float(max(r))}
    if not a.random_only:
        res["undo_over_random_kl"] = res["conds"]["undo@1"]["kl"] / float(np.mean(r))
    (RESULTS / f"turn_undo{'_' + a.tag if a.tag else ''}{'_rand' if a.random_only else ''}{'_keepmass' if a.keep_massive else ''}{'_cleankv' if a.clean_kv else ''}_L{L_IN}.json").write_text(json.dumps(res, indent=1))
    log("saved", res["random_kl_at_1"])


if __name__ == "__main__":
    main()
