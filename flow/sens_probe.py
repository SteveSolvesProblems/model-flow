"""research/PREREG_null_followups.md P3: output sensitivity per decoder block (Gemma 4 E2B-it, bf16 GGUF reference).

For block i: add z ~ N(0, I) to the block's output, scaled per token to ||z|| = EPS x ||h_i||, 3 draws (seeds 0-2);
sens_i = mean KL(reference || perturbed) over the first 200 qkl.py documents, final logits, non-pad positions
(token mean, as qkl.py). Reference logits are qkl.py's stored ones (same model, same documents).
Scores: prod_i = sens_i x err_i (err_i = Q4_K_M activation-error increment, as score_b4bc.py), and sens_i.
P3-a: Spearman with the single-block KL reduction (robust_b4b.json). Writes results/sens_probe.json.
Run with DIAG_LOW_VRAM=1 DIAG_ATTN=eager on the shared 3060 (as gemma_bos_queue.sh).

python sens_probe.py --scratch data/scratch/qkl_scratch_bos
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from extract_hf import load_causal_lm
from qkl import FILES, OFFICIAL, REPO, TOKENIZER, batches, log

N_SENS, EPS, DRAWS = 200, 0.02, 3


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--dev", default="cuda:0")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    tok.padding_side = "right"
    sc = Path(a.scratch)
    L = np.load(sc / "ref_logits.npy", mmap_mode="r")
    masks = np.load(sc / "ref_masks.npy")
    m = load_causal_lm(REPO, torch.float16, a.dev, gguf_file=FILES["ref"], official=OFFICIAL)
    layers = m.model.layers
    state = {"block": None, "gen": None}

    def make(i):
        def hook(mod, inp, out):
            if state["block"] != i:
                return out
            h = out[0] if isinstance(out, tuple) else out
            z = torch.randn(h.shape, generator=state["gen"], device="cpu").to(h.device, torch.float32)
            z = z / z.norm(dim=-1, keepdim=True) * EPS * h.float().norm(dim=-1, keepdim=True)
            h2 = (h.float() + z).to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return hook

    hs = [layer.register_forward_hook(make(i)) for i, layer in enumerate(layers)]

    def kl_run():
        tot, n = 0.0, 0
        for s, enc in batches(tok, a.dev, a.bs):
            if s >= N_SENS:
                break
            lq = torch.log_softmax(m(**enc, use_cache=False).logits.float(), -1)
            lr = torch.log_softmax(torch.from_numpy(np.asarray(L[s:s + len(lq)])).to(a.dev).float(), -1)
            mk = torch.from_numpy(masks[s:s + len(lq)]).to(a.dev).bool()
            assert torch.equal(mk, enc["attention_mask"].bool())
            kl = (lr.exp() * (lr - lq)).sum(-1)
            tot += float(torch.where(mk, kl, torch.zeros_like(kl)).sum())
            n += int(mk.sum())
        return tot / n

    state["block"] = None
    base = kl_run()
    log("clean KL vs stored reference (should be ~0):", base)
    sens = []
    for i in range(len(layers)):
        v = []
        for d in range(DRAWS):
            state["block"], state["gen"] = i, torch.Generator().manual_seed(1000 * i + d)
            v.append(kl_run())
        sens.append(float(np.mean(v)))
        log("block", i, "sens", round(sens[-1], 5), [round(x, 5) for x in v])
    for h in hs:
        h.remove()
    R = Path("results")
    rows = sorted((r for r in json.loads((R / "gq_bf16__quantlens.json").read_text())["rows"] if r["variant"] == "gq_q4km"),
                  key=lambda r: r["layer"])
    err = np.diff([r["rel_l2"] for r in rows])                   # decoder j: state j+1 - state j
    b = json.loads((R / "robust_b4b.json").read_text())["configs"]
    red = np.array([b["none"]["kl_token_mean"] - b[f"block{i}"]["kl_token_mean"] for i in range(len(layers))])
    prod = np.array(sens) * err
    rho_prod, rho_sens, rho_err = (float(spearmanr(x, red)[0]) for x in (prod, np.array(sens), err))
    top4 = sorted(int(i) for i in np.argsort(-prod, kind="stable")[:4])
    out = {"eps": EPS, "draws": DRAWS, "n_docs": N_SENS, "clean_kl": base, "sens": sens, "err": err.tolist(),
           "prod": prod.tolist(), "kl_reduction": red.tolist(), "rho_prod": rho_prod, "rho_sens": rho_sens,
           "rho_err": rho_err, "top4_prod": top4,
           "P3a": {"supported": bool(rho_prod > 0.40 and rho_prod > rho_err)}}
    (R / "sens_probe.json").write_text(json.dumps(out, indent=1))
    log("rho prod/sens/err", round(rho_prod, 3), round(rho_sens, 3), round(rho_err, 3), "top4", top4, out["P3a"])


if __name__ == "__main__":
    main()
