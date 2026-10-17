"""Q3 of research/PREREG_quantization_lens.md: mixed precision chosen by the lens vs by activation error.

Start from Q4_K_M, restore 4 decoder layers to the reference (bf16 GGUF), measure mean next-token
KL(reference ‖ mixed) over 1,000 held-out DBpedia test documents, all non-pad positions, final logits.

Two phases, so it runs on one P100 (reference and Q4_K_M don't both fit):
  --phase ref    run the reference once; write its fp16 logits (the model computes in fp16, so this is lossless)
                 and its decoder-layer weights to --scratch
  --phase mixed  load Q4_K_M; for each configuration, copy reference weights into the chosen layers, score KL
                 against the stored logits, restore the Q4_K_M weights

Documents: the DBpedia test split with the main study's sampling (seed 0), positions 6000–6999. These are
disjoint from the 6,000 test documents the lens used to choose layers (stricter than "held-out" requires).
Lens layer ℓ is hidden_states[ℓ], the output of decoder layer ℓ−1, so a jump at ℓ is attributed to decoder ℓ−1.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from extract_hf import dbpedia, load_causal_lm

REPO = "bartowski/google_gemma-4-E2B-it-GGUF"
OFFICIAL = "google/gemma-4-E2B-it"
TOKENIZER = "google/gemma-4-E2B"   # same vocabulary; the -it tokenizer adds no <bos> (NOTE_geometry.md, 2026-10-05)
FILES = {"ref": "google_gemma-4-E2B-it-bf16.gguf", "q4km": "google_gemma-4-E2B-it-Q4_K_M.gguf"}
N_DOC, OFFSET, MAX_LEN = 1000, 6000, 64


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def batches(tok, dev, bs):
    data, _ = dbpedia(1, OFFSET + N_DOC)
    texts = data["test"][0][OFFSET:OFFSET + N_DOC]
    for s in range(0, N_DOC, bs):
        enc = tok(texts[s:s + bs], padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
        yield s, enc.to(dev)


def choose(lens_json, n_layers, k=4):
    """Pre-registered layer choices. n_layers = number of decoder layers."""
    rows = [r for r in json.loads(Path(lens_json).read_text())["rows"] if r["variant"] == "gq_q4km"]
    rows.sort(key=lambda r: r["layer"])
    flip = np.array([r["flip_norm_K512"] for r in rows])
    rel = np.array([r["rel_l2"] for r in rows])
    # Amendment 3: raw flip (1 − NMI, no re-seed normalization), declared secondary before any KL was seen
    raw = np.array([r["flip_norm_K512"] * r["ceiling_K512"] + 1 - r["ceiling_K512"] for r in rows])
    assert len(flip) == n_layers + 1
    top = lambda v: sorted(int(i) for i in np.argsort(-np.diff(v), kind="stable")[:k])  # diff[j] = v[j+1]-v[j] → decoder j
    cfg = {"none": [], "all": list(range(n_layers)), "lens": top(flip), "lens_raw": top(raw), "actl2": top(rel),
           "first4": list(range(k)), "last4": list(range(n_layers - k, n_layers))}
    for s in range(5):
        cfg[f"random{s}"] = sorted(int(i) for i in np.random.default_rng(s).choice(n_layers, k, replace=False))
    return cfg


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["ref", "mixed"])
    ap.add_argument("--scratch", default=str(Path("data/qkl_scratch")))
    ap.add_argument("--lens-json", default="results/gq_bf16__quantlens.json")
    ap.add_argument("--out", default="results/gq__qkl.json")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--custom", default=None, help="with --extra custom: comma-separated blocks to restore")
    ap.add_argument("--extra", default=None, choices=["single", "random20", "custom"],
                    help="PREREG_robustness.md B4: restore each block alone, or 20 random 4-block sets")
    ap.add_argument("--per-doc", default=None, help="also save per-document KL sums and token counts (.npz)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    tok.padding_side = "right"
    assert tok("x")["input_ids"][0] == tok.bos_token_id
    sc = Path(a.scratch)
    sc.mkdir(parents=True, exist_ok=True)

    if a.phase == "ref":
        m = load_causal_lm(REPO, torch.float16, a.dev, gguf_file=FILES["ref"], official=OFFICIAL)
        V = m.get_output_embeddings().weight.shape[0]
        L = np.lib.format.open_memmap(sc / "ref_logits.npy", "w+", np.float16, (N_DOC, MAX_LEN, V))
        masks = np.zeros((N_DOC, MAX_LEN), np.int8)
        for s, enc in batches(tok, a.dev, a.bs):
            lg = m(**enc, use_cache=False).logits
            if not torch.isfinite(lg).all():
                raise RuntimeError(f"non-finite reference logits at batch {s}")
            L[s:s + len(lg)] = lg.cpu().numpy()
            masks[s:s + len(lg)] = enc["attention_mask"].cpu().numpy()
        L.flush()
        np.save(sc / "ref_masks.npy", masks)
        torch.save([{k: v.cpu() for k, v in layer.state_dict().items()} for layer in m.model.layers],
                   sc / "ref_layers.pt")
        log(f"reference done: logits {L.shape}, {len(m.model.layers)} layers saved")
        return

    m = load_causal_lm(REPO, torch.float16, a.dev, gguf_file=FILES["q4km"], official=OFFICIAL)
    layers = m.model.layers
    q4 = [{k: v.cpu().clone() for k, v in layer.state_dict().items()} for layer in layers]
    ref = torch.load(sc / "ref_layers.pt")
    L = np.load(sc / "ref_logits.npy", mmap_mode="r")
    masks = np.load(sc / "ref_masks.npy")
    if a.extra == "single":
        cfgs = {"none": [], **{f"block{i}": [i] for i in range(len(layers))}}
    elif a.extra == "custom":                    # PREREG_null_followups.md P3-b
        cfgs = {"none": [], "custom": sorted(int(i) for i in a.custom.split(","))}
    elif a.extra == "random20":
        cfgs = {"none": [], **{f"random{s}": sorted(np.random.default_rng(s).choice(len(layers), 4, replace=False).tolist())
                               for s in range(20)}}
    else:
        cfgs = choose(a.lens_json, len(layers))
    log("configurations:", json.dumps(cfgs))
    res, docs = {}, {}
    for name, restore in cfgs.items():
        for i in restore:
            layers[i].load_state_dict(ref[i])
        tot, ntok, per_doc, doc_sum, doc_n = 0.0, 0, [], [], []
        for s, enc in batches(tok, a.dev, a.bs):
            lq = torch.log_softmax(m(**enc, use_cache=False).logits.float(), -1)
            lr = torch.log_softmax(torch.from_numpy(np.asarray(L[s:s + len(lq)])).to(a.dev).float(), -1)
            kl = (lr.exp() * (lr - lq)).sum(-1)                      # (bs, T)
            mk = torch.from_numpy(masks[s:s + len(lq)]).to(a.dev).bool()
            assert torch.equal(mk, enc["attention_mask"].bool())
            kl = torch.where(mk, kl, torch.zeros_like(kl))
            tot += float(kl.sum())
            ntok += int(mk.sum())
            per_doc += (kl.sum(1) / mk.sum(1)).tolist()
            doc_sum += kl.double().sum(1).tolist()
            doc_n += mk.sum(1).tolist()
        for i in restore:
            layers[i].load_state_dict(q4[i])
        docs[name] = (np.array(doc_sum), np.array(doc_n))
        res[name] = {"layers": restore, "kl_token_mean": tot / ntok, "kl_doc_mean": float(np.mean(per_doc)),
                     "kl_doc_sd": float(np.std(per_doc))}
        log(f"{name:8s} {restore if len(restore) <= 4 else 'all'} KL={tot / ntok:.5f}")
    if a.extra:
        Path(a.out).write_text(json.dumps({"extra": a.extra, "configs": res}, indent=2))
        if a.per_doc:
            np.savez(a.per_doc, **{f"{k}_sum": v[0] for k, v in docs.items()}, **{f"{k}_ntok": v[1] for k, v in docs.items()})
        return
    if a.per_doc:
        np.savez(a.per_doc, **{f"{k}_sum": v[0] for k, v in docs.items()}, **{f"{k}_ntok": v[1] for k, v in docs.items()})
    rnd = [res[f"random{s}"]["kl_token_mean"] for s in range(5)]
    lens, act = res["lens"]["kl_token_mean"], res["actl2"]["kl_token_mean"]
    verdict = ("supported" if lens < act and lens < np.mean(rnd) else
               "partial" if lens < np.mean(rnd) else "killed")
    raw_ = res["lens_raw"]["kl_token_mean"]
    verdict_raw = ("supported" if raw_ < act and raw_ < np.mean(rnd) else
                   "partial" if raw_ < np.mean(rnd) else "killed")
    out = {"verdict_Q3_secondary_rawflip": verdict_raw, "configs": res, "random_mean": float(np.mean(rnd)), "random_sd": float(np.std(rnd)),
           "verdict_Q3": verdict, "docs": f"dbpedia test[{OFFSET}:{OFFSET + N_DOC}]", "max_len": MAX_LEN}
    Path(a.out).write_text(json.dumps(out, indent=2))
    log("Q3 verdict:", verdict, f"lens={lens:.5f} actl2={act:.5f} random={np.mean(rnd):.5f}±{np.std(rnd):.5f}")


if __name__ == "__main__":
    main()
