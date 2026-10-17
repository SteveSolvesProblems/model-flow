"""research/PREREG_token_lens.md: token-level features for the attractor lens (pilot).

Fit set: first 3,000 DBpedia train articles, at most 30 occurrences per canonical word (seed 0).
Evaluation set: the viewer's 1,500 test articles (all tokens). First 64 tokens; <bos> and padding dropped.
Canonical word = normalization as in Cheng et al. 2026 (NFKC -> NFD -> strip accents -> lowercase -> strip whitespace).

Writes cache/tok_<name>_L##/{fit,eval}_X.npy (float16) and, once, cache/tok_meta.npz (token bookkeeping).

python token_extract.py --name tok_base --model google/gemma-4-E2B
python token_extract.py --name tok_it --model bartowski/google_gemma-4-E2B-it-GGUF --gguf-file google_gemma-4-E2B-it-bf16.gguf
"""
import argparse
import json
import unicodedata

import numpy as np
import torch

from common import CACHE
from extract_hf import dbpedia, load_causal_lm, log

OFFICIAL = "google/gemma-4-E2B-it"
N_FIT_DOCS, CAP, MAX_LEN = 3000, 30, 64


def canon(s):
    s = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower().strip()
    return s if s else "<ws>"


def token_type(piece, text):
    t = text.strip()
    if any(c.isdigit() for c in t):
        return "digit"
    if t and not any(c.isalnum() for c in t):
        return "punct"
    if not t:
        return "space"
    return "word_start" if piece.startswith("▁") or text.startswith(" ") else "continuation"


def build_meta(tok, path):
    data, maps = dbpedia(24000, 6000)
    fit_texts = data["train"][0][:N_FIT_DOCS]
    rng = np.random.default_rng(0)
    eval_idx = np.sort(rng.choice(6000, 1500, replace=False))           # same sample as viewer_data.py
    eval_texts = [data["test"][0][i] for i in eval_idx]
    l1_names = [k for k, _ in sorted(maps["l1"].items(), key=lambda kv: kv[1])]
    l2_names = [k for k, _ in sorted(maps["l2"].items(), key=lambda kv: kv[1])]
    canon_ids, pieces, texts = {}, {}, {}

    def cid(t):
        if t not in pieces:
            pieces[t] = tok.convert_ids_to_tokens(int(t))
            texts[t] = tok.decode([int(t)])
        c = canon(texts[t])
        return canon_ids.setdefault(c, len(canon_ids))

    def rows(txts):
        out = []
        for d, t in enumerate(txts):
            ids = tok(t, truncation=True, max_length=MAX_LEN)["input_ids"]
            if tok.bos_token_id is not None and "gemma" in tok.name_or_path.lower():
                assert ids[0] == tok.bos_token_id, "Gemma input must start with <bos> (see NOTE_geometry.md, 2026-10-05)"
            start = 1   # position 0: <bos> in Gemma, attention-sink outliers in any model
            cs = [cid(i) for i in ids]
            for p in range(start, len(ids)):
                out.append((d, p, ids[p], cs[p], cs[p + 1] if p + 1 < len(ids) else -1))
        return np.array(out, dtype=np.int64)

    fit_all, ev = rows(fit_texts), rows(eval_texts)
    freq = np.bincount(fit_all[:, 3], minlength=len(canon_ids))
    keep = []
    for c in np.unique(fit_all[:, 3]):
        occ = np.flatnonzero(fit_all[:, 3] == c)
        keep.extend(occ if len(occ) <= CAP else rng.choice(occ, CAP, replace=False))
    fit = fit_all[np.sort(np.array(keep))]
    inv = {v: k for k, v in canon_ids.items()}
    vocab = np.array([inv[i] for i in range(len(canon_ids))], dtype=object)
    ttype = np.array([token_type(pieces[t], texts[t]) for t in ev[:, 2]])
    meta = {"fit": fit, "eval": ev, "freq": freq, "vocab": vocab, "eval_type": ttype,
            "eval_l1": data["test"][1]["l1"][eval_idx][ev[:, 0]], "eval_l2": data["test"][1]["l2"][eval_idx][ev[:, 0]],
            "eval_doc_index": eval_idx, "eval_texts": np.array(eval_texts, dtype=object),
            "fit_texts": np.array(fit_texts, dtype=object),
            "token_text": np.array([texts[t] for t in ev[:, 2]], dtype=object),
            "l1_names": np.array(l1_names, dtype=object), "l2_names": np.array(l2_names, dtype=object)}
    np.savez(path, **meta)
    log(f"meta: fit {len(fit)} of {len(fit_all)} tokens (cap {CAP}), eval {len(ev)} tokens, {len(canon_ids)} canonical words")
    return meta


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf-file", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gpu-mem", default="15GiB")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--tokenizer", default=OFFICIAL)
    ap.add_argument("--official", default=OFFICIAL, help="official repo for GGUF config/patching")
    ap.add_argument("--meta", default="tok_meta.npz")
    ap.add_argument("--ablate-ple", default=None,
                    help="PREREG_flow.md F4: zero Gemma's per-layer inputs for decoder blocks >= this index (0 = all)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    tok.padding_side = "right"
    mp = CACHE / a.meta
    meta = dict(np.load(mp, allow_pickle=True)) if mp.exists() else build_meta(tok, mp)
    m = load_causal_lm(a.model, getattr(torch, a.dtype), "cuda", offload=True, gpu_mem=a.gpu_mem, cpu_mem="24GiB",
                       gguf_file=a.gguf_file, official=a.official)
    if a.ablate_ple is not None:
        start = int(a.ablate_ple)
        mods = [mod for mod in m.modules() if hasattr(mod, "project_per_layer_inputs")]
        assert len(mods) == 1, f"expected one module with per-layer inputs, found {len(mods)}"
        orig = mods[0].project_per_layer_inputs

        def ablated(*args, **kw):
            out = orig(*args, **kw).clone()
            out[:, :, start:, :] = 0
            return out
        mods[0].project_per_layer_inputs = ablated
        log("per-layer inputs zeroed for decoder blocks", start, "and up of", mods[0].config.num_hidden_layers)
    for split, texts, rows in (("fit", meta["fit_texts"], meta["fit"]), ("eval", meta["eval_texts"], meta["eval"])):
        where = {}
        for r, (d, p) in enumerate(rows[:, :2]):
            where.setdefault(int(d), []).append((int(p), r))
        out = None
        for s in range(0, len(texts), a.batch):
            enc = tok(list(texts[s:s + a.batch]), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            if tok.bos_token_id is not None and "gemma" in tok.name_or_path.lower():
                assert (enc["input_ids"][:, 0] == tok.bos_token_id).all(), "missing <bos>"
            hs = m(**{k: v.to("cuda") for k, v in enc.items()}, output_hidden_states=True, use_cache=False).hidden_states
            if out is None:
                out = []
                for li in range(len(hs)):
                    dd = CACHE / f"{a.name}_L{li:02d}"
                    dd.mkdir(parents=True, exist_ok=True)
                    out.append(np.lib.format.open_memmap(dd / f"{split}_X.npy", "w+", np.float16, (len(rows), hs[0].shape[-1])))
            b_idx, t_idx, r_idx = [], [], []
            for b in range(len(enc["input_ids"])):
                d = s + b
                ids = enc["input_ids"][b]
                for p, r in where.get(d, []):
                    assert int(ids[p]) == int(rows[r, 2]), "token mismatch"
                    b_idx.append(b); t_idx.append(p); r_idx.append(r)
            for li, h in enumerate(hs):
                x = h[b_idx, t_idx].float()
                if not torch.isfinite(x).all():
                    raise RuntimeError(f"non-finite at layer {li}")
                out[li][r_idx] = x.cpu().numpy().astype(np.float16)
            if s % (a.batch * 40) == 0:
                log(split, s, len(texts))
        for o in out:
            o.flush()
        for li in range(len(out)):
            (CACHE / f"{a.name}_L{li:02d}" / "meta.json").write_text(json.dumps(
                {"model": a.model, "gguf_file": a.gguf_file, "layer": li, "dtype": a.dtype, "unit": "token",
                 "ablate_ple_from_block": a.ablate_ple, "tokenizer": a.tokenizer, "meta_file": a.meta}))
    log("done")


if __name__ == "__main__":
    main()
