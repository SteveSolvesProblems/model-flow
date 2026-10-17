"""Cache frozen features from HF models (run on a P100; analysis runs elsewhere on CPU).

  vision:  python extract_hf.py vision --model google/siglip2-base-patch16-224 --name siglip2_b
  embed:   python extract_hf.py embed  --model Qwen/Qwen3-Embedding-0.6B --name qwen3emb_06b
  llm:     python extract_hf.py llm    --model Qwen/Qwen3.5-2B --name qwen35_2b --dtype float32

Datasets: CIFAR-100 (labels fine=100, coarse=20) for vision; DBpedia Classes (l1=9, l2=70, l3=219) for text.
LLM features: mean-pooled residual stream over non-pad tokens, every layer (hidden_states[0] = embeddings),
one cache per layer: <name>_L<ii>.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from common import save_cache

DBPEDIA = "DeveloperOats/DBPedia_Classes"
CIFAR = "uoft-cs/cifar100"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dbpedia(n_train, n_test, seed=0):
    from datasets import load_dataset
    ds = load_dataset(DBPEDIA)
    out = {}
    maps = {}
    for split, n, src in (("train", n_train, "train"), ("test", n_test, "test")):
        d = ds[src].shuffle(seed=seed).select(range(min(n, len(ds[src]))))
        labels = {}
        for lv in ("l1", "l2", "l3"):
            if lv not in maps:
                maps[lv] = {v: i for i, v in enumerate(sorted(set(ds["train"][lv])))}
            labels[lv] = np.array([maps[lv].get(v, -1) for v in d[lv]])
        out[split] = (list(d["text"]), labels)
    return out, maps


def newsgroups():
    """20 Newsgroups (SetFit copy): fine = 20 groups, coarse = 7 top-level prefixes."""
    from datasets import load_dataset
    ds = load_dataset("SetFit/20_newsgroups")
    fine_names = sorted(set(ds["train"]["label_text"]))
    coarse_names = sorted({n.split(".")[0] for n in fine_names})
    fmap = {n: i for i, n in enumerate(fine_names)}
    cmap = {n: i for i, n in enumerate(coarse_names)}
    out = {}
    for split in ("train", "test"):
        d = ds[split]
        texts = [t if t.strip() else "(empty)" for t in d["text"]]
        lt = d["label_text"]
        out[split] = (texts, {"fine": np.array([fmap[x] for x in lt]),
                              "coarse": np.array([cmap[x.split(".")[0]] for x in lt])})
    return out, {"fine": fine_names, "coarse": coarse_names}


def yelp(n_train, n_test, seed=0):
    """Yelp Review Full (research/PREREG_stability_v2.md): stars = rating 0-4; sampled like dbpedia()."""
    from datasets import load_dataset
    ds = load_dataset("Yelp/yelp_review_full")
    out = {}
    for split, n in (("train", n_train), ("test", n_test)):
        d = ds[split].shuffle(seed=seed).select(range(min(n, len(ds[split]))))
        out[split] = (list(d["text"]), {"stars": np.array(d["label"])})
    return out, {"stars": [str(i + 1) for i in range(5)]}


def text_data(a):
    return {"20ng": newsgroups, "yelp": lambda: yelp(a.n_train, a.n_test)}.get(a.data, lambda: dbpedia(a.n_train, a.n_test))()


def meta(a, extra=None):
    from huggingface_hub import HfApi
    try:
        sha = HfApi().model_info(a.model, revision=getattr(a, "revision", None)).sha
    except Exception:        # offline: the local snapshot directory is named by its commit sha
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(a.model, "config.json", revision=getattr(a, "revision", None))
        sha = Path(p).parent.name if isinstance(p, str) else None
    return {"model": a.model, "revision": sha, "checkpoint": getattr(a, "revision", None), "mode": a.mode, **(extra or {})}


# ── vision ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_vision(a, dev):
    from datasets import load_dataset
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float32).to(dev).eval()
    is_siglip = "siglip" in a.model.lower()
    ds = load_dataset(CIFAR)
    img_col = "img" if "img" in ds["train"].column_names else "image"
    for split in ("train", "test"):
        d = ds[split]
        cls_f, mean_f = [], []
        for s in range(0, len(d), a.batch):
            imgs = [im.convert("RGB") for im in d[s:s + a.batch][img_col]]
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev)
            if is_siglip:
                out = model.vision_model(pixel_values=px)
                cls_f.append(out.pooler_output.float().cpu())
                mean_f.append(out.last_hidden_state.mean(1).float().cpu())
            else:
                out = model(pixel_values=px)
                n_reg = getattr(model.config, "num_register_tokens", 0)
                cls_f.append(out.last_hidden_state[:, 0].float().cpu())
                mean_f.append(out.last_hidden_state[:, 1 + n_reg:].mean(1).float().cpu())
            if s % (a.batch * 40) == 0:
                log(split, s, len(d))
        labels = {"fine": np.array(d["fine_label"]), "coarse": np.array(d["coarse_label"])}
        save_cache(a.name, split, torch.cat(cls_f).numpy(), labels, meta(a, {"pool": "cls/pooler"}))
        save_cache(a.name + "_mean", split, torch.cat(mean_f).numpy(), labels, meta(a, {"pool": "mean patch"}))


# ── text embedding model (last-token pooling, normalized) ────────────────────

@torch.no_grad()
def run_embed(a, dev):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float32).to(dev).eval()
    data, maps = text_data(a)
    for split, (texts, labels) in data.items():
        feats = []
        for s in range(0, len(texts), a.batch):
            enc = tok(texts[s:s + a.batch], padding=True, truncation=True, max_length=a.max_len,
                      return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state[:, -1]
            feats.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu())
        save_cache(a.name, split, torch.cat(feats).numpy(), labels,
                   meta(a, {"pool": "last token, L2-normalized", "max_len": a.max_len,
                            "label_sizes": {k: len(v) for k, v in maps.items()}}))
        log(split, "done")


# ── LLM residual stream, every layer ─────────────────────────────────────────

class _CpuModule(torch.nn.Module):
    """Run a wrapped module on the CPU (weights stay in RAM): for lookups such as Gemma's per-layer embeddings."""
    def __init__(self, mod):
        super().__init__()
        self._m = [mod.cpu()]        # a plain list, so .to(device) on the parent skips it

    def forward(self, x, *args, **kw):
        return self._m[0](x.cpu(), *args, **kw).to(x.device)


def low_vram(m, dev):
    """DIAG_LOW_VRAM=1 (shared desktop GPU): Gemma's per-layer embedding table (4.7 GB fp16) is looked up on the CPU,
    unused vision/audio towers stay in RAM. Hidden states are unchanged (same weights, same ops)."""
    if os.environ.get("DIAG_LOW_VRAM") != "1":
        return m.to(dev)
    for name, mod in list(m.named_modules()):
        parent, _, leaf = name.rpartition(".")
        if leaf == "embed_tokens_per_layer":
            setattr(m.get_submodule(parent), leaf, _CpuModule(mod))
            log("low-vram:", name, "on cpu")
    m.to(dev)
    for name, mod in m.named_modules():
        if name.rpartition(".")[2] in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
            mod.cpu()
    torch.cuda.empty_cache()
    return m


def _attn():
    """DIAG_ATTN=eager: Gemma 4 E2B-it in float16 overflows with SDPA on an RTX 3060 (non-finite from layer 2);
    eager attention stays finite. Unset = transformers' default."""
    a = os.environ.get("DIAG_ATTN")
    return {"attn_implementation": a} if a else {}


def load_causal_lm(name, dtype, dev, offload=False, gpu_mem="14GiB", cpu_mem="24GiB", gguf_file=None,
                   official=None, revision=None, use_safetensors=None):
    from transformers import AutoConfig, AutoModelForCausalLM
    if gguf_file:  # transformers dequantizes GGUF weights to `dtype` (weight-only quantization effect)
        cfg = AutoConfig.from_pretrained(official)
        cfg = getattr(cfg, "text_config", cfg)   # Gemma 4 GGUF config conversion fails; use the official one
        m = AutoModelForCausalLM.from_pretrained(name, gguf_file=gguf_file, config=cfg, dtype=dtype,
                                                 low_cpu_mem_usage=True, **_attn())
        patch_missing_from_official(m, official)
        if os.environ.get("DIAG_LOW_VRAM") == "1":
            return low_vram(m, dev).eval()
        return m.to(dev).eval() if not offload else _dispatch(m, gpu_mem, cpu_mem)
    if offload and os.environ.get("DIAG_LOW_VRAM") != "1":   # models larger than one P100: let accelerate spill layers to CPU RAM
        m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, device_map="auto",
                                                 max_memory={0: gpu_mem, "cpu": cpu_mem})
        return m.eval()
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, low_cpu_mem_usage=True, revision=revision,
                                             use_safetensors=use_safetensors, **_attn())
    return low_vram(m, dev).eval()


def weights_fingerprint(model):
    """sha256 of a few parameter tensors: distinguishes checkpoints even when hub revisions share a stale file
    (pythia-2.8b: one model.safetensors blob is shared by 8 of the 10 branches used here)."""
    import hashlib
    h = hashlib.sha256()
    ps = list(model.parameters())
    for p in (ps[1], ps[len(ps) // 2], ps[-2]):
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def official_tensors(official, want):
    """Stream selected tensors from the official safetensors without loading the whole model."""
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors import safe_open
    out = {}
    for f in [f for f in list_repo_files(official) if f.endswith(".safetensors")]:
        with safe_open(hf_hub_download(official, f), framework="pt") as st:
            for k in st.keys():
                kk = k.replace("model.language_model.", "model.")
                if want(kk):
                    out[kk] = st.get_tensor(k)
    return out


def patch_missing_from_official(m, official):
    """transformers' GGUF mapping drops Gemma 4 `layer_scalar` (learned per-layer scales). Restore them."""
    names = {n for n, _ in m.named_parameters()} | {n for n, _ in m.named_buffers()}
    fix = official_tensors(official, lambda k: k.endswith("layer_scalar") and k in names)
    sd = m.state_dict()
    for k, v in fix.items():
        sd[k].copy_(v.to(sd[k].dtype))
    log(f"patched {len(fix)} tensors from {official} (e.g. {sorted(fix)[:1]})")
    if not fix:
        log("WARNING: nothing patched — check whether layer_scalar is still missing")


def _dispatch(m, gpu_mem, cpu_mem):
    from accelerate import dispatch_model, infer_auto_device_map
    dm = infer_auto_device_map(m, max_memory={0: gpu_mem, "cpu": cpu_mem},
                               no_split_module_classes=getattr(m, "_no_split_modules", None))
    return dispatch_model(m, device_map=dm).eval()


def load_encoder(a, dtype, dev):
    """Bidirectional encoder (e.g. ModernBERT). --weights-from REPO:FILE:PREFIX swaps in a fine-tuned encoder stored
    inside another checkpoint (e.g. convaiinnovations/laya:model.safetensors:encoder.); strict key match required."""
    from transformers import AutoModel
    m = AutoModel.from_pretrained(a.model, dtype=dtype, revision=a.revision)
    if a.weights_from:
        from huggingface_hub import HfApi, hf_hub_download
        from safetensors.torch import load_file
        repo, fname, prefix = a.weights_from.split(":")
        a.weights_sha = HfApi().model_info(repo).sha
        sd = {k[len(prefix):]: v for k, v in load_file(hf_hub_download(repo, fname, revision=a.weights_sha)).items()
              if k.startswith(prefix)}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        missing = [k for k in missing if "rotary" not in k and "inv_freq" not in k]
        if missing or unexpected:
            raise RuntimeError(f"weight swap mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
        log(f"loaded {len(sd)} encoder tensors from {repo}@{a.weights_sha[:10]}:{fname} (prefix {prefix})")
    return m.to(dev).eval()


@torch.no_grad()
def run_encoder(a, dev):
    run_llm(a, dev, encoder=True)


@torch.no_grad()
def run_llm(a, dev, encoder=False):
    from transformers import AutoTokenizer
    dtype = getattr(torch, a.dtype)
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model, revision=a.revision)
    if tok.pad_token is None and tok.eos_token is None and a.revision:
        # early Pythia-2.8b checkpoints ship no eos in the tokenizer config; the tokenizer is identical across steps
        tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if "gemma" in (a.tokenizer or a.model).lower():
        # the -it tokenizers add no <bos>, which Gemma needs (NOTE_geometry.md, 2026-10-05); use the base tokenizer
        assert tok("x")["input_ids"][0] == tok.bos_token_id, "Gemma input must start with <bos>: --tokenizer google/gemma-4-E2B"
    model = load_encoder(a, dtype, dev) if encoder else load_causal_lm(
        a.model, dtype, dev, a.offload, a.gpu_mem, a.cpu_mem, a.gguf_file, a.official or a.tokenizer, revision=a.revision,
        use_safetensors=False if a.no_safetensors else None)
    fp = weights_fingerprint(model)
    log("weights fingerprint", fp[:16])
    extra = {} if encoder else {"use_cache": False}
    data, maps = text_data(a)
    layer_types = getattr(getattr(model.config, "text_config", model.config), "layer_types", None)
    for split, (texts, labels) in data.items():
        per_layer = None
        n_bad = 0
        for s in range(0, len(texts), a.batch):
            enc = tok(texts[s:s + a.batch], padding=True, truncation=True, max_length=a.max_len,
                      return_tensors="pt").to(dev)
            out = model(**enc, output_hidden_states=True, **extra)
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = [((h.float() * m).sum(1) / m.sum(1)) for h in out.hidden_states]
            bad = sum(int(~torch.isfinite(p).all()) for p in pooled)
            n_bad += bad
            if per_layer is None:
                per_layer = [[] for _ in pooled]
            for i, p in enumerate(pooled):
                per_layer[i].append(p.half().cpu())
            if s % (a.batch * 50) == 0:
                log(split, s, len(texts), "nonfinite batches so far:", n_bad)
        if n_bad:
            raise RuntimeError(f"{n_bad} non-finite layer outputs — rerun with --dtype float32")
        for i, chunks in enumerate(per_layer):
            X = torch.cat(chunks).numpy()
            save_cache(f"{a.name}_L{i:02d}", split, X, labels,
                       meta(a, {"data": a.data, "tokenizer": a.tokenizer or a.model, "bos": bool(enc["input_ids"][0, 0] == tok.bos_token_id), "gguf_file": a.gguf_file, "weights_from": a.weights_from,
                                "weights_sha": getattr(a, "weights_sha", None), "weights_fingerprint": fp, "layer": i, "n_layers": len(per_layer) - 1, "pool": "mean over tokens",
                                "max_len": a.max_len, "dtype": a.dtype,
                                "layer_type": (layer_types[i - 1] if layer_types and i > 0 else
                                               ("embeddings" if i == 0 else None)),
                                "label_sizes": {k: len(v) for k, v in maps.items()}}))
        log(split, "saved", len(per_layer), "layers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["vision", "embed", "llm", "encoder"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=24000)
    ap.add_argument("--n-test", type=int, default=6000)
    ap.add_argument("--data", default="dbpedia", choices=["dbpedia", "20ng", "yelp"])
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--gpu-mem", default="14GiB")
    ap.add_argument("--gguf-file", default=None, help="load this GGUF file from --model repo (dequantized)")
    ap.add_argument("--tokenizer", default=None, help="tokenizer repo, if different from --model")
    ap.add_argument("--official", default=None, help="GGUF: official repo for config and layer_scalar (default --tokenizer)")
    ap.add_argument("--cpu-mem", default="24GiB")
    ap.add_argument("--no-safetensors", action="store_true", help="load pytorch_model.bin (stale safetensors on hub)")
    ap.add_argument("--revision", default=None, help="model revision, e.g. a Pythia checkpoint such as step1000")
    ap.add_argument("--weights-from", default=None, help="encoder mode: REPO:FILE:PREFIX of fine-tuned encoder weights")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
    {"vision": run_vision, "embed": run_embed, "llm": run_llm, "encoder": run_encoder}[a.mode](a, dev)
    log("done")


if __name__ == "__main__":
    main()
