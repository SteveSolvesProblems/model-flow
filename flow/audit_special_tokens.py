"""Audit: did every cache feed its model the special tokens the model was pretrained with?

The Gemma 4 E2B-it tokenizer adds no <bos>, while Gemma needs it (NOTE_geometry.md, 2026-10-05). This checks every
cache for that class of error. For each cache (one per name, read from <name>_L00/meta.json or <name>/meta.json):
  used      the tokenizer the extraction loaded: meta "tokenizer" if recorded, else what the launching script passed
            (extract_hf.py: the model repo; token_extract.py: google/gemma-4-E2B-it unless --tokenizer; see used_tokenizer)
  reference the pretrained base model's own tokenizer (same family, not instruction-tuned)
  first     the first token id the used tokenizer puts before "Hello world", and what the reference puts there
A cache passes when used and reference agree on the leading special tokens. Vision / MNIST caches are skipped.
Writes results/audit_special_tokens.json.
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

from common import CACHE_DIRS, RESULTS

BASE = {"google/gemma-4-E2B-it": "google/gemma-4-E2B", "google/gemma-4-E4B-it": "google/gemma-4-E4B",
        "bartowski/google_gemma-4-E2B-it-GGUF": "google/gemma-4-E2B",
        "google/gemma-4-E2B-it-qat-q4_0-gguf": "google/gemma-4-E2B",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct": "HuggingFaceTB/SmolLM2-1.7B",
        "Qwen/Qwen3-1.7B": "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3.5-2B": "Qwen/Qwen3.5-2B-Base",
        "Qwen/Qwen3.5-0.8B": "Qwen/Qwen3.5-0.8B-Base", "Qwen/Qwen3.5-4B": "Qwen/Qwen3.5-4B-Base"}
SKIP_MODES = {"vision"}
TEXT = "Hello world"


def caches():
    out = {}
    for root in CACHE_DIRS:
        for m in sorted(Path(root).glob("*/meta.json")):
            d = m.parent.name
            name = d[:-4] if d[-4:-2] == "_L" and d[-2:].isdigit() else d
            if d[-4:-2] == "_L" and d[-2:] != "00":
                continue
            if name not in out:
                out[name] = (json.loads(m.read_text()), str(m))
    return out


def used_tokenizer(name, meta):
    if meta.get("tokenizer"):
        return meta["tokenizer"]
    model = meta.get("model") or ""
    if meta.get("unit") == "token":              # token_extract.py did not record its tokenizer before 2026-10-07
        if name == "tok_qwen3":
            return "Qwen/Qwen3-1.7B"             # qwen_token.sh passed --tokenizer
        if name.endswith("_bos"):
            return "google/gemma-4-E2B"          # --tokenizer google/gemma-4-E2B; token_extract.py asserts <bos>
        return "google/gemma-4-E2B-it"           # token_extract.py's default tokenizer
    return model


def lead(tok):
    ids = tok(TEXT)["input_ids"]
    plain = tok(TEXT, add_special_tokens=False)["input_ids"]
    k = len(ids) - len(plain) if ids[-len(plain):] == plain else None   # number of leading special tokens
    return ids[:k] if k else []


def main():
    rows, toks = [], {}
    get = lambda r: toks.setdefault(r, AutoTokenizer.from_pretrained(r))
    for name, (meta, path) in sorted(caches().items()):
        model = meta.get("model")
        if not model or meta.get("mode") in SKIP_MODES or "/" not in str(model):
            continue
        used = used_tokenizer(name, meta)
        ref = BASE.get(model, model)
        try:
            u, r = lead(get(used)), lead(get(ref))
        except Exception as e:                       # noqa: BLE001 — record and move on
            rows.append({"cache": name, "model": model, "used": used, "reference": ref, "error": repr(e)[:200]})
            continue
        rows.append({"cache": name, "model": model, "used": used, "reference": ref, "lead_used": u, "lead_ref": r,
                     "ok": u == r, "bos_in_vocab": get(ref).bos_token_id, "meta": path})
        print(("ok  " if u == r else "FAIL"), name, model, "used", used, u, "ref", r)
    bad = [r["cache"] for r in rows if not r.get("ok")]
    out = {"n": len(rows), "failures": bad, "rows": rows}
    (RESULTS / "audit_special_tokens.json").write_text(json.dumps(out, indent=1))
    print(len(rows), "caches;", len(bad), "fail or error:", bad)


if __name__ == "__main__":
    main()
