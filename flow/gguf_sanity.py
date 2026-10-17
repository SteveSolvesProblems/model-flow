"""Pre-registered loader sanity check (PREREG_quantization_lens.md): the bf16 GGUF, loaded through our GGUF path
and patched, must match the official safetensors tensor-for-tensor before any quant is trusted.

python gguf_sanity.py --repo bartowski/google_gemma-4-E2B-it-GGUF --file google_gemma-4-E2B-it-bf16.gguf \
                      --official google/gemma-4-E2B-it
"""
import argparse
import json

import torch

from extract_hf import load_causal_lm, log


def compare(k, g, r):
    if g.shape != r.shape:
        return {"key": k, "shape_mismatch": [list(g.shape), list(r.shape)]}
    # chunk along dim 0: the per-layer embedding table is ~2.3B params, too big to diff in float32 at once
    mx, dss, rss = 0.0, 0.0, 0.0
    g, r = g.reshape(g.shape[0], -1) if g.dim() else g.reshape(1), r.reshape(r.shape[0], -1) if r.dim() else r.reshape(1)
    for i in range(0, g.shape[0], 4096):
        gi, ri = g[i:i + 4096].double(), r[i:i + 4096].double()
        d = (gi - ri).abs()
        mx, dss, rss = max(mx, float(d.max())), dss + float((d * d).sum()), rss + float((ri * ri).sum())
    return {"key": k, "max_abs": mx, "rel_l2": (dss ** 0.5) / (rss ** 0.5 + 1e-12)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--official", required=True)
    ap.add_argument("--out", default="results/gguf_sanity.json")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors import safe_open
    # bf16 holds the GGUF bf16 weights exactly; float32 plus all official tensors ran out of RAM
    m = load_causal_lm(a.repo, torch.bfloat16, "cpu", gguf_file=a.file, official=a.official)
    sd = m.state_dict()
    import gc; gc.collect()
    rows, seen = [], set()
    # stream official tensors one at a time
    for f in [f for f in list_repo_files(a.official) if f.endswith(".safetensors")]:
        with safe_open(hf_hub_download(a.official, f), framework="pt") as st:
            for key in st.keys():
                k = key.replace("model.language_model.", "model.")
                if k in sd and k not in seen:
                    seen.add(k)
                    rows.append(compare(k, sd[k], st.get_tensor(key)))
    missing = [k for k in sd if k not in seen]
    worst = sorted([r for r in rows if "rel_l2" in r], key=lambda r: -r["rel_l2"])[:10]
    summary = {"n_params": len(sd), "n_compared": len(rows), "n_missing_in_official": len(missing),
               "missing_examples": missing[:10],
               "shape_mismatches": [r for r in rows if "shape_mismatch" in r][:10], "worst_rel_l2": worst}
    json.dump(summary, open(a.out, "w"), indent=2)
    log(json.dumps({k: v for k, v in summary.items() if k != "worst_rel_l2"}))
    log("worst rel_l2:", [(r["key"], round(r["rel_l2"], 6)) for r in worst[:5]])


if __name__ == "__main__":
    main()
