"""Per-layer tensor quantization types in the Gemma 4 E2B-it GGUFs: which decoder layers did llama.cpp's recipe
(use_more_bits) already give extra bits? Bits per weight are approximate (ggml block sizes). Writes results/gguf_layer_bits.json."""
import json
import re
from pathlib import Path

import numpy as np
from gguf import GGUFReader

from common import RESULTS

BPW = {"F32": 32, "F16": 16, "BF16": 16, "Q8_0": 8.5, "Q6_K": 6.5625, "Q5_K": 5.5, "Q5_0": 5.5, "Q4_K": 4.5, "Q4_0": 4.5,
       "Q3_K": 3.4375, "Q2_K": 2.625, "IQ2_S": 2.5, "IQ2_XS": 2.3125, "IQ3_S": 3.4375, "IQ3_XXS": 3.0625, "IQ4_XS": 4.25,
       "IQ4_NL": 4.5, "IQ2_XXS": 2.0625, "IQ1_M": 1.75}
SNAP = Path.home() / ".cache/huggingface/hub/models--bartowski--google_gemma-4-E2B-it-GGUF/snapshots"


def main():
    out = {}
    for f in sorted(SNAP.glob("*/*.gguf")):
        r = GGUFReader(str(f))
        per = {}
        for t in r.tensors:
            m = re.match(r"blk\.(\d+)\.(.+)\.weight", t.name)
            if not m:
                continue
            q = t.tensor_type.name
            li = int(m.group(1))
            per.setdefault(li, {"bits": 0.0, "n": 0, "types": {}})
            n = int(np.prod(t.shape))
            per[li]["bits"] += BPW.get(q, float("nan")) * n
            per[li]["n"] += n
            per[li]["types"][m.group(2)] = q
        layers = sorted(per)
        bpw = [per[l]["bits"] / per[l]["n"] for l in layers]
        med = float(np.nanmedian(bpw))
        out[f.name] = {"bpw_by_decoder_layer": bpw, "median_bpw": med,
                       "above_median_layers": [l for l, b in zip(layers, bpw) if b > med + 1e-6],
                       "types_layer0": per[0]["types"], "types_mid": per[layers[len(layers) // 2]]["types"]}
        print(f.name, round(med, 3), out[f.name]["above_median_layers"])
    (RESULTS / "gguf_layer_bits.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
