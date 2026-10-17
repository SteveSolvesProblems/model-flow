"""POST HOC check (not pre-registered): are QAT's near-total bin flips at layers 1-3 a global shift?
Assign each variant's test features to the bf16 reference's K=512 bins (reference frame, seed 0) three ways:
raw (as in the quantization study), re-centered on the variant's own train mean, and re-centered + rescaled to the
reference's mean centered norm. Writes results/qat_shift.json."""
import json

import numpy as np

from common import RESULTS, load_cache, set_threads
from lens import layers_of, log
from partition import assign, kmeans
from quant_lens import fit_frame

VARIANTS = ["gq_qat4_0", "gq_q4_0"]


def main():
    set_threads()
    out = {v: [] for v in VARIANTS}
    for li, ref in enumerate(layers_of("gq_bf16")):
        Rtr, Rte = load_cache(ref, "train"), load_cache(ref, "test")
        mu, W = fit_frame(Rtr)
        C = kmeans((Rtr - mu) @ W, 512, seed=0, nredo=1)
        base = assign((Rte - mu) @ W, C)[0]
        r_norm = np.linalg.norm(Rtr - mu, axis=1).mean()
        for v in VARIANTS:
            name = ref.replace("gq_bf16", v)
            Vtr, Vte = load_cache(name, "train"), load_cache(name, "test")
            vmu = Vtr.mean(0)
            scale = r_norm / np.linalg.norm(Vtr - vmu, axis=1).mean()
            res = {"layer": li}
            for k, X in (("raw", Vte), ("recentered", Vte - vmu + mu), ("recentered_rescaled", (Vte - vmu) * scale + mu)):
                res[k] = float((assign((X - mu) @ W, C)[0] != base).mean())
            res["mean_shift_over_spread"] = float(np.linalg.norm(vmu - mu) / r_norm)
            res["scale"] = float(scale)
            out[v].append(res)
            log(v, li, {k: round(x, 3) for k, x in res.items() if isinstance(x, float)})
    (RESULTS / "qat_shift.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
