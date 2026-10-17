"""Do the flow map's groups (sankey_validity.py consensus codes, pca64, K = 4 and 8) track text length rather than
content? Same 1,500 eval articles as viewer_data.py / tok_meta.npz. Length = tokens per article (first 64 kept), in
three bins (< 48, 48-63, 64 = at the cap). Per layer: AMI(groups, length bin) and AMI(groups, topic).
Writes results/flow_confounds.json.
"""
import json

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami

from common import CACHE_DIRS, RESULTS


def main():
    meta = np.load(next(c / "tok_meta.npz" for c in CACHE_DIRS if (c / "tok_meta.npz").exists()), allow_pickle=True)
    n_tok = np.bincount(meta["eval"][:, 0], minlength=len(meta["eval_texts"]))   # positions per doc (no <bos> in this file)
    lbin = np.digitize(n_tok, [48, 64])
    SV = json.loads((RESULTS / "sankey_validity.json").read_text())
    l1 = [d["l1"] for d in json.loads((RESULTS / "viewer_data.json").read_text())["docs"]]
    out = {"share_at_cap": float((n_tok >= 64).mean()), "bin_counts": np.bincount(lbin).tolist()}
    for K in (4, 8):
        C = SV["per_layer"][f"pca64_K{K}"]["codes"]
        out[f"K{K}"] = {"length_ami": [float(ami(lbin, c)) for c in C], "topic_ami": [float(ami(l1, c)) for c in C]}
        print(K, "max length AMI (layers >= 1)", round(max(out[f"K{K}"]["length_ami"][1:]), 3),
              "median topic AMI", round(float(np.median(out[f"K{K}"]["topic_ami"][1:])), 3))
    print("share at cap", round(out["share_at_cap"], 3), "bins", out["bin_counts"])
    (RESULTS / "flow_confounds.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
