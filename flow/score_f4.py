"""Score research/PREREG_flow.md F4 (Amendment 1) from results/tok_ple__token_lens.json."""
import json

import numpy as np

from common import RESULTS


def rebound(idv):
    pk = int(idv.argmax())
    mn = int(idv[pk:].argmin()) + pk
    return mn, float(idv[mn:].max() - idv[mn])


def main():
    # base rows from the first run; ablated rows from the rerun (the first run skipped non-base/it names)
    rows = [r for r in json.loads((RESULTS / "tok_ple__token_lens.json").read_text())["rows"] if r["model"] == "base"]
    rows += json.loads((RESULTS / "tok_ple_abl__token_lens.json").read_text())["rows"]
    out = {}
    for m in ("base", "nople18", "nople0"):
        R = sorted([r for r in rows if r["model"] == m], key=lambda r: r["layer"])
        idv = np.array([r["id_K512"] for r in R])
        mn, rb = rebound(idv)
        out[m] = {"id_min_layer": mn, "id_rebound": round(rb, 4), "id_K512": [round(float(x), 4) for x in idv],
                  "next_K512": [round(float(r["next_K512"]), 4) for r in R]}
    out["F4a"] = bool(out["nople18"]["id_rebound"] <= 0.08)
    out["F4b"] = bool(out["nople18"]["id_rebound"] < 0.5 * out["base"]["id_rebound"])
    (RESULTS / "f4_score.json").write_text(json.dumps(out, indent=1))
    print({m: (out[m]["id_min_layer"], out[m]["id_rebound"]) for m in ("base", "nople18", "nople0")}, out["F4a"], out["F4b"])


if __name__ == "__main__":
    main()
