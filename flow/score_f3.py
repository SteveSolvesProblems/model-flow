"""research/PREREG_flow.md F3-a/b/c: what chat tuning moves. Reads <post>__select.json, <base>__select.json and
<base>__chatflips_<post>.json (quant_lens.py, K = 512). Writes results/flow_f3.json.

python score_f3.py
"""
import json

import numpy as np

from common import RESULTS

PAIRS = [("qwen3_17b_base", "qwen3_17b"), ("qwen35_2b_base", "qwen35_2b"), ("smollm2_17b", "smollm2_17b_it"),
         ("gemma4_e2b", "gemma4_e2b_it")]


def p10_and_S(name, lab):
    rows = json.loads((RESULTS / f"{name}__select.json").read_text())["rows"]
    p = np.array([r[f"p@10_{lab}"] for r in rows])
    st = np.array([r["stability_K512"] for r in rows])
    S = int(np.flatnonzero(st == st.max()).max())
    return p, S


def main():
    out, a, b, c = {}, 0, 0, 0
    for base, post in PAIRS:
        for sfx in ("", "_20ng"):
            lab = "fine" if sfx else "l2"
            bp, _ = p10_and_S(base + sfx, lab)
            pp, S = p10_and_S(post + sfx, lab)
            fl = json.loads((RESULTS / f"{base + sfx}__chatflips_{post + sfx}.json").read_text())["rows"]
            ch = np.array([r["changed_K512"] for r in sorted(fl, key=lambda r: r["layer"])])
            L = len(ch) - 1
            first_half = ch[1:L // 2 + 1].mean()            # layers 1 .. L/2 (layer 0 = embeddings)
            last_q = ch[L - L // 4 + 1:].mean()              # last quarter of layers
            r = {"base_last_regret": float(bp.max() - bp[-1]), "post_last_regret": float(pp.max() - pp[-1]),
                 "post_S": S, "post_oracle": int(pp.argmax()), "post_S_regret": float(pp.max() - pp[S]),
                 "flips_first_half": float(first_half), "flips_last_quarter": float(last_q), "flips": ch.tolist()}
            r["F3a"] = bool(r["post_last_regret"] > r["base_last_regret"])
            r["F3b"] = bool(last_q > first_half)
            r["F3c"] = bool(r["post_S_regret"] <= 0.010)
            a, b, c = a + r["F3a"], b + r["F3b"], c + r["F3c"]
            out[f"{base + sfx}|{post + sfx}"] = r
            print(f"{post + sfx:22s} last-regret base {r['base_last_regret']:.3f} post {r['post_last_regret']:.3f} | flips "
                  f"first half {first_half:.3f} last quarter {last_q:.3f} | S {S} oracle {r['post_oracle']} regret {r['post_S_regret']:.3f}")
    out["summary"] = {"F3a_count": int(a), "F3b_count": int(b), "F3c_count": int(c),
                      "F3a": bool(a >= 6), "F3b": bool(b >= 6), "F3c": bool(c >= 6)}
    (RESULTS / "flow_f3.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
