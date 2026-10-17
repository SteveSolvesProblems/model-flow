"""Score PREREG_token_lens.md Part 2 (Qwen3-1.7B) and the per-layer-embedding contrast with Gemma base."""
import json

import numpy as np

from common import RESULTS


def curves(f, model):
    R = sorted([r for r in json.loads((RESULTS / f).read_text())["rows"] if r["model"] == model], key=lambda r: r["layer"])
    return R, {k: np.array([r[k] for r in R]) for k in ("id_K512", "next_K512", "ctx_K512", "stab_K512")}


def main():
    Rq, q = curves("tok_qwen3__token_lens.json", "base")
    Rg, g = curves("tok__token_lens.json", "base")
    idv, nx, ctx = q["id_K512"], q["next_K512"], q["ctx_K512"]
    pk = int(idv.argmax()); drop = (idv[pk] - idv[-1]) / idv[pk]
    mn = int(idv[pk:].argmin()) + pk; rebound = float(idv[mn:].max() - idv[mn])
    cpk = int(ctx.argmax())
    gi = g["id_K512"]; gpk = int(gi.argmax()); gdrop = (gi[gpk] - gi[-1]) / gi[gpk]
    gmn = int(gi[gpk:].argmin()) + gpk; greb = float(gi[gmn:].max() - gi[gmn])
    out = {"n_layers": len(idv) - 1,
           "Q-T1": bool(pk <= 3 and drop >= 0.30), "id_peak_layer": pk, "id_drop": round(float(drop), 3),
           "Q-T2": bool(int(nx.argmax()) >= 21), "next_peak_layer": int(nx.argmax()),
           "Q-T3": bool(nx[-1] > idv[-1]), "id_last": round(float(idv[-1]), 3), "next_last": round(float(nx[-1]), 3),
           "Q-T4": bool(7 <= cpk <= 21 and Rq[cpk]["ctx_K512_ci"][0] > 0), "ctx_peak_layer": cpk,
           "ctx_peak": round(float(ctx[cpk]), 4), "ctx_peak_ci": Rq[cpk]["ctx_K512_ci"],
           "id_min_layer_after_peak": mn, "id_rebound": round(rebound, 3),
           "gemma_base": {"id_drop": round(float(gdrop), 3), "id_rebound": round(greb, 3), "id_min_layer": gmn},
           "contrast_supported": bool(drop > gdrop and rebound < 0.1),
           "qwen_curves": {k: [round(float(v), 4) for v in arr] for k, arr in q.items()}}
    (RESULTS / "token_qwen3_score.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "qwen_curves"}, indent=1))
    print("layer: id / next / ctx / stab")
    for l in list(range(0, len(idv), 3)) + [len(idv) - 1]:
        print(f"  L{l:02d} {idv[l]:.3f} {nx[l]:.3f} {ctx[l]:.4f} {q['stab_K512'][l]:.3f}")


if __name__ == "__main__":
    main()
