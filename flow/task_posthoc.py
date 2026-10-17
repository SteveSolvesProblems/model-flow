"""POST HOC (not pre-registered): E2 outcomes rescored outside the raw centered space.
From layer ~3 on, a handful of "massive activation" dimensions hold >98% of the variance of mean-pooled features of
short texts (Banking77, STS-B), so raw cosine there mostly compares those dimensions. This rescoring computes the same
outcomes (Banking77 intent p@10, STS-B Spearman) after per-dimension z-scoring and in the whiten-64 frame map stability
already uses (both fit on train), and recomputes regret for the selections in results/task_score.json.
Writes results/task_posthoc.json."""
import json

import faiss
import numpy as np
from scipy.stats import spearmanr

from common import RESULTS, load_cache, set_threads
from lens import layers_of, log
from quant_lens import fit_frame


def unit(A):
    return np.ascontiguousarray(A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12), dtype=np.float32)


def outcomes(data, name):
    Xtr, Xte = load_cache(name, "train"), load_cache(name, "test")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    v = np.sort(Xtr.var(0))[::-1]
    _, W = fit_frame(Xtr)
    spaces = {"centered": (Xtr - mu, Xte - mu), "zscored": ((Xtr - mu) / sd, (Xte - mu) / sd),
              "whiten64": ((Xtr - mu) @ W, (Xte - mu) @ W)}
    r = {"top5_var_share": float(v[:5].sum() / v.sum())}
    for k, (A, B) in spaces.items():
        if data == "banking77":
            ytr, yte = load_cache(name, "train", "intent")[1], load_cache(name, "test", "intent")[1]
            idx = faiss.IndexFlatIP(A.shape[1]); idx.add(unit(A))
            _, I = idx.search(unit(B), 10)
            r[k] = float((ytr[I] == yte[:, None]).mean())
        else:
            gold = load_cache(name, "test", "score")[1]
            U, n = unit(B), len(gold)
            r[k] = float(spearmanr((U[:n] * U[n:2 * n]).sum(1), gold).statistic)
    return r


def main():
    set_threads()
    sel = json.loads((RESULTS / "task_score.json").read_text())
    out = {}
    for task, per in sel.items():
        if task == "verdict":
            continue
        data = task.split("_")[0]
        out[task] = {}
        for m, v in per.items():
            rows = [outcomes(data, n) for n in layers_of(f"{task.replace(data + '_', data + '_' + m + '_')}")]
            res = {"S": v["S"], "top5_var_share": [x["top5_var_share"] for x in rows]}
            for k in ("centered", "zscored", "whiten64"):
                p = np.array([x[k] for x in rows])
                res[k] = {"perf": p.tolist(), "oracle": int(p.argmax()), "regret_S": float(p.max() - p[v["S"]]),
                          "regret_last": float(p.max() - p[-1]), "S_better": bool(p[v["S"]] > p[-1] + 1e-12)}
            out[task][m] = res
            log(task, m, {k: (res[k]["oracle"], round(res[k]["regret_S"], 3), round(res[k]["regret_last"], 3))
                          for k in ("centered", "zscored", "whiten64")})
    summ = {}
    for task, per in out.items():
        for k in ("centered", "zscored", "whiten64"):
            summ[f"{task}|{k}"] = {"S_better_than_last": sum(r[k]["S_better"] for r in per.values()), "n": len(per),
                                   "mean_regret_S": float(np.mean([r[k]["regret_S"] for r in per.values()])),
                                   "mean_regret_last": float(np.mean([r[k]["regret_last"] for r in per.values()]))}
    out["summary"] = summ
    (RESULTS / "task_posthoc.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
