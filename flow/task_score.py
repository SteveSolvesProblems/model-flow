"""research/PREREG_overnight.md E2: map stability vs task performance per layer for Banking77 (intent p@10) and STS-B
(Spearman of pair cosine vs gold), mean and last-token pooling. Writes results/task_score.json."""
import json

import faiss
import numpy as np
from scipy.stats import spearmanr

from common import RESULTS, load_cache, set_threads
from lens import frame, layers_of, log
from partition import assign, kmeans
from token_lens import fast_nmi

MODELS = ["qwen3_17b", "qwen35_2b", "smollm2_17b", "gemma4_e2b", "qwen35_4b"]


def unit(X, mu):
    A = X - mu
    return np.ascontiguousarray(A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12), dtype=np.float32)


def outcome(data, name):
    Xtr, Xte = load_cache(name, "train"), load_cache(name, "test")
    mu = Xtr.mean(0)
    if data == "banking77":
        ytr, yte = load_cache(name, "train", "intent")[1], load_cache(name, "test", "intent")[1]
        idx = faiss.IndexFlatIP(Xtr.shape[1]); idx.add(unit(Xtr, mu))
        _, I = idx.search(unit(Xte, mu), 10)
        return float((ytr[I] == yte[:, None]).mean())
    gold = load_cache(name, "test", "score")[1]
    U = unit(Xte, mu)
    n = len(gold)
    return float(spearmanr((U[:n] * U[n:2 * n]).sum(1), gold).statistic)


def main():
    set_threads()
    out, verdict = {}, {}
    for data in ("banking77", "stsb"):
        for pool in ("mean", "last"):
            per = {}
            for m in MODELS:
                names = layers_of(f"{data}_{m}_{pool}")
                if not names:
                    continue
                stab, perf = [], []
                for name in names:
                    Ftr, Fte = frame(load_cache(name, "train"), load_cache(name, "test"))
                    c = [assign(Fte, kmeans(Ftr, 512, seed=s, nredo=1))[0] for s in (0, 1)]
                    stab.append(fast_nmi(c[0], c[1]))
                    perf.append(outcome(data, name))
                stab, perf = np.array(stab), np.array(perf)
                S_all = int(np.flatnonzero(stab == stab.max()).max())
                S_no0 = 1 + int(np.flatnonzero(stab[1:] == stab[1:].max()).max())
                S = S_no0 if pool == "last" else S_all
                L = len(perf)
                per[m] = {"S": S, "S_frozen": S_all, "S_no_layer0": S_no0, "oracle": int(perf.argmax()),
                          "regret_S": float(perf.max() - perf[S]), "regret_S_frozen": float(perf.max() - perf[S_all]),
                          "regret_S_no_layer0": float(perf.max() - perf[S_no0]),
                          "regret_last": float(perf.max() - perf[L - 1]), "stability": stab.tolist(), "perf": perf.tolist()}
                log(data, pool, m, {k: v for k, v in per[m].items() if not isinstance(v, list)})
            out[f"{data}_{pool}"] = per
            better = sum(v["regret_S"] < v["regret_last"] - 1e-12 for v in per.values())
            not_better = sum(v["regret_S"] >= v["regret_last"] - 1e-12 for v in per.values())
            verdict[f"{data}_{pool}"] = {"n_models": len(per), "S_better_than_last": int(better),
                                         "S_not_better": int(not_better),
                                         "mean_regret_S": float(np.mean([v["regret_S"] for v in per.values()])),
                                         "mean_regret_last": float(np.mean([v["regret_last"] for v in per.values()]))}
    verdict["C1_holds"] = verdict["banking77_mean"]["S_better_than_last"] >= 4
    verdict["C2_holds"] = verdict["banking77_last"]["S_not_better"] >= 3
    verdict["C3_holds"] = sum(m in out["stsb_mean"] and out["stsb_mean"][m]["regret_S"] < out["stsb_mean"][m]["regret_last"]
                              for m in MODELS[:4]) >= 3
    out["verdict"] = verdict
    (RESULTS / "task_score.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
