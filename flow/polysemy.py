"""research/PREREG_overnight.md E6 (+ Amendment 1): a label-free polysemy score from token bins, checked against WordNet
synset counts with frequency and topic-spread controls. Writes results/polysemy.json.

python polysemy.py
"""
import json

import numpy as np
from scipy.stats import rankdata, spearmanr

from common import CACHE, RESULTS, load_cache, set_threads
from lens import layers_of, log
from partition import assign, kmeans
from quant_lens import fit_frame

MODELS = {"gemma4_e2b_base": ("tok_base", "tok_meta.npz", 18), "qwen3_17b": ("tok_qwen3", "tok_meta_qwen3.npz", 23)}
N_OCC = 20


def partial_spearman(x, y, covs):
    """Spearman correlation of x and y after regressing both ranks on the covariates' ranks."""
    R = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in covs])
    rx, ry = rankdata(x), rankdata(y)
    ex = rx - R @ np.linalg.lstsq(R, rx, rcond=None)[0]
    ey = ry - R @ np.linalg.lstsq(R, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    from nltk.corpus import wordnet as wn
    set_threads()
    out = {}
    for key, (prefix, meta_file, primary) in MODELS.items():
        meta = dict(np.load(CACHE / meta_file, allow_pickle=True))
        ev, vocab, freq, l1 = meta["eval"], meta["vocab"], meta["freq"], meta["eval_l1"]
        w = ev[:, 3]
        rng = np.random.default_rng(0)
        words, occ = [], []
        for c in np.unique(w):
            s = str(vocab[c])
            idx = np.flatnonzero(w == c)
            if len(idx) >= N_OCC and s.isalpha() and len(s) >= 3:
                n_syn = len(wn.synsets(s))
                if n_syn > 0:
                    words.append((c, s, n_syn))
                    occ.append(rng.choice(idx, N_OCC, replace=False))
        syn = np.array([x[2] for x in words], float)
        logf = np.log(np.array([freq[x[0]] for x in words], float) + 1)
        topics = np.array([len(np.unique(l1[o])) for o in occ], float)
        log(key, "words", len(words))
        per_layer = []
        names = layers_of(prefix)
        for li, name in enumerate(names):
            Xf, Xe = load_cache(name, "fit"), load_cache(name, "eval")
            mu, W = fit_frame(Xf)
            cells = assign((Xe - mu) @ W, kmeans((Xf - mu) @ W, 512, seed=0, nredo=1))[0]
            score = []
            for o in occ:
                _, cnt = np.unique(cells[o], return_counts=True)
                p = cnt / cnt.sum()
                score.append(float(-(p * np.log(p)).sum() / np.log(N_OCC)))
            score = np.array(score)
            r = {"layer": li, "spearman_raw": float(spearmanr(score, syn).statistic),
                 "partial_freq": partial_spearman(score, syn, [logf]),
                 "partial_freq_topics": partial_spearman(score, syn, [logf, topics])}
            if li == primary:
                boots = {"partial_freq": [], "partial_freq_topics": []}
                for _ in range(1000):
                    i = rng.integers(0, len(score), len(score))
                    boots["partial_freq"].append(partial_spearman(score[i], syn[i], [logf[i]]))
                    boots["partial_freq_topics"].append(partial_spearman(score[i], syn[i], [logf[i], topics[i]]))
                r["ci95"] = {k: np.percentile(v, [2.5, 97.5]).tolist() for k, v in boots.items()}
                order = np.argsort(-score)
                r["top30"] = [[words[i][1], round(float(score[i]), 3), int(syn[i])] for i in order[:30]]
                r["bottom30"] = [[words[i][1], round(float(score[i]), 3), int(syn[i])] for i in order[-30:]]
            per_layer.append(r)
            log(key, li, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)})
        p = per_layer[primary]
        out[key] = {"primary_layer": primary, "n_words": len(words), "layers": per_layer,
                    "W1_holds": bool(p["ci95"]["partial_freq"][0] > 0),
                    "W1_holds_with_topic_control": bool(p["ci95"]["partial_freq_topics"][0] > 0)}
    out["W1_supported"] = all(out[k]["W1_holds"] for k in MODELS)
    out["W1_supported_with_topic_control"] = all(out[k]["W1_holds_with_topic_control"] for k in MODELS)
    (RESULTS / "polysemy.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
