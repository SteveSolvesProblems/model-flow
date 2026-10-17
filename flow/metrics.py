"""Clustering and detection metrics. Labels are used for evaluation only."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score


def contingency(assign, y):
    ca, ai = np.unique(assign, return_inverse=True)
    cy, yi = np.unique(y, return_inverse=True)
    M = np.zeros((len(ca), len(cy)), dtype=np.int64)
    np.add.at(M, (ai, yi), 1)
    return M


def purity(assign, y):
    """Majority-label accuracy. Rises with K by construction — never compare across K."""
    return contingency(assign, y).max(axis=1).sum() / len(y)


def hungarian_acc(assign, y):
    """One-to-one matching accuracy (clusters beyond #classes count as wrong)."""
    M = contingency(assign, y)
    r, c = linear_sum_assignment(-M)
    return M[r, c].sum() / len(y)


def cluster_scores(assign, y):
    return {
        "nmi": float(normalized_mutual_info_score(y, assign)),
        "ari": float(adjusted_rand_score(y, assign)),
        "purity": float(purity(assign, y)),
        "hungarian": float(hungarian_acc(assign, y)),
        "n_clusters": int(len(np.unique(assign))),
    }


def auroc(score, target):
    target = np.asarray(target).astype(bool)
    if target.all() or (~target).all():
        return float("nan")
    return float(roc_auc_score(target, score))


def fpr_at_tpr(score, target, tpr=0.95):
    """FPR at the threshold that catches `tpr` of positives (higher score = more positive)."""
    target = np.asarray(target).astype(bool)
    thr = np.quantile(score[target], 1 - tpr)
    return float((score[~target] >= thr).mean())


def bootstrap_ci(fn, *arrays, n=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    N = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        v = fn(*[a[idx] for a in arrays])
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
