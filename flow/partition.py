"""K-means partitions and K-fined trees (bisecting refinement with pluggable split score and local projection)."""
import faiss
import numpy as np


def kmeans(X, k, seed=0, niter=50, nredo=3):
    X = np.ascontiguousarray(X, dtype=np.float32)
    km = faiss.Kmeans(X.shape[1], k, niter=niter, nredo=nredo, seed=seed, verbose=False,
                      min_points_per_centroid=1, max_points_per_centroid=100000)
    km.train(X)
    return km.centroids.copy()


def assign(X, C):
    idx = faiss.IndexFlatL2(C.shape[1])
    idx.add(np.ascontiguousarray(C, dtype=np.float32))
    D, I = idx.search(np.ascontiguousarray(X, dtype=np.float32), min(3, len(C)))
    return I[:, 0], np.sqrt(np.maximum(D, 0))


# ── local projections applied before re-clustering a cell ────────────────────

class Identity:
    def fit(self, X, seed=0):
        return self

    def __call__(self, X):
        return X


class PCAWhiten:
    """Local PCA to r dims + whitening, fit on the cell's own train points."""

    def __init__(self, r=32, eps=1e-6):
        self.r, self.eps = r, eps

    def fit(self, X, seed=0):
        self.mu = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        r = min(self.r, len(S))
        self.W = Vt[:r].T / (S[:r] / np.sqrt(max(len(X) - 1, 1)) + self.eps)
        return self

    def __call__(self, X):
        return (X - self.mu) @ self.W


PROJECTIONS = {"none": Identity, "pcaw": PCAWhiten}


class Tree:
    """Root partition (K0 centroids) whose leaves can be bisected, each split in its own projected space."""

    def __init__(self, C0):
        # node: {"proj": callable, "C": centers in projected space, "child": {j: node_id}}
        self.nodes = [{"proj": Identity(), "C": np.asarray(C0, np.float32), "child": {}}]

    def leaves(self):
        out = []
        for nid, n in enumerate(self.nodes):
            out += [(nid, j) for j in range(len(n["C"])) if j not in n["child"]]
        return out

    def route(self, X):
        """Return leaf key (node_id, j) per row as a single int code."""
        codes = np.empty(len(X), dtype=np.int64)
        self._route(X, np.arange(len(X)), 0, codes)
        return codes

    def _route(self, X, rows, nid, codes):
        n = self.nodes[nid]
        j, _ = assign(n["proj"](X[rows]), n["C"])
        for jj in np.unique(j):
            sub = rows[j == jj]
            if jj in n["child"]:
                self._route(X, sub, n["child"][jj], codes)
            else:
                codes[sub] = nid * 100000 + jj

    def split(self, X_leaf, key, proj_name="none", seed=0, k=2):
        nid, j = key
        proj = PROJECTIONS[proj_name]().fit(X_leaf, seed=seed)
        Z = proj(X_leaf).astype(np.float32)
        C = kmeans(Z, k, seed=seed)
        self.nodes.append({"proj": proj, "C": C, "child": {}})
        self.nodes[nid]["child"][j] = len(self.nodes) - 1


def reoptimize_leaves(X, tree, seed=0, niter=25):
    """Control from the prior-art review: take a tree's leaves as the initial centroids for flat Lloyd
    iterations. If this beats the tree, the hierarchy only served as an initializer."""
    codes = tree.route(X)
    init = np.stack([X[codes == c].mean(0) for c in np.unique(codes)]).astype(np.float32)
    km = faiss.Kmeans(X.shape[1], len(init), niter=niter, seed=seed, verbose=False,
                      min_points_per_centroid=1, max_points_per_centroid=100000)
    km.train(np.ascontiguousarray(X, dtype=np.float32), init_centroids=init)
    return km.centroids.copy()


def grow_tree(X, K0, n_leaves, score_fn, proj_name="none", seed=0, min_n=40):
    """Bisect the highest-scoring leaf until n_leaves. score_fn(X_leaf, seed) -> float (higher = split first)."""
    tree = Tree(kmeans(X, K0, seed=seed))
    cache = {}
    rng = np.random.default_rng(seed)
    while len(tree.leaves()) < n_leaves:
        codes = tree.route(X)
        best, best_s = None, -np.inf
        for key in tree.leaves():
            code = key[0] * 100000 + key[1]
            m = codes == code
            if m.sum() < min_n:
                continue
            if code not in cache:
                cache[code] = score_fn(X[m], seed) if score_fn != "random" else rng.random()
            if cache[code] > best_s:
                best, best_s = key, cache[code]
        if best is None:
            break
        code = best[0] * 100000 + best[1]
        tree.split(X[codes == code], best, proj_name=proj_name, seed=seed)
        cache.pop(code, None)
    return tree
