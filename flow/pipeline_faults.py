"""research/PREREG_pipeline_faults.md: does a bin / neighbour fingerprint catch pipeline faults?

  extract MODEL   reference run R, then every run in RUNS: mean-pooled residual after every block (block i = hidden
                  state i + 1) for 2,000 DBpedia test articles; next-token KL vs R and NLL on the article tokens
  score  MODEL    per block R vs run: kNN10 overlap and bins NMI (K = 64, whiten-64 frame fit on R); D, PF-a/b/c

Inputs are built from the article's own token ids (no special tokens), so every run predicts the same article tokens
and KL / NLL are compared on the article tokens present in both inputs. Asserted equal to the tokenizer's own output
for the reference. KL is computed against R's top-128 next-token probabilities plus one bucket for the rest (KL of the
coarsened distributions, a lower bound on the full KL; full R distributions would need ~30 GB per model).
States go to $PFD (default data/pfaults/<model>); results to results/pipeline_faults_<model>.json.
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from common import RESULTS, set_threads

MODELS = {"llama32_1b": ("meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct"),
          "gemma3_270m": ("google/gemma-3-270m", "google/gemma-3-270m-it"),
          "tinyllama11": ("TinyLlama/TinyLlama_v1.1", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")}
N, CAP, KNN, K, TOPK, BATCH = int(os.environ.get("PF_N", 2000)), 64, 10, 64, 128, 32
HARMLESS, GRADED = ["H1_batch8", "H2_cpu", "H3_shuffled"], ["G1_bf16", "G2_rtn8", "G3_rtn6", "G4_rtn4"]
FAULTS = ["F1_no_bos", "F2_chat_template", "F3_cap16", "F4_pad_in_mean", "F5_rtn2"]
RUNS = HARMLESS + GRADED + FAULTS


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def out_dir(key):
    d = Path(os.environ.get("PFD", "data/pfaults")) / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def rtn_(m, bits, group=32):
    """Round every linear weight inside the blocks to `bits` (symmetric, per group of 32 along the input dim)."""
    import torch
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for name, mod in m.named_modules():
            if isinstance(mod, torch.nn.Linear) and ".layers." in name:
                W = mod.weight
                o, i = W.shape
                assert i % group == 0, (name, W.shape)
                g = W.float().reshape(o, i // group, group)
                s = g.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
                W.copy_((torch.round(g / s).clamp(-qmax - 1, qmax) * s).reshape(o, i).to(W.dtype))


class Pipe:
    def __init__(self, key):
        from transformers import AutoTokenizer
        self.key, (self.name, self.inst) = key, MODELS[key]
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.bos = self.tok.bos_token_id
        assert self.bos is not None and self.tok("x")["input_ids"][0] == self.bos, "tokenizer adds no start token"
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        from extract_hf import dbpedia
        d, _ = dbpedia(24000, 6000)
        self.texts = d["test"][0][:N]
        self.art = [self.tok(t, add_special_tokens=False)["input_ids"][:CAP + 16] for t in self.texts]
        for t, a in zip(self.texts[:50], self.art[:50]):          # built reference input == tokenizer's own
            assert self.tok(t, truncation=True, max_length=CAP)["input_ids"] == [self.bos] + a[:CAP - 1]
        it = AutoTokenizer.from_pretrained(self.inst)
        mark = "XQXQXQ"
        kw = {"date_string": "01 Jan 2026"} if "llama-3" in self.inst.lower() else {}
        s = it.apply_chat_template([{"role": "user", "content": mark}], tokenize=False, add_generation_prompt=True, **kw)
        pre, suf = s.split(mark)
        pre_ids = self.tok(pre, add_special_tokens=False)["input_ids"]
        if pre_ids[:1] != [self.bos]:                  # template without a start token: the tokenizer adds it
            pre_ids = [self.bos] + pre_ids
        self.tpl = (pre_ids, self.tok(suf, add_special_tokens=False)["input_ids"])
        self.tpl_text = s

    def inputs(self, run, a):
        """(input ids, offset of article token 0 in the input)."""
        if run == "F1_no_bos":
            return a[:CAP], 0
        if run == "F2_chat_template":
            pre, suf = self.tpl
            return (pre + a + suf)[:CAP], len(pre)
        if run == "F3_cap16":
            return [self.bos] + a[:15], 1
        return [self.bos] + a[:CAP - 1], 1

    def load(self, dtype, dev):
        import torch
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(self.name, dtype=dtype, attn_implementation="eager")
        return m.to(dev).eval()

    def forward(self, m, dev, run, idx, ref=None):
        """pooled states (len(idx), L, d), per-doc NLL (mean over predicted article tokens), KL vs ref, and (ref only) top-k."""
        import torch
        seqs = [self.inputs(run, self.art[i]) for i in idx]
        T = max(len(s) for s, _ in seqs)
        ids = torch.full((len(idx), T), self.pad, dtype=torch.long)
        mask = torch.zeros((len(idx), T), dtype=torch.long)
        for r, (s, _) in enumerate(seqs):
            ids[r, :len(s)] = torch.tensor(s)
            mask[r, :len(s)] = 1
        with torch.no_grad():
            o = m(input_ids=ids.to(dev), attention_mask=mask.to(dev), output_hidden_states=True, use_cache=False)
            mk = (torch.ones_like(mask) if run == "F4_pad_in_mean" else mask).to(dev).unsqueeze(-1).float()
            H = torch.stack([(h.float() * mk).sum(1) / mk.sum(1) for h in o.hidden_states[1:]], 1).cpu().numpy()
            lp = torch.log_softmax(o.logits.float(), -1)
        nll, kl, tops = [], [], []
        for r, (i, (s, off)) in enumerate(zip(idx, seqs)):
            a = self.art[i]
            js = [j for j in range(1, len(a)) if off + j < len(s)]     # article token j present (and predicted) in input
            pos = torch.tensor([off + j - 1 for j in js], device=dev)
            tgt = torch.tensor([a[j] for j in js], device=dev)
            L = lp[r, pos]                                              # (n, V)
            nll.append((js, (-L.gather(1, tgt[:, None])[:, 0]).cpu().numpy()))
            if ref is None:
                p, v = L.exp().topk(TOPK, -1)
                tops.append((js, p.cpu().numpy().astype(np.float32), v.cpu().numpy().astype(np.int32)))
            else:
                rj, rp, rv = ref[i]
                sj = set(js)
                both = [k for k, j in enumerate(rj) if j in sj]
                where = {j: k for k, j in enumerate(js)}
                rows = torch.tensor([where[rj[k]] for k in both], device=dev)
                P = torch.tensor(rp[both], device=dev)
                V = torch.tensor(rv[both], device=dev).long()
                Q = L[rows].exp().gather(1, V)
                pt, qt = (1 - P.sum(1)).clamp_min(1e-12), (1 - Q.sum(1)).clamp_min(1e-12)
                k = (P * (P.clamp_min(1e-12).log() - Q.clamp_min(1e-12).log())).sum(1) + pt * (pt.log() - qt.log())
                kl.append((np.array([rj[k_] for k_ in both]), k.cpu().numpy()))
        return H, nll, kl, tops


def extract(key):
    import torch
    P, D = Pipe(key), out_dir(key)
    gpu = "cuda" if torch.cuda.is_available() else "cpu"
    log(key, "template:", repr(P.tpl_text[:200]))
    order = np.arange(N)
    if not (D / "ref.pkl").exists():
        m = P.load(torch.float32, gpu)
        Hs, nll, tops = [], [], []
        for s in range(0, N, BATCH):
            H, n, _, t = P.forward(m, gpu, "R", order[s:s + BATCH])
            Hs.append(H); nll += n; tops += t
        X = np.concatenate(Hs)
        assert np.isfinite(X).all()
        np.save(D / "R.npy", X)
        with open(D / "ref.pkl", "wb") as f:
            pickle.dump({"nll": nll, "tops": tops}, f)
        log("R", X.shape)
        del m
    with open(D / "ref.pkl", "rb") as f:
        z = pickle.load(f)
    ref, ref_nll = z["tops"], z["nll"]
    for run in RUNS:
        if (D / f"{run}.npy").exists():
            continue
        dev, dtype, bits = gpu, torch.float32, None
        if run == "H2_cpu":
            dev = "cpu"
        if run == "G1_bf16":
            dtype = torch.bfloat16
        if run.startswith(("G2", "G3", "G4", "F5")):
            bits = int(run.split("rtn")[1])
        m = P.load(dtype, dev)
        if bits:
            rtn_(m, bits)
        bs = 8 if run == "H1_batch8" else BATCH
        perm = np.random.default_rng(1).permutation(N) if run == "H3_shuffled" else order
        X = np.zeros((N,) + np.load(D / "R.npy", mmap_mode="r").shape[1:], np.float32)
        nll, kl = [None] * N, [None] * N
        t0 = time.time()
        for s in range(0, N, bs):
            idx = perm[s:s + bs]
            H, n, k, _ = P.forward(m, dev, run, idx, ref)
            X[idx] = H
            for r, i in enumerate(idx):
                nll[i], kl[i] = n[r], k[r]
        assert np.isfinite(X).all(), run
        # per-doc mean over the article tokens predicted in both R and the run
        d_nll, mkl = [], []
        for i in range(N):
            rj, rn = ref_nll[i]
            j, n = nll[i]
            common = sorted(set(rj) & set(j))
            ri, ji = {x: q for q, x in enumerate(rj)}, {x: q for q, x in enumerate(j)}
            d_nll.append(float(np.mean([n[ji[c]] - rn[ri[c]] for c in common])))
            mkl.append(float(np.mean(kl[i][1])))
        np.save(D / f"{run}.npy", X)
        (D / f"{run}.json").write_text(json.dumps({"dnll": d_nll, "kl": mkl, "sec": time.time() - t0, "dtype": str(dtype),
                                                     "device": dev, "batch": bs, "bits": bits}))
        log(run, "KL", round(float(np.mean(mkl)), 5), "dNLL", round(float(np.mean(d_nll)), 5), f"{time.time() - t0:.0f}s")
        del m
        if gpu == "cuda":
            torch.cuda.empty_cache()



KNN = 10


def knn_lists(X):
    U = X - X.mean(0)
    U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
    S = U @ U.T
    np.fill_diagonal(S, -np.inf)
    return np.argpartition(-S, KNN, axis=1)[:, :KNN]


def layer_overlap(A, B):
    """per-item kNN10 overlap between two state sets (N, d)."""
    a, b = knn_lists(A), knn_lists(B)
    return np.array([len(set(x) & set(y)) / KNN for x, y in zip(a, b)])


def score(key):
    from scipy.stats import spearmanr
    from sklearn.metrics import normalized_mutual_info_score as nmi
    from quant_lens import fit_frame
    from partition import assign, kmeans
    D = out_dir(key)
    R = np.load(D / "R.npy")
    frames = []
    for l in range(R.shape[1]):
        mu, W = fit_frame(R[:, l])
        C = kmeans((R[:, l] - mu) @ W, K, seed=0, nredo=1)
        frames.append((mu, W, C, assign((R[:, l] - mu) @ W, C)[0]))
    res = {"model": MODELS[key][0], "runs": {}}
    for run in RUNS:
        if not (D / f"{run}.npy").exists():
            continue
        X = np.load(D / f"{run}.npy")
        meta = json.loads((D / f"{run}.json").read_text())
        knn = [float(layer_overlap(R[:, l], X[:, l]).mean()) for l in range(R.shape[1])]
        bins = [float(nmi(a, assign((X[:, l] - mu) @ W, C)[0])) for l, (mu, W, C, a) in enumerate(frames)]
        res["runs"][run] = {"knn_by_block": knn, "bins_by_block": bins, "D": 1 - min(knn), "D_bins": 1 - min(bins),
                            "min_block": int(np.argmin(knn)), "KL": float(np.mean(meta["kl"])),
                            "dNLL": float(np.mean(meta["dnll"])), **{k: meta[k] for k in ("device", "batch", "bits")}}
        print(run, {k: round(res["runs"][run][k], 4) for k in ("D", "D_bins", "KL", "dNLL")}, "min block",
              res["runs"][run]["min_block"], flush=True)
    r = res["runs"]
    hD, hN = max(r[h]["D"] for h in HARMLESS), max(abs(r[h]["dNLL"]) for h in HARMLESS)
    res["PFa"] = {"max_harmless_D": hD, "faults_D": {f: r[f]["D"] for f in FAULTS},
                  "supported": all(r[f]["D"] > 2 * hD for f in FAULTS)}
    g = HARMLESS + GRADED
    rho = spearmanr([r[x]["D"] for x in g], [r[x]["KL"] for x in g]).statistic
    res["PFb"] = {"spearman": float(rho), "supported": bool(rho >= 0.8)}
    hidden = [f for f in FAULTS if abs(r[f]["dNLL"]) < 2 * hN and r[f]["D"] > 2 * hD]
    res["PFc"] = {"max_harmless_abs_dNLL": hN, "faults_missed_by_dNLL_caught_by_D": hidden, "supported": bool(hidden)}
    (RESULTS / f"pipeline_faults_{key}.json").write_text(json.dumps(res))
    print(json.dumps({k: res[k] for k in ("PFa", "PFb", "PFc")}, indent=1))


if __name__ == "__main__":
    set_threads()
    {"extract": extract, "score": score}[sys.argv[1]](sys.argv[2])
