"""Micro to macro (exploratory, not pre-registered): weight-level structure per layer, to compare with the residual
geometry blocks (geometry_probe.py). Weights only, no activations. For each layer i:
  writes  W_i = what the layer adds to the residual: attention output and MLP output matrices (d x ...), with any
          output-side norm gains applied (Gemma's post-attention / post-feedforward norms; Gemma's per-layer-input
          projection included)
  reads   R_i = what the layer reads from the residual: Q, K, V and MLP input matrices (... x d), with its input-norm
          gains applied
Per layer: write and read norms, effective rank (participation ratio of the d x d Gram spectrum), norm-gain vectors.
Per layer pair (i, j):
  compose   ||R_j W_i||_F / (||R_j||_F ||W_i||_F): how strongly layer j reads what layer i writes (Elhage et al. 2021
            composition), computed from d x d Grams as sqrt(sum(G_R(j) * G_W(i))); also the same with the residual
            dimensions of W_i shuffled (null that keeps norms and spectra, breaks alignment)
  write_overlap / read_overlap   overlap of the top-64 write (read) subspaces, mean squared cosine of principal angles
  gain_cos  cosine between input-norm gain vectors
Writes results/weight_structure_<name>.json.

python weight_structure.py --name gemma4_e2b --model google/gemma-4-E2B
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from safetensors import safe_open

from common import RESULTS, set_threads
from lens import log

ROOTS = [os.path.expanduser("~/.cache/huggingface/hub"), "data/hf"]
TOPK = 64


def open_model(model):
    for r in ROOTS:
        snap = glob.glob(f"{r}/models--{model.replace('/', '--')}/snapshots/*/")
        if snap:
            files = glob.glob(snap[0] + "*.safetensors")
            idx = {}
            for f in files:
                with safe_open(f, "pt") as h:
                    for k in h.keys():
                        idx[k] = f
            return idx, json.load(open(snap[0] + "config.json"))
    raise FileNotFoundError(model)


def tensor(idx, key):
    with safe_open(idx[key], "pt") as h:
        return h.get_tensor(key).float().numpy()


def layer_mats(idx, kind, i):
    """Return (reads: list of (out x d) matrices, writes: list of (d x in) matrices, input gain vector)."""
    g = lambda k: tensor(idx, k) if k in idx else None
    if kind == "gemma":
        p = f"model.language_model.layers.{i}."
        ga, gm = g(p + "input_layernorm.weight"), g(p + "pre_feedforward_layernorm.weight")
        reads = [m * ga for m in (g(p + "self_attn.q_proj.weight"), g(p + "self_attn.k_proj.weight"),
                                  g(p + "self_attn.v_proj.weight")) if m is not None]
        reads += [g(p + "mlp.gate_proj.weight") * gm, g(p + "mlp.up_proj.weight") * gm]
        if g(p + "per_layer_input_gate.weight") is not None:
            reads.append(g(p + "per_layer_input_gate.weight"))
        writes = [g(p + "post_attention_layernorm.weight")[:, None] * g(p + "self_attn.o_proj.weight"),
                  g(p + "post_feedforward_layernorm.weight")[:, None] * g(p + "mlp.down_proj.weight")]
        if g(p + "per_layer_projection.weight") is not None:
            writes.append(g(p + "post_per_layer_input_norm.weight")[:, None] * g(p + "per_layer_projection.weight"))
        return reads, writes, ga
    if kind == "qwen":
        p = f"model.layers.{i}."
        ga, gm = g(p + "input_layernorm.weight"), g(p + "post_attention_layernorm.weight")
        reads = [g(p + f"self_attn.{n}_proj.weight") * ga for n in "qkv"]
        reads += [g(p + "mlp.gate_proj.weight") * gm, g(p + "mlp.up_proj.weight") * gm]
        return reads, [g(p + "self_attn.o_proj.weight"), g(p + "mlp.down_proj.weight")], ga
    if kind == "modernbert":
        p = f"model.layers.{i}."
        ga = g(p + "attn_norm.weight")
        d = tensor(idx, p + "attn.Wo.weight").shape[0]
        ga = np.ones(d, np.float32) if ga is None else ga                 # layer 0 has no attention norm
        gm = g(p + "mlp_norm.weight")
        return ([g(p + "attn.Wqkv.weight") * ga, g(p + "mlp.Wi.weight") * gm],
                [g(p + "attn.Wo.weight"), g(p + "mlp.Wo.weight")], ga)
    raise ValueError(kind)


def read_gains(reads, ga, kind, idx, i):
    """The input-norm gain vector that was multiplied into each read matrix (ones where none was applied)."""
    one = np.ones_like(ga)
    if kind == "gemma":
        p = f"model.language_model.layers.{i}."
        gm = tensor(idx, p + "pre_feedforward_layernorm.weight")
        n_attn = len(reads) - 2 - (1 if p + "per_layer_input_gate.weight" in idx else 0)
        return [ga] * n_attn + [gm, gm] + ([one] if p + "per_layer_input_gate.weight" in idx else [])
    if kind == "qwen":
        gm = tensor(idx, f"model.layers.{i}.post_attention_layernorm.weight")
        return [ga] * 3 + [gm, gm]
    return [ga, tensor(idx, f"model.layers.{i}.mlp_norm.weight")]


def pr(ev):
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    set_threads()
    idx, cfg = open_model(a.model)
    tc = cfg.get("text_config", cfg)
    L = tc["num_hidden_layers"]
    kind = "gemma" if "gemma" in a.model.lower() else "modernbert" if "modernbert" in a.model.lower() else "qwen"
    GR, GW, UR, UW, gains, stats, URraw = [], [], [], [], [], [], []
    for i in range(L):
        reads, writes, ga = layer_mats(idx, kind, i)
        R = np.concatenate(reads, 0)
        Rraw = np.concatenate([r / gv[None, :] for r, gv in zip(reads, read_gains(reads, ga, kind, idx, i))], 0)
        W = np.concatenate(writes, 1)
        gr, gw = R.T @ R, W @ W.T                                          # d x d Grams
        er, Vr = np.linalg.eigh(gr)
        URraw.append(np.linalg.eigh(Rraw.T @ Rraw)[1][:, -TOPK:])
        ew, Vw = np.linalg.eigh(gw)
        GR.append(gr / np.trace(gr)); GW.append(gw / np.trace(gw))
        UR.append(Vr[:, -TOPK:]); UW.append(Vw[:, -TOPK:])
        gains.append(ga / np.linalg.norm(ga))
        stats.append({"write_norm": float(np.sqrt(np.trace(gw))), "read_norm": float(np.sqrt(np.trace(gr))),
                      "write_pr": pr(ew), "read_pr": pr(er), "write_top64_share": float(ew[-TOPK:].sum() / ew.sum()),
                      "attn_write_norm": float(np.linalg.norm(writes[0])), "mlp_write_norm": float(np.linalg.norm(writes[1]))})
        log(a.name, i, {k: round(v, 2) for k, v in stats[-1].items()})
    d = GR[0].shape[0]
    perm = np.random.default_rng(0).permutation(d)
    comp, comp_null, wov, rov, gcos, rraw = (np.zeros((L, L)) for _ in range(6))
    for i in range(L):
        Wp = GW[i][perm][:, perm]
        for j in range(L):
            comp[i, j] = np.sqrt(max((GR[j] * GW[i]).sum(), 0))            # traces normalized to 1 above
            comp_null[i, j] = np.sqrt(max((GR[j] * Wp).sum(), 0))
            wov[i, j] = float((UW[i].T @ UW[j]) .__pow__(2).sum() / TOPK)
            rov[i, j] = float((UR[i].T @ UR[j]).__pow__(2).sum() / TOPK)
            gcos[i, j] = float(gains[i] @ gains[j])
            rraw[i, j] = float((URraw[i].T @ URraw[j]).__pow__(2).sum() / TOPK)
    out = {"name": a.name, "model": a.model, "kind": kind, "n_layers": L, "d": d, "topk": TOPK,
           "layer_types": tc.get("layer_types"), "global_attn_every_n_layers": tc.get("global_attn_every_n_layers"),
           "num_kv_shared_layers": tc.get("num_kv_shared_layers"), "per_layer": stats,
           "compose": comp.tolist(), "compose_null": comp_null.tolist(), "write_overlap": wov.tolist(),
           "read_overlap": rov.tolist(), "read_overlap_nogain": rraw.tolist(), "gain_cos": gcos.tolist(), "chance_overlap": TOPK / d}
    if kind == "gemma":
        out["layer_scalar"] = [float(tensor(idx, f"model.language_model.layers.{i}.layer_scalar").item()) for i in range(L)]
    (RESULTS / f"weight_structure_{a.name}.json").write_text(json.dumps(out))
    log("saved")


if __name__ == "__main__":
    main()
