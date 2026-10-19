"""Stability page figures.
stability_tracks: Llama-3.2-1B on DBpedia, one panel per layer track (fixed in PREREG_stability_v2.md, "Figure").
stability_pairs:  shortfall of the stability pick and of the second-to-last layer on every model-collection pair
                  (the 11 pairs of the first test, the 6 topic and 9 rating pairs of the second)."""
import json

import matplotlib.pyplot as plt
import numpy as np

from style import BG, BLUE, GREEN, GRID, INK, MUTED, ORANGE, PURPLE, RESULTS, load, save

NAMES = {"llama32_1b": "Llama-3.2-1B", "smollm2_360m": "SmolLM2-360M", "qwen35_4b_base": "Qwen3.5-4B-Base",
         "qwen3_17b": "Qwen3-1.7B", "qwen35_2b": "Qwen3.5-2B", "gemma4_e2b": "Gemma 4 E2B", "smollm2_17b": "SmolLM2-1.7B",
         "qwen35_4b": "Qwen3.5-4B", "pythia28b": "Pythia-2.8B", "qwen35_08b": "Qwen3.5-0.8B", "gemma4_e4b": "Gemma 4 E4B"}
DATA = {"": "DBpedia", "_20ng": "20 Newsgroups", "_yelp": "Yelp"}


def label(k):
    for suf in ("_20ng", "_yelp"):
        if k.endswith(suf):
            return NAMES[k[:-len(suf)]], DATA[suf]
    return NAMES[k], DATA[""]


# ── tracks ───────────────────────────────────────────────────────────────────
rows = load("llama32_1b__stabv2.json")["rows"]
sc = load("stabv2_score.json")["pairs"]["llama32_1b"]
lay = np.array([r["layer"] for r in rows])
tracks = [("repeatability of k-means groups\n(the stability score)", [np.mean(r["stability"]) for r in rows], BLUE),
          ("intrinsic dimension (TwoNN)", [r["twonn"] for r in rows], PURPLE),
          ("topic search: neighbours\nsharing the topic (%)", [100 * r["p10"] for r in rows], ORANGE),
          ("linear probe accuracy (%)", [100 * r["probe"] for r in rows], GREEN)]
picks = [("stability pick", sc["picks"]["S10"], BLUE), ("second-to-last", sc["picks"]["second_last"], MUTED),
         ("first ID peak (FEPoID)", sc["picks"]["fepoid"], PURPLE), ("ID minimum", sc["picks"]["id_min"], INK)]
fig, axs = plt.subplots(4, 1, figsize=(7.2, 7.4), sharex=True)
for ax, (name, v, col) in zip(axs, tracks):
    v = np.array(v, float)
    ax.plot(lay[1:], v[1:], "-o", color=col, ms=3.5)
    ax.set_ylabel(name, fontsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6)
at = {}
for name, L, col in picks:                                             # picks on the same layer share one label
    at.setdefault(L, []).append((name, col))
for L, items in at.items():
    col = items[0][1]
    for ax in axs:
        ax.axvline(L, color=col, lw=1.0, ls="-" if items[0][0] == "stability pick" else "--", alpha=0.8)
    axs[0].text(L, 1.02, " = ".join(n for n, _ in items), transform=axs[0].get_xaxis_transform(), rotation=35,
                fontsize=7.5, color=col, ha="left", va="bottom")
axs[2].axvline(sc["p10"]["best_layer"], color=ORANGE, lw=6, alpha=0.15)
axs[-1].set_xlabel("layer (Llama-3.2-1B, 16 layers; layer 0, the embeddings, left out)")
axs[-1].set_xticks(lay[1:])
fig.tight_layout()
save(fig, "stability_tracks")

# ── pairs ────────────────────────────────────────────────────────────────────
pairs = []
b1 = load("robust_b1.json")
for k, v in b1.items():
    if k == "summary":
        continue
    p = 100 * np.array(v["p10"])
    pairs.append(("first test (held out)", k, p.max() - p[v["S10"]], p.max() - p[len(p) - 2]))
v2 = load("stabv2_score.json")["pairs"]
for k, v in v2.items():
    s = v["p10"]["shortfall"]
    pairs.append(("second test: ratings" if k.endswith("_yelp") else "second test: topics", k, s["S10"], s["second_last"]))
groups = ["first test (held out)", "second test: topics", "second test: ratings"]
pairs.sort(key=lambda r: (groups.index(r[0]), label(r[1])))
fig, ax = plt.subplots(figsize=(7.6, 0.27 * len(pairs) + 1.6))
y = np.arange(len(pairs))[::-1]
for yi, (g, k, s, l1) in zip(y, pairs):
    gem = k.startswith("gemma")
    if abs(s - l1) < 1e-9:                                           # the pick is the second-to-last layer
        ax.scatter(s, yi, facecolor=BG, edgecolor=INK, linewidth=1.3, s=40, zorder=3)
    else:
        ax.plot([s, l1], [yi, yi], color=GRID, lw=2.5, zorder=1)
        ax.scatter(l1, yi, color=MUTED, s=28, zorder=2)
        ax.scatter(s, yi, color=BLUE, s=28, zorder=3)
    m, d = label(k)
    ax.text(-0.45, yi, f"{m} · {d}", ha="right", va="center", fontsize=7.5, color=ORANGE if gem else INK,
            fontweight="bold" if gem else "normal")
prev = None
for yi, (g, *_ ) in zip(y, pairs):
    if g != prev:
        ax.text(9.0, yi + 0.55, g, ha="right", va="bottom", fontsize=8, color=MUTED, style="italic")
        if prev is not None:
            ax.axhline(yi + 0.5, color=GRID, lw=0.8)
        prev = g
ax.set_yticks([])
ax.set_xlim(-0.25, 9)
ax.spines["bottom"].set_bounds(0, 9)
ax.set_xlabel("points below the best layer (topic search, or star rating for Yelp)")
ax.scatter([], [], color=BLUE, label="stability pick")
ax.scatter([], [], color=MUTED, label="second-to-last layer")
ax.scatter([], [], facecolor=BG, edgecolor=INK, linewidth=1.3, s=40, label="the pick is the second-to-last layer")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.spines["left"].set_visible(False)
fig.tight_layout()
save(fig, "stability_pairs")
