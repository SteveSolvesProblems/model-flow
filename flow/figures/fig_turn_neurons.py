"""Addendum: swaps at the turn layers change the content but not the turn (left); the previous layer's scalar sets
part of the turn (right). Gemma 4 E2B. Coordinate turn = 1 - centered cosine, as in the post's track figure."""
import matplotlib.pyplot as plt
import numpy as np

from style import BLUE, GREEN, GRID, INK, MUTED, ORANGE, load, save

T = load("turn_company_e2b.json")
C = T["conds"]
ROWS = [("clean", "no change"), ("rand1024", "1,024 random\nneurons to mean"), ("top1024", "1,024 largest\nneurons to mean"),
        ("all", "every neuron\nto its mean"), ("resample", "every neuron from\nanother text"),
        ("noise", "write replaced\nby noise")]
BARS = [("turn", "coordinate turn across layers 13-14", BLUE, lambda r: 1 - r["turn"]),
        ("company", "company kept after layer 14", ORANGE, lambda r: r["company_vs_clean"]["15"]),
        ("top1", "next-token prediction unchanged", GREEN, lambda r: r["top1"])]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [1.7, 1]})
y = np.arange(len(ROWS))[::-1]
h = 0.26
for j, (_, lab, col, f) in enumerate(BARS):
    ax.barh(y + (1 - j) * h, [f(C[k]) for k, _ in ROWS], height=h, color=col, label=lab)
ax.set_yticks(y, [lab for _, lab in ROWS], fontsize=8)
ax.set_xlim(0, 1.02)
ax.set_xlabel("fraction (1 = full turn / all kept)")
ax.set_title("Content changes, the turn does not", loc="left")
ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)

R = load("block_ablation_gemma4_e2b_l12.json")["results"]
pts = [(x, 1 - R[k]["adj_cos"][13]) for x, k in [(0.5, "scalex0.5_12"), (1.0, "clean"), (2.0, "scalex2_12"),
                                                  (1 / 0.3105, "scalar1_12"), (4.0, "scalex4_12")]]   # layer 12's scalar is 0.3105
bx.axhspan(0.3, 0.6, color=MUTED, alpha=0.12, lw=0)
bx.text(0.55, 0.45, "ordinary layers", color=MUTED, fontsize=8, va="center")
bx.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=BLUE, ms=5)
bx.annotate("as trained", (1.0, pts[1][1]), xytext=(1.15, 0.97), fontsize=8, color=INK)
bx.annotate("set to 1", (pts[3][0], pts[3][1]), xytext=(2.3, 0.66), fontsize=8, color=INK)
bx.set_xscale("log", base=2)
bx.set_xticks([0.5, 1, 2, 4], ["x0.5", "x1", "x2", "x4"])
bx.set_ylim(0, 1.02)
bx.set_xlabel("layer 12's scalar, multiplied by")
bx.set_ylabel("coordinate turn across layer 13")
bx.set_title("The previous scalar sets part of it", loc="left")
fig.tight_layout()
save(fig, "turn_neurons")
