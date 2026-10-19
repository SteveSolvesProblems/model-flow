"""Addendum: damage from a random rotation of the residual stream at each layer's input, Gemma 4 E2B and E4B.
The stream just after the turn is far less sensitive to its axes than elsewhere. Hollow points (E2B, version 0.3):
the same rotations with every attention layer's keys and values taken from the unrotated model."""
import glob
import re

import matplotlib.pyplot as plt

from style import BG, BLUE, MUTED, ORANGE, RESULTS, load, save


def profile(patterns, key="random_kl_at_1"):
    pts = {}
    for pat in patterns:
        for f in glob.glob(str(RESULTS / pat)):
            L = int(re.search(r"_L(\d+)\.json$", f).group(1))
            pts[L] = load(f.split("/")[-1])[key]
    return sorted((L, v["mean"], v["min"], v["max"]) for L, v in pts.items())


CLEAN = [p for p in profile(["turn_undo_kv1_bf16_rand_cleankv_L*.json"], "random_cleankv_kl_at_1") if p[0] < 15]
MODELS = [("Gemma 4 E2B", profile(["turn_undo_L*.json", "turn_undo_rand_L*.json"]), (13, 14), 35, CLEAN),
          ("Gemma 4 E4B", profile(["turn_undo_e4b_rand_L*.json"]), (22, 23), 42, [])]
SHARED_FROM = {"Gemma 4 E2B": 15, "Gemma 4 E4B": 24}           # first layer that reuses earlier keys and values
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.3), sharey=True)
for ax, (name, pts, turn, nL, clean) in zip(axs, MODELS):
    ax.axvspan(turn[0] - 0.5, turn[1] + 0.5, color=ORANGE, alpha=0.15, lw=0)
    ax.text((turn[0] + turn[1]) / 2, 8.3, "turn", color=ORANGE, ha="center", fontsize=8)
    ax.axvline(SHARED_FROM[name] - 0.5, color=MUTED, lw=0.8, ls=":")
    ax.text(SHARED_FROM[name] + 0.3, 7.6, "reuses keys\nand values →", color=MUTED, fontsize=7.5, va="top")
    L = [p[0] for p in pts]
    ax.plot(L, [p[1] for p in pts], "-o", color=BLUE, ms=5)
    ax.vlines(L, [p[2] for p in pts], [p[3] for p in pts], color=BLUE, lw=2)
    if clean:
        ax.plot([p[0] for p in clean], [p[1] for p in clean], "o", mfc=BG, mec=BLUE, mew=1.4, ms=6)
        ax.text(0.8, 0.75, "hollow: keys and values\nfrom the unrotated model", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, nL)
    ax.set_ylim(0, 8.8)
    ax.set_title(name, loc="left")
    ax.set_xlabel("rotated at the input of layer")
axs[0].set_ylabel("damage to predictions (KL)")
axs[1].text(41, 0.4, "points: mean of 3 random rotations\n(bars: their range)", ha="right", fontsize=7.5, color=MUTED)
fig.tight_layout()
save(fig, "rotation_profile")
