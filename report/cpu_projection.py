"""Leaderboard trajectory + CPU-only projection (static PNG)."""
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for f in ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"]:
    fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10})

SURFACE, INK, INK2, MUTED, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#2a78d6"

actual = [("v2", 0.972), ("v3", 0.965), ("v4a", 0.973), ("v5a", 0.973), ("v6", 0.9736),
          ("v7b", 0.977), ("v9", 0.97919), ("v9 France\nprobe", 0.978)]
proj = [("v10a\nFrance\nstrict", 0.9795, 0.9790, 0.9802),
        ("v10b\ncandidate\nexpansion", 0.9805, 0.9795, 0.9815),
        ("best\nCPU-only\nfinish", 0.981, 0.980, 0.982)]
SUBGOAL, BENCH = 0.9887, 0.990748

fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=150)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
xa = list(range(len(actual)))
xp = list(range(len(actual), len(actual) + len(proj)))

# actual trajectory (best-so-far line through the submissions)
ax.plot(xa, [s for _, s in actual], color=BLUE, lw=2, zorder=3, label="Leaderboard score (submitted)")
ax.scatter(xa, [s for _, s in actual], s=46, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=4)
# projection: dashed continuation from v9, point + range whisker
ax.plot([xa[6]] + xp, [actual[6][1]] + [p[1] for p in proj], color=BLUE, lw=2, ls=(0, (4, 3)), alpha=0.8, zorder=3,
        label="CPU-only projection (range)")
for x, (_, c, lo, hi) in zip(xp, proj):
    ax.plot([x, x], [lo, hi], color=BLUE, lw=2, alpha=0.55, solid_capstyle="round", zorder=2)
    ax.scatter([x], [c], s=46, facecolor=SURFACE, edgecolor=BLUE, linewidth=2, zorder=4)

# targets: ink-coloured reference lines with direct labels
for y, txt in [(BENCH, f"Benchmark (top of leaderboard)  {BENCH:.4f}"), (SUBGOAL, f"Sub-goal  {SUBGOAL:.4f}  (v9 + 0.0095)")]:
    ax.axhline(y, color=INK2, lw=1.2, ls=(0, (6, 4)), zorder=1)
    ax.text(-0.35, y + 0.00045, txt, color=INK2, fontsize=9.5, va="bottom")

# selective direct labels on key points
labels = {0: "0.972", 1: "0.965 (stage 2 promoted twins)", 5: "0.977", 6: "0.9792 best", 7: "0.978"}
for i, t in labels.items():
    dy = -0.0016 if i in (1, 7) else 0.0009
    ax.text(xa[i], actual[i][1] + dy, t, ha="center", fontsize=8.8, color=INK2)
for x, (_, c, lo, hi) in zip(xp, proj):
    ax.text(x, lo - 0.0005, f"~{c:.4f}", fontsize=8.8, color=INK2, ha="center", va="top")

# the gap
ax.annotate("", xy=(xp[-1] + 0.45, SUBGOAL), xytext=(xp[-1] + 0.45, proj[-1][1]),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
ax.text(xp[-1] + 0.55, (SUBGOAL + proj[-1][1]) / 2, "gap ≈ 0.008\nneeds GPU\n(retrieval +\nlarge model)",
        fontsize=9, color=INK, va="center")

ax.set_xticks(xa + xp, [n for n, _ in actual] + [p[0] for p in proj], fontsize=8.8, color=INK2)
ax.axvspan(len(actual) - 0.5, xp[-1] + 1.4, color="#f0efeb", zorder=0)
ax.text(len(actual) - 0.4, 0.9868, "projected" + chr(10) + "(CPU only, ~13 h left)", fontsize=8.8, color=MUTED, va="top")
ax.set_xlim(-0.5, xp[-1] + 1.4)
ax.set_ylim(0.963, 0.9925)
ax.set_ylabel("Macro F0.5 (public leaderboard)", color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, length=0)
ax.set_title("Where we end up on CPU: about 0.980–0.982, short of the 0.9887 sub-goal", loc="left",
             fontsize=13, color=INK, fontweight="bold", pad=14)
ax.legend(loc="lower left", bbox_to_anchor=(0.36, 0.02), frameon=False, fontsize=9, labelcolor=INK2)
fig.tight_layout()
fig.savefig(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "docs" / "assets" / "cpu_projection.png"), facecolor=SURFACE)
print("saved")
