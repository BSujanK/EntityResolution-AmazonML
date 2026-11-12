"""Regenerate the README / report figures in docs/assets/ from the recorded results.

    python tools/make_figures.py [--out docs/assets]

All numbers are the ones recorded during the challenge (validation runs and the final test-set
profile); nothing here needs the dataset.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 160
BLUE, ORANGE, GREY, DARK, RED = "#2563eb", "#ea580c", "#9ca3af", "#1f2937", "#b91c1c"
LIGHT_BLUE = "#93b4f5"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 11, "axes.edgecolor": "#6b7280", "axes.labelcolor": DARK,
    "xtick.color": DARK, "ytick.color": DARK, "text.color": DARK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e7eb", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False,
})


def _save(fig, out_dir, name):
    path = out_dir / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {path}")


def score_progression(out_dir):
    versions = ["v2", "v3", "v4a", "v5a", "v6", "v7b", "v9", "v10a", "v10b", "v10c"]
    held_out = [0.9812, 0.9833, 0.9813, np.nan, 0.98146, 0.98397, 0.98528, 0.98528, 0.98612, 0.98637]
    public = [0.972, 0.965, 0.973, 0.973, 0.973612, 0.977, 0.979189, 0.979396, 0.980128, 0.980385]
    x = np.arange(len(versions))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x, held_out, "-o", color=BLUE, lw=2.2, ms=7, label="Held-out validation")
    ax.plot(x, public, "-s", color=ORANGE, lw=2.2, ms=7, label="Public leaderboard")
    ax.set_xticks(x, versions)
    ax.set_ylim(0.9615, 0.9895)
    ax.set_ylabel("Macro F0.5")
    ax.set_title("Held-out vs public leaderboard macro F0.5 by version", pad=14)
    ax.legend(loc="lower right", ncol=1, fontsize=11)
    ax.text(3, held_out[2] - 0.0028, "no held-out run", ha="center", va="top", fontsize=9, color=BLUE,
            style="italic")

    ax.annotate("v3: stage 2 allowed to promote\n-> overfit to train", xy=(1, public[1]), xytext=(1.55, 0.9655),
                fontsize=10, va="center", arrowprops=dict(arrowstyle="->", color=DARK, lw=1.1))
    ax.annotate("v7b: + cross-encoders", xy=(5, public[5]), xytext=(4.05, 0.9715), fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.1))
    ax.annotate("v10c: final 0.980385", xy=(9, public[9]), xytext=(8.3, 0.9765), fontsize=10, ha="center",
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=DARK, lw=1.1))
    ax.annotate("0.98637", xy=(9, held_out[9]), xytext=(9, held_out[9] + 0.0016), fontsize=10, ha="center",
                color=BLUE)
    _save(fig, out_dir, "score_progression.png")


def candidate_funnel(out_dir):
    labels = ["All S1 × S2/S3 pairs", "Key blocking (top-10 per record)", "Learned pruning (final candidate set)",
              "Final predicted matches"]
    values = [1.65e13, 94.7e6, 6_595_983, 5_772_673]
    text = ["16.5 T", "94.7 M", "6.60 M", "5.77 M"]
    colors = [GREY, LIGHT_BLUE, BLUE, ORANGE]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_xscale("log")
    ax.set_xlim(1e5, 1e15)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    for yi, v, t in zip(y, values, text):
        ax.text(v * 1.5, yi, t, va="center", ha="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Test-set pairs (log scale)")
    ax.set_title("Candidate funnel on the test set", pad=14)
    _save(fig, out_dir, "candidate_funnel.png")


def validation_ablation(out_dir):
    labels = [
        "Stage 1, global threshold",
        "Stage 1, expected-F0.5 decision",
        "+ house-number features, 1.2M queries",
        "+ stage-2 group consensus",
        "+ digit-position features, veto-only (v6)",
        "+ cross-encoder stack (v7b)",
        "+ CE ensemble, soft veto (v9)",
        "+ candidate expansion (v10b)",
        "+ India ranks 31–50 (v10c, final)",
    ]
    values = [0.9764, 0.9768, 0.9792, 0.9812, 0.98146, 0.98397, 0.98528, 0.98612, 0.98637]
    shown = ["0.9764", "0.9768", "0.9792", "0.9812", "0.98146", "0.98397", "0.98528", "0.98612", "0.98637"]
    colors = [GREY, GREY] + [BLUE] * 6 + [ORANGE]
    lo, hi, ceiling = 0.974, 0.994, 0.9926

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, [v - lo for v in values], left=lo, color=colors, height=0.68)
    ax.set_xlim(lo, hi)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    for yi, v, t in zip(y, values, shown):
        ax.text(v + 0.0002, yi, t, va="center", ha="left", fontsize=10.5)
    ax.axvline(ceiling, color=RED, ls="--", lw=1.6)
    ax.text(ceiling - 0.0002, y[0] + 0.55, "blocking ceiling 0.9926", color=RED, ha="right", va="center",
            fontsize=10.5, fontweight="bold")
    ax.set_ylim(-0.6, len(labels) - 0.1)
    ax.set_xlabel("Held-out macro F0.5 (x axis starts at 0.974)")
    ax.set_title("Held-out validation ablation", pad=14)
    _save(fig, out_dir, "validation_ablation.png")


def error_budget(out_dir):
    labels = ["Blocking misses (true link never retrieved)", "Empty-address records (name-only, ambiguous)",
              "Rejected look-alikes (true match vetoed)", "False merges (wrong link accepted)"]
    values = [0.0076, 0.0065, 0.0030, 0.0019]

    fig, ax = plt.subplots(figsize=(10, 4.9))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=[BLUE, BLUE, LIGHT_BLUE, LIGHT_BLUE], height=0.62)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 0.0092)
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values):
        ax.text(v + 0.00012, yi, f"{v:.4f}", va="center", ha="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Loss contribution (1 − macro F0.5)")
    ax.set_title("Where the remaining loss comes from (held-out, v10b)", pad=28)
    ax.text(0, 1.04, "Total remaining loss: 1 − F0.5 = 0.0147", transform=ax.transAxes, fontsize=11, color="#4b5563")
    fig.text(0.01, -0.02, "Categories overlap (empty-address records also appear in the other categories), "
             "so they sum to more than 0.0147.", fontsize=9.5, color="#4b5563", ha="left", va="top")
    _save(fig, out_dir, "error_budget.png")


def country_profile(out_dir):
    countries = ["France\n(unseen in training)", "US", "India"]
    s1_entities = [259_452, 663_106, 809_986]
    matches = [3.31, 3.37, 3.31]
    singleton = [5.6, 5.8, 5.9]
    colors = [ORANGE, BLUE, BLUE]
    ticks = [f"{c}\n{n:,} S1 entities" for c, n in zip(countries, s1_entities)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    panels = [
        (axes[0], matches, 3.46, "Predicted matches per S1 entity", "{:.2f}", (0, 4.2), "training ground truth 3.46"),
        (axes[1], singleton, 5.6, "Predicted singleton rate (%)", "{:.1f}%", (0, 8), "training ground truth 5.6%"),
    ]
    for ax, vals, ref, title, fmt, ylim, ref_label in panels:
        xs = np.arange(len(countries))
        ax.bar(xs, vals, color=colors, width=0.58)
        ax.set_xticks(xs, ticks, fontsize=10)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=12.5, pad=10)
        ax.grid(axis="x", visible=False)
        ax.axhline(ref, color=DARK, ls="--", lw=1.4, label=ref_label)
        ax.legend(loc="upper right", fontsize=10)
        for xi, v in zip(xs, vals):
            ax.text(xi, v * 0.5, fmt.format(v), ha="center", va="center", fontsize=12.5, fontweight="bold",
                    color="white")
    fig.suptitle("Test-set prediction profile by country", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_dir, "test_profile_by_country.png")


def main():
    default_out = Path(__file__).resolve().parents[1] / "docs" / "assets"
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=default_out, help="output directory (default: docs/assets)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for fn in (score_progression, candidate_funnel, validation_ablation, error_budget, country_profile):
        fn(args.out)


if __name__ == "__main__":
    main()
