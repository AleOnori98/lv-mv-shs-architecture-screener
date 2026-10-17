"""Figure for the convergence experiment: customers left off-grid once the loop settles.

Reads docs/figures/convergence/convergence_results.json (written by
scripts/convergence_experiment.py) and plots, for the MV/LV screen (Case D, the
recommended architecture), the converged number of off-grid customers against the grid
price level, one line per generation-cost curve shape.

Run:  python scripts/plot_convergence.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "figures" / "convergence" / "convergence_results.json"
OUT = ROOT / "docs" / "figures" / "convergence" / "offgrid_vs_price.png"

# Validated categorical slots 1-4 (light surface). Aqua and yellow sit below 3:1 contrast,
# so every line is also direct-labelled and carries its own marker shape.
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [
    ((0.00, 1.00), "Flat price (today)", "#2a78d6", "o"),
    ((0.15, 0.85), "Mild scale economies", "#eb6834", "s"),
    ((0.30, 0.70), "Medium scale economies", "#1baf7a", "D"),
    ((0.40, 0.60), "Strong scale economies", "#eda100", "^"),
]


def main() -> None:
    d = json.loads(DATA.read_text())
    runs = [r for r in d["runs"] if r["screen"] == "D" and r["damping"] == 1.0]
    n_total = d["meta"]["n_customers"]

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ends = {}  # end point -> labels, so lines that finish together share one direct label
    for shape, label, color, marker in SERIES:
        pts = sorted((r["lcoe_level"], r["rounds"][-1]["n_off_grid"]) for r in runs
                     if (r["fixed_share"], r["alpha"]) == shape)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=8, mec=SURFACE, mew=2,
                solid_capstyle="round", zorder=3, label=label)
        ends.setdefault((xs[-1], ys[-1]), []).append(label.replace(" scale economies", ""))
    for (x, y), labels in ends.items():
        text = (labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1])
        ax.annotate(f"{text}: {y}", (x, y), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=9, color=TEXT_MUTED)

    ax.axvline(d["meta"]["flat_lcoe"], color=TEXT_MUTED, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.text(d["meta"]["flat_lcoe"], 174, "  dataset price", ha="left", va="center",
            fontsize=8.5, color=TEXT_MUTED)

    ax.set_xlim(0.09, 0.37)
    ax.set_ylim(0, 180)
    ax.set_xticks(sorted({r["lcoe_level"] for r in runs}))
    ax.set_xticklabels([f"{x:.2f}" for x in sorted({r["lcoe_level"] for r in runs})])
    ax.set_xlabel("Grid generation price at full village size [EUR/kWh]", color=TEXT_MUTED, fontsize=9.5)
    ax.set_ylabel(f"Customers off-grid at convergence (of {n_total})", color=TEXT_MUTED, fontsize=9.5)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    fig.suptitle("Scale economies only change the answer once the grid price falls",
                 x=0.07, y=0.955, ha="left", fontsize=12.5, color=TEXT, fontweight="bold")
    ax.set_title("Mafa, MV/LV + SHS screen. Every run converged in 2-3 rounds; "
                 "no cycles, no run-away.", loc="left", fontsize=9, color=TEXT_MUTED, pad=6)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=TEXT_MUTED, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
