"""Per-region bar charts and metric tables for the bias writeup.

Region order in plots is data-rich → underrepresented, matching the
pretraining-bias hypothesis (Western Europe → Central Africa).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REGION_LABELS = {
    "WE": "Western\nEurope",
    "BA": "Brazilian\nAmazon",
    "SEA": "Southeast\nAsia",
    "CA": "Central\nAfrica",
}

REGION_COLORS = {
    "WE": "#1f77b4",   # blue, control / data-rich
    "BA": "#2ca02c",   # green, tropical w/ task-specific data
    "SEA": "#ff7f0e",  # orange, heterogeneous
    "CA": "#d62728",   # red, hypothesized-underrepresented
}

REGION_ORDER = ["WE", "BA", "SEA", "CA"]


def per_region_f1_bar(
    metrics: pd.DataFrame,
    chip_manifest: pd.DataFrame,
    out_dir: Path,
    target_year: int,
) -> Path:
    """Per-region F1 bar chart with pooled + random baselines and per-bar context."""
    per_region = metrics[metrics["probe"] == "per_region"].set_index("region_short")
    pooled_row = metrics[metrics["probe"] == "pooled"].iloc[0]
    valid_by_region = chip_manifest.groupby("region_short")["valid_pixel_fraction"].mean()

    f1s = [per_region.loc[r, "f1"] for r in REGION_ORDER]
    aucs = [per_region.loc[r, "roc_auc"] for r in REGION_ORDER]
    valids = [valid_by_region.loc[r] for r in REGION_ORDER]
    n_tests = [int(per_region.loc[r, "n_test"]) for r in REGION_ORDER]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(REGION_ORDER))
    bars = ax.bar(
        x, f1s,
        color=[REGION_COLORS[r] for r in REGION_ORDER],
        edgecolor="black", linewidth=0.6, width=0.65,
    )

    ax.axhline(
        pooled_row["f1"], color="gray", linestyle="--", linewidth=1.2,
        label=f"Pooled probe F1 = {pooled_row['f1']:.3f}",
    )
    ax.axhline(
        0.5, color="black", linestyle=":", alpha=0.5, linewidth=1.0,
        label="Random baseline F1 = 0.5",
    )

    for bar, f1, auc, v, n in zip(bars, f1s, aucs, valids, n_tests):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.012,
            f"F1 = {f1:.3f}\nAUC = {auc:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2, 0.03,
            f"valid={v:.2f}\nn_test={n}",
            ha="center", va="bottom", fontsize=8, color="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([REGION_LABELS[r] for r in REGION_ORDER])
    ax.set_ylabel("F1 (per-region linear probe)")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"OlmoEarth-v1-Base frozen embeddings — Hansen GFC deforestation, {target_year}\n"
        f"Single-year S2 median composite, n=200 pos + 200 neg per region",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"per_region_f1_y{target_year}.png"
    pdf = out_dir / f"per_region_f1_y{target_year}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png
