"""Per-region bar charts, world heatmap, and metrics heatmap.

Region order is data-rich → underrepresented (WE → CA), matching the
OSM-density-driven pretraining-bias hypothesis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches

REGION_LABELS = {
    "WE": "Western\nEurope",
    "BA": "Brazilian\nAmazon",
    "SEA": "Southeast\nAsia",
    "CA": "Central\nAfrica",
}

REGION_COLORS = {
    "WE": "#1f77b4",
    "BA": "#2ca02c",
    "SEA": "#ff7f0e",
    "CA": "#d62728",
}

REGION_ORDER = ["WE", "BA", "SEA", "CA"]


def per_region_f1_bar(
    metrics: pd.DataFrame,
    chip_manifest: pd.DataFrame,
    out_dir: Path,
    target_year: int,
    tag: str = "t1",
    caption: str = "",
) -> Path:
    """Per-region F1 bar chart with pooled + random baselines."""
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

    if {"f1_lo", "f1_hi"}.issubset(per_region.columns):
        lo = [per_region.loc[r, "f1"] - per_region.loc[r, "f1_lo"] for r in REGION_ORDER]
        hi = [per_region.loc[r, "f1_hi"] - per_region.loc[r, "f1"] for r in REGION_ORDER]
        ax.errorbar(
            x, f1s, yerr=np.array([lo, hi]),
            fmt="none", ecolor="black", capsize=4, linewidth=1.2, zorder=4,
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
    sub = caption or "n=200 pos + 200 neg per region"
    ax.set_title(
        f"OlmoEarth-v1-Base frozen embeddings — Hansen GFC deforestation, {target_year}\n{sub}",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"per_region_f1_{tag}_y{target_year}.png"
    fig.savefig(png, dpi=300)
    fig.savefig(out_dir / f"per_region_f1_{tag}_y{target_year}.pdf")
    plt.close(fig)
    return png


def world_heatmap(
    metrics: pd.DataFrame,
    regions_cfg: dict,
    out_dir: Path,
    target_year: int,
    tag: str = "t1",
    caption: str = "",
) -> Path:
    """World map with each region's bbox colored by F1."""
    per_region = metrics[metrics["probe"] == "per_region"].set_index("region_short")
    f1_vals = [per_region.loc[r, "f1"] for r in REGION_ORDER]
    vmin = min(0.4, min(f1_vals) - 0.02)
    vmax = max(0.7, max(f1_vals) + 0.02)

    cmap = plt.get_cmap("RdYlGn")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(13, 6.5))

    # Light-gray "graticule" background.
    ax.set_facecolor("#f0f0f0")
    for lat in range(-60, 81, 30):
        ax.axhline(lat, color="white", linewidth=0.8, zorder=1)
    for lon in range(-180, 181, 30):
        ax.axvline(lon, color="white", linewidth=0.8, zorder=1)

    for region in regions_cfg["regions"].values():
        bb = region["bbox"]
        code = region["short_code"]
        f1 = per_region.loc[code, "f1"]
        color = cmap(norm(f1))
        rect = patches.Rectangle(
            (bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1],
            facecolor=color, edgecolor="black", linewidth=2, alpha=0.95, zorder=3,
        )
        ax.add_patch(rect)
        cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        ax.annotate(
            f"{code}\nF1 = {f1:.3f}",
            xy=(cx, cy), xytext=(cx, cy),
            ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", pad=3, boxstyle="round,pad=0.3"),
            zorder=4,
        )

    # Continent text labels for orientation (no shapefiles needed).
    for label, lon, lat in [
        ("North\nAmerica", -100, 45),
        ("South\nAmerica", -60, -25),
        ("Europe", 15, 55),
        ("Africa", 20, 5),
        ("Asia", 90, 40),
        ("Australia", 135, -25),
    ]:
        ax.text(lon, lat, label, ha="center", va="center",
                fontsize=11, color="#888", style="italic", zorder=2)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    sub = f"\n{caption}" if caption else ""
    ax.set_title(
        f"Geographic F1 heatmap — OlmoEarth-v1-Base frozen embeddings, deforestation {target_year}{sub}",
        fontsize=12,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="F1", shrink=0.6, pad=0.02)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"world_heatmap_{tag}_y{target_year}.png"
    fig.savefig(png, dpi=300)
    fig.savefig(out_dir / f"world_heatmap_{tag}_y{target_year}.pdf")
    plt.close(fig)
    return png


def transfer_heatmap(
    transfer_df: pd.DataFrame,
    out_dir: Path,
    target_year: int,
    tag: str = "t1",
    caption: str = "",
) -> Path:
    """Cross-region transfer F1 grid — train_region rows, test_region cols.

    Diagonal cells are equivalent to the per-region probes. Off-diagonal
    cells show transfer: train on A, evaluate on B's test split.
    """
    grid = transfer_df.pivot(index="train_region", columns="test_region", values="f1")
    train_order = [r for r in [*REGION_ORDER, "GLOBAL"] if r in grid.index]
    test_order = [r for r in REGION_ORDER if r in grid.columns]
    grid = grid.loc[train_order, test_order]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(
        grid, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.4, vmax=0.75, linewidths=0.6, linecolor="white",
        cbar_kws={"label": "F1"}, ax=ax,
    )

    # Outline the diagonal (train_region == test_region) cells.
    for i, r in enumerate(train_order):
        if r in test_order:
            j = test_order.index(r)
            ax.add_patch(
                plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=2.2)
            )

    ax.set_xlabel("Test region")
    ax.set_ylabel("Train region")
    ax.set_xticklabels([REGION_LABELS[r].replace("\n", " ") for r in test_order], rotation=0)
    ax.set_yticklabels(
        [r if r == "GLOBAL" else REGION_LABELS[r].replace("\n", " ") for r in train_order],
        rotation=0,
    )
    sub = f"\n{caption}" if caption else ""
    ax.set_title(f"Cross-region transfer F1 — deforestation {target_year}{sub}", fontsize=11)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"transfer_heatmap_{tag}_y{target_year}.png"
    fig.savefig(png, dpi=300)
    fig.savefig(out_dir / f"transfer_heatmap_{tag}_y{target_year}.pdf")
    plt.close(fig)
    return png


def metrics_heatmap(
    metrics: pd.DataFrame,
    out_dir: Path,
    target_year: int,
    tag: str = "t1",
    caption: str = "",
) -> Path:
    """Region × metric heatmap, color-encoded by metric value."""
    per_region = metrics[metrics["probe"] == "per_region"].set_index("region_short")
    cols = ["f1", "precision", "recall", "roc_auc", "pr_auc"]
    grid = per_region.loc[REGION_ORDER, cols]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    sns.heatmap(
        grid, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.4, vmax=0.75, linewidths=0.6, linecolor="white",
        cbar_kws={"label": "metric value"}, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(["F1", "Precision", "Recall", "ROC-AUC", "PR-AUC"], rotation=0)
    ax.set_yticklabels(
        [REGION_LABELS[r].replace("\n", " ") for r in REGION_ORDER], rotation=0
    )
    sub = f"\n{caption}" if caption else ""
    ax.set_title(f"Per-region probe metrics — deforestation {target_year}{sub}", fontsize=11)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"metrics_heatmap_{tag}_y{target_year}.png"
    fig.savefig(png, dpi=300)
    fig.savefig(out_dir / f"metrics_heatmap_{tag}_y{target_year}.pdf")
    plt.close(fig)
    return png
