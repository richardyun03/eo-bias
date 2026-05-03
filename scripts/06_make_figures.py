"""Step 06: render the per-region F1 bar, world heatmap, and metrics heatmap.

Inputs:  data/results/probe_metrics_t{N}_y{year}.parquet
         data/raw/chip_manifest_t{N}_y{year}.parquet
Output:  data/results/figures/{per_region_f1,world_heatmap,metrics_heatmap}_t{N}_y{year}.{png,pdf}
         data/results/metrics_table_t{N}_y{year}.csv

Usage:
    uv run python scripts/06_make_figures.py
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

from geo_bias.eval.viz import (
    metrics_heatmap,
    per_region_f1_bar,
    transfer_heatmap,
    world_heatmap,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("06_make_figures")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RES_DIR = ROOT / "data" / "results"
FIG_DIR = RES_DIR / "figures"


def main() -> None:
    sampling_cfg = yaml.safe_load((ROOT / "configs" / "sampling.yaml").read_text())
    target_year = sampling_cfg["labels"]["target_year"]
    offsets = sampling_cfg["imagery"]["temporal_offsets"]
    years = [target_year + o for o in offsets]
    n_t = len(years)
    tag = f"t{n_t}"

    if n_t == 1:
        caption = f"Single-year S2 median composite (year={years[0]})"
    else:
        caption = f"Multi-year S2 composite (years={years}, T={n_t})"

    regions_cfg = yaml.safe_load((ROOT / "configs" / "regions.yaml").read_text())
    metrics = pd.read_parquet(RES_DIR / f"probe_metrics_{tag}_y{target_year}.parquet")
    chip_manifest = pd.read_parquet(RAW_DIR / f"chip_manifest_{tag}_y{target_year}.parquet")

    bar_png = per_region_f1_bar(
        metrics, chip_manifest, FIG_DIR, target_year, tag=tag, caption=caption
    )
    map_png = world_heatmap(metrics, regions_cfg, FIG_DIR, target_year, tag=tag, caption=caption)
    grid_png = metrics_heatmap(metrics, FIG_DIR, target_year, tag=tag, caption=caption)

    transfer_path = RES_DIR / f"transfer_matrix_{tag}_y{target_year}.parquet"
    figures = [bar_png, map_png, grid_png]
    if transfer_path.exists():
        transfer_df = pd.read_parquet(transfer_path)
        figures.append(
            transfer_heatmap(transfer_df, FIG_DIR, target_year, tag=tag, caption=caption)
        )
    else:
        log.warning("Transfer matrix not found at %s — skipping that figure.", transfer_path)

    table_path = RES_DIR / f"metrics_table_{tag}_y{target_year}.csv"
    metrics.round(3).to_csv(table_path, index=False)

    for p in figures:
        log.info("Wrote %s", p)
    log.info("Wrote metrics CSV to %s", table_path)


if __name__ == "__main__":
    main()
