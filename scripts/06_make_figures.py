"""Step 06: render the per-region F1 figure and metric tables.

Inputs:  data/results/probe_metrics_y{year}.parquet
         data/raw/chip_manifest_y{year}.parquet
Output:  data/results/figures/per_region_f1_y{year}.{png,pdf}
         data/results/metrics_table_y{year}.csv

Usage:
    uv run python scripts/06_make_figures.py
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

from geo_bias.eval.viz import per_region_f1_bar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("06_make_figures")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RES_DIR = ROOT / "data" / "results"
FIG_DIR = RES_DIR / "figures"


def main() -> None:
    sampling_cfg = yaml.safe_load((ROOT / "configs" / "sampling.yaml").read_text())
    target_year = sampling_cfg["labels"]["target_year"]

    metrics = pd.read_parquet(RES_DIR / f"probe_metrics_y{target_year}.parquet")
    chip_manifest = pd.read_parquet(RAW_DIR / f"chip_manifest_y{target_year}.parquet")

    png = per_region_f1_bar(metrics, chip_manifest, FIG_DIR, target_year)

    table_path = RES_DIR / f"metrics_table_y{target_year}.csv"
    metrics.round(3).to_csv(table_path, index=False)

    log.info("Wrote figure to %s", png)
    log.info("Wrote metrics CSV to %s", table_path)


if __name__ == "__main__":
    main()
