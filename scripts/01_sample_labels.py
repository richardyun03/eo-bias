"""Step 01: sample positives/negatives from Hansen GFC per region; write manifest.

Inputs:  configs/regions.yaml, configs/sampling.yaml
Output:  data/raw/sampling_manifest_y{year}.parquet

Usage:
    export EE_PROJECT=<your-gcp-project-id>
    uv run python scripts/01_sample_labels.py
"""

import logging
from pathlib import Path

import ee
import pandas as pd
import yaml

from geo_bias.data.ee_client import init_ee
from geo_bias.data.hansen import sample_region
from geo_bias.data.splits import assign_block_split
from geo_bias.utils.seeds import set_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("01_sample_labels")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw"


def main() -> None:
    regions_cfg = yaml.safe_load((ROOT / "configs" / "regions.yaml").read_text())
    sampling_cfg = yaml.safe_load((ROOT / "configs" / "sampling.yaml").read_text())

    set_all(sampling_cfg["seeds"]["sampling"])
    init_ee()

    labels = sampling_cfg["labels"]
    budget = sampling_cfg["budget"]
    geom_cfg = sampling_cfg["geometry"]

    per_region: list[pd.DataFrame] = []
    for short_key, region in regions_cfg["regions"].items():
        west, south, east, north = region["bbox"]
        geom = ee.Geometry.Rectangle([west, south, east, north])

        log.info("Sampling region %s (%s) ...", region["name"], region["short_code"])
        df = sample_region(
            region_name=region["name"],
            region_short=region["short_code"],
            geom=geom,
            asset=labels["asset"],
            target_year=labels["target_year"],
            treecover_min_pct=labels["treecover2000_min_pct"],
            n_pos=budget["positives_per_region"],
            n_neg=budget["negatives_per_region"],
            min_separation_km=geom_cfg["min_chip_separation_km"],
            seed=sampling_cfg["seeds"]["sampling"],
        )
        df = assign_block_split(
            df,
            block_grid_deg=geom_cfg["block_grid_deg"],
            test_fraction=geom_cfg["test_block_fraction"],
            seed=sampling_cfg["seeds"]["split"],
        )
        log.info("[%s] split counts: %s", region["short_code"],
                 df.groupby(["split", "label"]).size().to_dict())
        per_region.append(df)

    manifest = pd.concat(per_region, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"sampling_manifest_y{labels['target_year']}.parquet"
    manifest.to_parquet(out_path, index=False)
    log.info("Wrote %d rows to %s", len(manifest), out_path)
    print(manifest.groupby(["region_short", "split", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
