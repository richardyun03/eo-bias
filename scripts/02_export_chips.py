"""Step 02: export Sentinel-2 chips for each manifest entry.

Inputs:  data/raw/sampling_manifest_y{year}.parquet
         configs/sampling.yaml, configs/models.yaml
Output:  data/chips/{region_short}/{chip_id}.npz   (one chip per file)
         data/raw/chip_manifest_y{year}.parquet    (sampling manifest + chip_path
                                                    + valid_pixel_fraction)
         data/raw/chip_export_failures.csv         (only if any chips fail)

Usage:
    export EE_PROJECT=<your-gcp-project-id>
    uv run python scripts/02_export_chips.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from geo_bias.data.ee_client import init_ee
from geo_bias.data.sentinel import DEFAULT_VALUE, export_chip
from geo_bias.utils.seeds import set_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("02_export_chips")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CHIPS_DIR = ROOT / "data" / "chips"


def main() -> None:
    sampling_cfg = yaml.safe_load((ROOT / "configs" / "sampling.yaml").read_text())
    models_cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())

    set_all(sampling_cfg["seeds"]["sampling"])
    init_ee()

    target_year = sampling_cfg["labels"]["target_year"]
    out_size = models_cfg["models"]["olmoearth_v1_base"]["expected_size"]
    pixel_size_m = sampling_cfg["imagery"]["pixel_size_m"]

    manifest_path = RAW_DIR / f"sampling_manifest_y{target_year}.parquet"
    manifest = pd.read_parquet(manifest_path)
    log.info("Loaded %d chips from %s", len(manifest), manifest_path)

    rows: list[dict] = []
    failures: list[dict] = []

    for row in tqdm(list(manifest.itertuples(index=False)), desc="exporting chips"):
        chip_dir = CHIPS_DIR / row.region_short
        chip_dir.mkdir(parents=True, exist_ok=True)
        chip_path = chip_dir / f"{row.chip_id}.npz"
        rel_path = str(chip_path.relative_to(ROOT))

        if chip_path.exists():
            with np.load(chip_path) as data:
                chip = data["image"]
            valid_fraction = float((chip[:, :, 0] != DEFAULT_VALUE).mean())
            rows.append(
                {"chip_id": row.chip_id, "chip_path": rel_path,
                 "valid_pixel_fraction": valid_fraction}
            )
            continue

        try:
            chip, valid_fraction = export_chip(
                lon=row.lon, lat=row.lat, year=int(row.year),
                out_size=out_size, scale_m=pixel_size_m,
            )
            np.savez_compressed(chip_path, image=chip)
            rows.append(
                {"chip_id": row.chip_id, "chip_path": rel_path,
                 "valid_pixel_fraction": valid_fraction}
            )
        except Exception as e:  # noqa: BLE001 - one bad chip shouldn't kill the run
            log.warning("[%s] export failed: %s", row.chip_id, e)
            failures.append({"chip_id": row.chip_id, "error": str(e)})

    chip_df = pd.DataFrame(rows, columns=["chip_id", "chip_path", "valid_pixel_fraction"])
    chip_manifest = manifest.merge(chip_df, on="chip_id", how="left")
    out_path = RAW_DIR / f"chip_manifest_y{target_year}.parquet"
    chip_manifest.to_parquet(out_path, index=False)
    log.info("Wrote chip manifest with %d successful exports to %s",
             chip_df["chip_id"].nunique(), out_path)

    if failures:
        fail_path = RAW_DIR / "chip_export_failures.csv"
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        log.warning("%d chips failed; details in %s", len(failures), fail_path)

    print(
        chip_manifest.groupby("region_short")["valid_pixel_fraction"]
        .agg(["count", "mean", "min"])
        .round(3)
    )


if __name__ == "__main__":
    main()
