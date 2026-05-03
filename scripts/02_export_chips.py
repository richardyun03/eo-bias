"""Step 02: export Sentinel-2 chips for each manifest entry.

Uses configs/sampling.yaml's `imagery.temporal_offsets` to decide how many
composites per chip:
  - [0]      → single-year median (T=1, legacy v1)
  - [-1, 1]  → pre/post composite pair (T=2)

Inputs:  data/raw/sampling_manifest_y{year}.parquet
Output:  data/chips_t{N}/{region_short}/{chip_id}.npz   (N = len(temporal_offsets))
         data/raw/chip_manifest_t{N}_y{year}.parquet
         data/raw/chip_export_failures_t{N}_y{year}.csv  (only on failure)

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
from geo_bias.data.sentinel import DEFAULT_VALUE, export_chip_temporal
from geo_bias.utils.seeds import set_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("02_export_chips")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def main() -> None:
    sampling_cfg = yaml.safe_load((ROOT / "configs" / "sampling.yaml").read_text())
    models_cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())

    set_all(sampling_cfg["seeds"]["sampling"])
    init_ee()

    target_year = sampling_cfg["labels"]["target_year"]
    offsets = sampling_cfg["imagery"]["temporal_offsets"]
    years = [target_year + o for o in offsets]
    n_t = len(years)
    tag = f"t{n_t}"

    out_size = models_cfg["models"]["olmoearth_v1_base"]["expected_size"]
    pixel_size_m = sampling_cfg["imagery"]["pixel_size_m"]
    chips_dir = ROOT / "data" / f"chips_{tag}"

    log.info("Pipeline tag: %s | composite years: %s", tag, years)

    sampling_path = RAW_DIR / f"sampling_manifest_y{target_year}.parquet"
    manifest = pd.read_parquet(sampling_path)
    log.info("Loaded %d chip rows from %s", len(manifest), sampling_path)

    rows: list[dict] = []
    failures: list[dict] = []

    for row in tqdm(list(manifest.itertuples(index=False)), desc=f"exporting chips ({tag})"):
        chip_dir = chips_dir / row.region_short
        chip_dir.mkdir(parents=True, exist_ok=True)
        chip_path = chip_dir / f"{row.chip_id}.npz"
        rel_path = str(chip_path.relative_to(ROOT))

        if chip_path.exists():
            with np.load(chip_path) as data:
                vfs = data["valid_fractions"].tolist() if "valid_fractions" in data.files else None
                if vfs is None:
                    chip = data["image"]
                    vfs = [float((chip[:, :, t, 0] != DEFAULT_VALUE).mean()) for t in range(n_t)]
            row_out = {"chip_id": row.chip_id, "chip_path": rel_path,
                       "valid_pixel_fraction": float(min(vfs))}
            for i, v in enumerate(vfs):
                row_out[f"valid_pixel_fraction_t{i}"] = float(v)
            rows.append(row_out)
            continue

        try:
            chip, vfs = export_chip_temporal(
                lon=row.lon, lat=row.lat, years=years,
                out_size=out_size, scale_m=pixel_size_m,
            )
            np.savez_compressed(chip_path, image=chip, valid_fractions=np.array(vfs))
            row_out = {"chip_id": row.chip_id, "chip_path": rel_path,
                       "valid_pixel_fraction": float(min(vfs))}
            for i, v in enumerate(vfs):
                row_out[f"valid_pixel_fraction_t{i}"] = float(v)
            rows.append(row_out)
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] export failed: %s", row.chip_id, e)
            failures.append({"chip_id": row.chip_id, "error": str(e)})

    cols = ["chip_id", "chip_path", "valid_pixel_fraction"] + [
        f"valid_pixel_fraction_t{i}" for i in range(n_t)
    ]
    chip_df = pd.DataFrame(rows, columns=cols)
    chip_manifest = manifest.merge(chip_df, on="chip_id", how="left")
    out_path = RAW_DIR / f"chip_manifest_{tag}_y{target_year}.parquet"
    chip_manifest.to_parquet(out_path, index=False)
    log.info("Wrote chip manifest with %d successful exports to %s",
             chip_df["chip_id"].nunique(), out_path)

    if failures:
        fail_path = RAW_DIR / f"chip_export_failures_{tag}_y{target_year}.csv"
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        log.warning("%d chips failed; details in %s", len(failures), fail_path)

    print(
        chip_manifest.groupby("region_short")["valid_pixel_fraction"]
        .agg(["count", "mean", "min"])
        .round(3)
    )


if __name__ == "__main__":
    main()
