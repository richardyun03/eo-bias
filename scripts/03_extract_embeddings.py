"""Step 03: run OlmoEarth-v1-Base over chips and persist embeddings.

Inputs:  data/raw/chip_manifest_t{N}_y{year}.parquet
         data/chips_t{N}/{region_short}/{chip_id}.npz
Output:  data/embeddings/embeddings_t{N}_y{year}.npz   (chip_ids, embeddings)

Usage:
    uv run python scripts/03_extract_embeddings.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id

from geo_bias.eval.embed import embed_chips
from geo_bias.utils.seeds import set_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("03_extract_embeddings")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
EMB_DIR = ROOT / "data" / "embeddings"

VALID_THRESHOLD = 0.3
BATCH_SIZE = 16


def main() -> None:
    sampling_cfg = yaml.safe_load((ROOT / "configs" / "sampling.yaml").read_text())
    target_year = sampling_cfg["labels"]["target_year"]
    offsets = sampling_cfg["imagery"]["temporal_offsets"]
    years = [target_year + o for o in offsets]
    n_t = len(years)
    tag = f"t{n_t}"
    set_all(sampling_cfg["seeds"]["sampling"])

    manifest_path = RAW_DIR / f"chip_manifest_{tag}_y{target_year}.parquet"
    manifest = pd.read_parquet(manifest_path)

    pre_n = len(manifest)
    manifest = manifest.dropna(subset=["chip_path"])
    manifest = manifest[manifest["valid_pixel_fraction"] >= VALID_THRESHOLD].reset_index(
        drop=True
    )
    log.info(
        "[%s] Kept %d chips (filtered %d below valid_fraction=%.2f or missing path); years=%s",
        tag, len(manifest), pre_n - len(manifest), VALID_THRESHOLD, years,
    )

    log.info("Loading OlmoEarth-v1-Base...")
    model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log.info("Device: %s", device)

    normalizer = Normalizer(Strategy.COMPUTED)

    chip_ids, embeddings = embed_chips(
        manifest=manifest,
        model=model,
        normalizer=normalizer,
        years=years,
        device=device,
        root=ROOT,
        batch_size=BATCH_SIZE,
    )

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EMB_DIR / f"embeddings_{tag}_y{target_year}.npz"
    np.savez_compressed(out_path, chip_ids=chip_ids, embeddings=embeddings)
    log.info("Wrote (%d, %d) embeddings to %s", *embeddings.shape, out_path)


if __name__ == "__main__":
    main()
