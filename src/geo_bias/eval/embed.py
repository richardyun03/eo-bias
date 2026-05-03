"""Batched frozen-encoder embedding extraction over chip manifests.

Loads chip arrays from disk, replaces the export sentinel (-9999) with 0,
applies OlmoEarth's normalizer, runs the encoder with the same fast_pass +
patch_size=4 settings as Inference-Quickstart, and returns a global-mean
pooled (N, D) embedding array aligned to chip_ids.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from tqdm import tqdm

from geo_bias.data.sentinel import DEFAULT_VALUE

log = logging.getLogger(__name__)

H = W = 64
T = 1
NUM_BAND_SETS = Modality.SENTINEL2_L2A.num_band_sets


def _load_chip(rel_path: str, root: Path) -> np.ndarray:
    """Load chip from .npz, replace sentinel with 0. Returns (H, W, C) int32."""
    with np.load(root / rel_path) as data:
        chip = data["image"]
    return np.where(chip == DEFAULT_VALUE, 0, chip).astype(np.int32)


def embed_chips(
    *,
    manifest: pd.DataFrame,
    model: torch.nn.Module,
    normalizer: Normalizer,
    target_year: int,
    device: torch.device,
    root: Path,
    batch_size: int = 16,
    patch_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Run OlmoEarth encoder over manifest chips. Returns (chip_ids, embeddings)."""
    embeddings: list[np.ndarray] = []
    chip_ids: list[str] = []

    # Day=15, month=5 (June, 0-indexed), year=target_year — middle of the composite window.
    ts_single = torch.tensor(
        [15, 5, target_year], dtype=torch.long, device=device
    ).reshape(1, 1, 3)

    for start in tqdm(range(0, len(manifest), batch_size), desc="encoding"):
        rows = manifest.iloc[start : start + batch_size]
        b = len(rows)

        chips = np.stack(
            [_load_chip(p, root) for p in rows["chip_path"]], axis=0
        )  # (B, H, W, C)
        chips = chips[:, :, :, None, :]  # (B, H, W, T, C)
        chips = normalizer.normalize(Modality.SENTINEL2_L2A, chips)

        image = torch.tensor(chips, dtype=torch.float32, device=device)
        mask = torch.full(
            (b, H, W, T, NUM_BAND_SETS),
            MaskValue.ONLINE_ENCODER.value,
            dtype=torch.float32,
            device=device,
        )
        timestamps = ts_single.expand(b, T, 3).contiguous()

        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=image,
            sentinel2_l2a_mask=mask,
            timestamps=timestamps,
        )
        with torch.no_grad():
            out = model.encoder(sample, fast_pass=True, patch_size=patch_size)
        features = out["tokens_and_masks"].sentinel2_l2a  # (B, H', W', T, S, D)
        pooled = features.mean(dim=[1, 2, 3, 4]).cpu().numpy()  # (B, D)
        embeddings.append(pooled)
        chip_ids.extend(rows["chip_id"].tolist())

    return np.array(chip_ids), np.concatenate(embeddings, axis=0)
