"""Batched frozen-encoder embedding extraction over chip manifests.

Loads chips with shape (H, W, T, C) — T can be 1 (single-year median) or
greater (multi-year stack). Each timestep gets its own (day, month, year)
timestamp passed to OlmoEarth so the encoder's temporal embeddings know
which year each composite represents. Sentinel -9999 values are replaced
with 0 prior to normalization. Output: (B, D) globally mean-pooled.
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

NUM_BAND_SETS = Modality.SENTINEL2_L2A.num_band_sets


def _load_chip(rel_path: str, root: Path, expected_T: int) -> np.ndarray:
    """Load chip from .npz, replace sentinel with 0, ensure (H, W, T, C) shape."""
    with np.load(root / rel_path) as data:
        chip = data["image"]
    chip = np.where(chip == DEFAULT_VALUE, 0, chip).astype(np.int32)
    if chip.ndim == 3:
        # Legacy single-year file: (H, W, C). Promote to (H, W, 1, C).
        chip = chip[:, :, None, :]
    if chip.shape[2] != expected_T:
        raise ValueError(
            f"Chip {rel_path} has T={chip.shape[2]}; config expects T={expected_T}."
        )
    return chip


def embed_chips(
    *,
    manifest: pd.DataFrame,
    model: torch.nn.Module,
    normalizer: Normalizer,
    years: list[int],
    device: torch.device,
    root: Path,
    batch_size: int = 16,
    patch_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Run OlmoEarth encoder over manifest chips. Returns (chip_ids, embeddings)."""
    T = len(years)
    embeddings: list[np.ndarray] = []
    chip_ids: list[str] = []

    # (1, T, 3): day=15, month=5 (June, 0-indexed), year=year_for_t.
    ts_template = torch.tensor(
        [[15, 5, y] for y in years], dtype=torch.long, device=device
    ).reshape(1, T, 3)

    for start in tqdm(range(0, len(manifest), batch_size), desc="encoding"):
        rows = manifest.iloc[start : start + batch_size]
        b = len(rows)

        chips = np.stack(
            [_load_chip(p, root, T) for p in rows["chip_path"]], axis=0
        )  # (B, H, W, T, C)
        chips = normalizer.normalize(Modality.SENTINEL2_L2A, chips)

        image = torch.tensor(chips, dtype=torch.float32, device=device)
        _, H, W, _, _ = image.shape
        mask = torch.full(
            (b, H, W, T, NUM_BAND_SETS),
            MaskValue.ONLINE_ENCODER.value,
            dtype=torch.float32,
            device=device,
        )
        timestamps = ts_template.expand(b, T, 3).contiguous()

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
