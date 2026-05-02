"""Smoke test: load OlmoEarth-v1-Base and run a synthetic forward pass.

Validates: HF weight download, encoder API, expected band count, output shape.
Run: uv run python scripts/_smoke_olmoearth.py
"""

import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id

H = W = 64
T = 1
C = Modality.SENTINEL2_L2A.num_bands
NUM_BAND_SETS = Modality.SENTINEL2_L2A.num_band_sets

print(f"Sentinel-2 band order: {Modality.SENTINEL2_L2A.band_order}")
print(f"  num_bands={C}, num_band_sets={NUM_BAND_SETS}")

print("Loading OlmoEarth-v1-Base from Hugging Face...")
model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device: {device}")

image = torch.randn(1, H, W, T, C, device=device)
mask = torch.ones(1, H, W, T, NUM_BAND_SETS, device=device) * MaskValue.ONLINE_ENCODER.value
timestamps = torch.tensor([[[15, 5, 2020]]], device=device)  # mid-June 2020

sample = MaskedOlmoEarthSample(
    sentinel2_l2a=image,
    sentinel2_l2a_mask=mask,
    timestamps=timestamps,
)

with torch.no_grad():
    out = model.encoder(sample, fast_pass=True, patch_size=4)

features = out["tokens_and_masks"].sentinel2_l2a
print(f"Raw token shape (B, H', W', T, S, D): {tuple(features.shape)}")

pooled = features.mean(dim=[1, 2, 3, 4])  # global pool -> (B, D)
print(f"Pooled embedding shape (B, D): {tuple(pooled.shape)}")
print(f"Embedding dim D = {pooled.shape[-1]}")
