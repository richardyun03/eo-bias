"""Common protocol every foundation-model wrapper must implement.

See CLAUDE.md §7.2.
"""

from typing import Protocol

import numpy as np
import torch


class FoundationModel(Protocol):
    name: str
    embedding_dim: int
    expected_bands: list[str]
    expected_size: int

    def preprocess(self, chip: np.ndarray) -> torch.Tensor: ...
    def embed(self, batch: torch.Tensor) -> torch.Tensor: ...  # (B, D)
