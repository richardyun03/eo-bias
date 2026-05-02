"""Global seed setter for numpy / torch / random."""

import os
import random

import numpy as np


def set_all(seed: int) -> None:
    """Seed every PRNG we use. Call once at the top of each script."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
