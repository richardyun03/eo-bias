"""Run export_chip on a single known-good location and surface the real error.

Run:
    export EE_PROJECT=<your-gcp-project-id>
    uv run python scripts/_smoke_export_one.py
"""

from geo_bias.data.ee_client import init_ee
from geo_bias.data.sentinel import export_chip

init_ee()

# Rondônia, Brazilian Amazon — known to have abundant clear S2 in 2020.
chip, vf = export_chip(lon=-62.5, lat=-10.0, year=2020)
print("chip shape:", chip.shape, "dtype:", chip.dtype)
print("valid_pixel_fraction:", vf)
print("min, max, mean:", chip.min(), chip.max(), chip.mean())
