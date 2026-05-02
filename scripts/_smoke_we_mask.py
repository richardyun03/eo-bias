"""Diagnose why Western Europe negatives return 0 candidates.

Counts pixels for the masks we use, and prints the lossyear histogram inside
the WE forest mask. This separates "the mask is empty" from "stratifiedSample
is misbehaving."

Run:
    export EE_PROJECT=<your-gcp-project-id>
    uv run python scripts/_smoke_we_mask.py
"""

import os
import sys

import ee

PROJECT = os.environ.get("EE_PROJECT")
if not PROJECT:
    sys.exit("EE_PROJECT not set.")
ee.Initialize(project=PROJECT)

WE = ee.Geometry.Rectangle([5.0, 46.0, 11.0, 50.5])
gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
forest = gfc.select("treecover2000").gte(30).And(gfc.select("datamask").eq(1))
lossyear = gfc.select("lossyear")

# Use a coarse scale for fast counts; we only need orders of magnitude.
SCALE = 300
KW = {"geometry": WE, "scale": SCALE, "maxPixels": 1e10, "tileScale": 16}


def count_unmasked(img: ee.Image, label: str) -> int:
    n = (
        img.selfMask()
        .reduceRegion(reducer=ee.Reducer.count(), **KW)
        .getInfo()
    )
    print(f"{label}: {n}")
    return n


print(f"--- WE bbox; counting at scale={SCALE} m (will be ~100x fewer than at 30 m) ---")
count_unmasked(forest, "forest pixels")
count_unmasked(forest.And(lossyear.eq(0)), "forest AND lossyear==0 (stable forest)")
count_unmasked(forest.And(lossyear.lt(1)), "forest AND lossyear<1  (alt formulation)")
count_unmasked(forest.And(lossyear.eq(20)), "forest AND lossyear==20 (loss-2020)")

print("\nLossyear histogram inside WE forest mask:")
hist = (
    lossyear.updateMask(forest)
    .reduceRegion(reducer=ee.Reducer.frequencyHistogram(), **KW)
    .getInfo()
)
print(hist)
