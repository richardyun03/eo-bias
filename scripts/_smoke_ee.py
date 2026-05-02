"""Smoke test: verify Earth Engine auth + Hansen GFC + Sentinel-2 access.

Pulls a tiny patch (~640 m × 640 m, matching our 64 px × 10 m chip) over a
known Brazilian Amazon clearing and prints summary stats. Cheap; no exports.

Prereqs (run once on your machine):
    uv run earthengine authenticate
    # then set your GEE Cloud project (any project with the EE API enabled):
    export EE_PROJECT=your-gcp-project-id

Run:
    uv run python scripts/_smoke_ee.py
"""

import os
import sys

import ee

PROJECT = os.environ.get("EE_PROJECT")
if not PROJECT:
    sys.exit("EE_PROJECT env var not set. Export your GCP project id and retry.")

ee.Initialize(project=PROJECT)
print(f"EE initialized with project={PROJECT}")

# Tiny AOI in Rondônia, Brazil — known deforestation arc.
LON, LAT = -62.5, -10.0
HALF_DEG = 0.0032  # ~640 m at this latitude
aoi = ee.Geometry.Rectangle([LON - HALF_DEG, LAT - HALF_DEG, LON + HALF_DEG, LAT + HALF_DEG])

# 1. Hansen GFC label availability
gfc = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
loss_count = (
    gfc.select("lossyear")
    .gt(0)
    .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e8)
    .getInfo()
)
treecover = (
    gfc.select("treecover2000")
    .reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=30, maxPixels=1e8)
    .getInfo()
)
print(f"Hansen GFC OK: lossyear>0 pixels in AOI = {loss_count}, mean treecover2000 = {treecover}")

# 2. Sentinel-2 collection availability for target year
s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate("2020-01-01", "2021-01-01")
)
n_images = s2.size().getInfo()
print(f"Sentinel-2 OK: {n_images} L2A images intersect AOI in 2020")

# 3. Confirm we can pull a small numeric sample (proxy for getDownloadURL working)
sample = s2.median().select(["B2", "B3", "B4"]).sample(region=aoi, scale=10, numPixels=4).getInfo()
print(f"Pulled {len(sample['features'])} sample pixels — auth + compute path OK")
