# geo-bias

Cross-region benchmark for geographical bias in Earth observation foundation
models. Binary deforestation classification (Hansen GFC labels, Sentinel-2
imagery) across four contrasting regions; frozen-embedding linear probes vs.
full fine-tuning. See [`CLAUDE.md`](CLAUDE.md) for the full project spec.

## Setup

```bash
uv venv --python 3.12
uv sync --extra dev
```

OlmoEarth is installed separately — see the comments at the top of
`pyproject.toml` for the two install paths.

Earth Engine auth (one-time, local):

```bash
uv run earthengine authenticate
```

## Pipeline

Numbered scripts run in order; each writes a manifest the next consumes.

```
scripts/01_sample_labels.py        # Hansen GFC -> chip manifest
scripts/02_export_chips.py         # Sentinel-2 composites
scripts/03_extract_embeddings.py   # frozen encoder pass
scripts/04_run_probes.py           # per-region + pooled logistic regression
scripts/05_finetune.py             # full fine-tuning
scripts/06_make_figures.py         # heatmap, bars, OSM-density scatter
```
