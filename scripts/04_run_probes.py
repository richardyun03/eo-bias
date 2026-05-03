"""Step 04: fit per-region and pooled linear probes; write metrics.

Inputs:  data/embeddings/embeddings_y{year}.npz
         data/raw/chip_manifest_y{year}.parquet
Output:  data/results/probe_metrics_y{year}.parquet

Usage:
    uv run python scripts/04_run_probes.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geo_bias.eval.probe import fit_and_eval_probe
from geo_bias.utils.seeds import set_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("04_run_probes")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
EMB_DIR = ROOT / "data" / "embeddings"
RES_DIR = ROOT / "data" / "results"


def main() -> None:
    sampling_cfg = yaml.safe_load((ROOT / "configs" / "sampling.yaml").read_text())
    target_year = sampling_cfg["labels"]["target_year"]
    seed = sampling_cfg["seeds"]["probe"]
    set_all(seed)

    emb = np.load(EMB_DIR / f"embeddings_y{target_year}.npz", allow_pickle=False)
    chip_ids = emb["chip_ids"]
    embeddings = emb["embeddings"]
    manifest = pd.read_parquet(RAW_DIR / f"chip_manifest_y{target_year}.parquet")

    # Align manifest rows to chip_ids order
    df = manifest.set_index("chip_id").loc[chip_ids].reset_index()
    X = embeddings
    y = df["label"].to_numpy()
    region = df["region_short"].to_numpy()
    split = df["split"].to_numpy()

    rows: list[dict] = []

    for r in sorted(np.unique(region)):
        sel = region == r
        Xr, yr, sr = X[sel], y[sel], split[sel]
        train, test = sr == "train", sr == "test"
        if train.sum() == 0 or test.sum() == 0:
            log.warning("Skipping %s: train=%d test=%d", r, train.sum(), test.sum())
            continue
        m = fit_and_eval_probe(Xr[train], yr[train], Xr[test], yr[test], seed=seed)
        m["region_short"] = r
        m["probe"] = "per_region"
        rows.append(m)
        log.info(
            "[%s per_region] F1=%.3f AUC=%.3f n_train=%d n_test=%d",
            r, m["f1"], m["roc_auc"], m["n_train"], m["n_test"],
        )

    train, test = split == "train", split == "test"
    m = fit_and_eval_probe(X[train], y[train], X[test], y[test], seed=seed)
    m["region_short"] = "GLOBAL"
    m["probe"] = "pooled"
    rows.append(m)
    log.info(
        "[GLOBAL pooled] F1=%.3f AUC=%.3f n_train=%d n_test=%d",
        m["f1"], m["roc_auc"], m["n_train"], m["n_test"],
    )

    res_df = pd.DataFrame(rows)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RES_DIR / f"probe_metrics_y{target_year}.parquet"
    res_df.to_parquet(out_path, index=False)
    log.info("Wrote metrics to %s", out_path)
    print(
        res_df[
            ["region_short", "probe", "n_train", "n_test", "f1", "precision", "recall", "roc_auc", "pr_auc"]
        ].round(3).to_string(index=False)
    )


if __name__ == "__main__":
    main()
