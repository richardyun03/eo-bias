"""Step 04: fit per-region/pooled probes + cross-region transfer matrix.

Inputs:  data/embeddings/embeddings_t{N}_y{year}.npz
         data/raw/chip_manifest_t{N}_y{year}.parquet
Output:  data/results/probe_metrics_t{N}_y{year}.parquet      (with bootstrap CIs)
         data/results/transfer_matrix_t{N}_y{year}.parquet    (train_region × test_region)

Usage:
    uv run python scripts/04_run_probes.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geo_bias.eval.probe import eval_probe, fit_and_eval_probe, fit_probe
from geo_bias.utils.seeds import set_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("04_run_probes")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
EMB_DIR = ROOT / "data" / "embeddings"
RES_DIR = ROOT / "data" / "results"

N_BOOT = 1000


def main() -> None:
    sampling_cfg = yaml.safe_load((ROOT / "configs" / "sampling.yaml").read_text())
    target_year = sampling_cfg["labels"]["target_year"]
    n_t = len(sampling_cfg["imagery"]["temporal_offsets"])
    tag = f"t{n_t}"
    seed = sampling_cfg["seeds"]["probe"]
    set_all(seed)

    emb = np.load(EMB_DIR / f"embeddings_{tag}_y{target_year}.npz", allow_pickle=False)
    chip_ids = emb["chip_ids"]
    embeddings = emb["embeddings"]
    manifest = pd.read_parquet(RAW_DIR / f"chip_manifest_{tag}_y{target_year}.parquet")

    df = manifest.set_index("chip_id").loc[chip_ids].reset_index()
    X = embeddings
    y = df["label"].to_numpy()
    region = df["region_short"].to_numpy()
    split = df["split"].to_numpy()

    regions = sorted(np.unique(region))
    rows: list[dict] = []

    # --- Per-region probes (fit + eval on same region, with CIs)
    fitted: dict[str, tuple] = {}  # train_region -> (clf, scaler)
    for r in regions:
        sel = region == r
        Xr, yr, sr = X[sel], y[sel], split[sel]
        train, test = sr == "train", sr == "test"
        if train.sum() == 0 or test.sum() == 0:
            log.warning("Skipping %s: train=%d test=%d", r, train.sum(), test.sum())
            continue
        fitted[r] = fit_probe(Xr[train], yr[train], seed=seed)
        m = fit_and_eval_probe(Xr[train], yr[train], Xr[test], yr[test], seed=seed, n_boot=N_BOOT)
        m["region_short"] = r
        m["probe"] = "per_region"
        rows.append(m)
        log.info(
            "[%s %s per_region] F1=%.3f [%.3f, %.3f]  AUC=%.3f [%.3f, %.3f]",
            tag, r, m["f1"], m["f1_lo"], m["f1_hi"], m["roc_auc"], m["roc_auc_lo"], m["roc_auc_hi"],
        )

    # --- Pooled probe (all regions)
    train, test = split == "train", split == "test"
    fitted["GLOBAL"] = fit_probe(X[train], y[train], seed=seed)
    m = fit_and_eval_probe(X[train], y[train], X[test], y[test], seed=seed, n_boot=N_BOOT)
    m["region_short"] = "GLOBAL"
    m["probe"] = "pooled"
    rows.append(m)
    log.info(
        "[%s GLOBAL pooled] F1=%.3f [%.3f, %.3f]  AUC=%.3f [%.3f, %.3f]",
        tag, m["f1"], m["f1_lo"], m["f1_hi"], m["roc_auc"], m["roc_auc_lo"], m["roc_auc_hi"],
    )

    metrics_path = RES_DIR / f"probe_metrics_{tag}_y{target_year}.parquet"
    RES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(metrics_path, index=False)
    log.info("Wrote per-region/pooled metrics to %s", metrics_path)

    # --- Cross-region transfer matrix (train_region × test_region)
    transfer_rows: list[dict] = []
    for tr in [*regions, "GLOBAL"]:
        if tr not in fitted:
            continue
        clf, scaler = fitted[tr]
        for te in regions:
            sel = (region == te) & (split == "test")
            if sel.sum() == 0:
                continue
            tm = eval_probe(clf, scaler, X[sel], y[sel], seed=seed, n_boot=N_BOOT)
            tm["train_region"] = tr
            tm["test_region"] = te
            transfer_rows.append(tm)
            log.info(
                "[%s transfer %s -> %s] F1=%.3f [%.3f, %.3f]",
                tag, tr, te, tm["f1"], tm["f1_lo"], tm["f1_hi"],
            )

    transfer_df = pd.DataFrame(transfer_rows)
    transfer_path = RES_DIR / f"transfer_matrix_{tag}_y{target_year}.parquet"
    transfer_df.to_parquet(transfer_path, index=False)
    log.info("Wrote transfer matrix to %s", transfer_path)

    print("\n--- per-region / pooled (with 95% CIs) ---")
    cols = ["region_short", "probe", "n_train", "n_test", "f1", "f1_lo", "f1_hi",
            "roc_auc", "roc_auc_lo", "roc_auc_hi"]
    print(pd.DataFrame(rows)[cols].round(3).to_string(index=False))

    print("\n--- transfer F1 (train_region rows × test_region cols) ---")
    pivot = transfer_df.pivot(index="train_region", columns="test_region", values="f1")
    print(pivot.round(3))


if __name__ == "__main__":
    main()
