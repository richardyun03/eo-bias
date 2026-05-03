"""Logistic-regression linear probes on frozen embeddings.

Standardizes embeddings on the train split only (CLAUDE.md §8.1), fits
LogisticRegression with class_weight="balanced", and evaluates on the test
split. Splits fit/eval into separate functions so cross-region transfer (one
trained probe applied to multiple test regions) is a natural composition.

Bootstrap 95% CIs are computed by resampling test-set indices (test-only
variance, not refitting the probe — standard light option). 1,000 resamples
default; ~5 ms per region.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def fit_probe(
    X_train: np.ndarray, y_train: np.ndarray, seed: int = 42
) -> tuple[LogisticRegression, StandardScaler]:
    """Fit StandardScaler + LogisticRegression on the train split."""
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    ).fit(scaler.transform(X_train), y_train)
    return clf, scaler


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """95% CI on F1 / ROC-AUC / PR-AUC via test-set resampling."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1s, aucs, praucs = [], [], []
    for _ in range(n_boot):
        ii = rng.integers(0, n, size=n)
        yt, yp, yr = y_true[ii], y_pred[ii], y_proba[ii]
        if len(np.unique(yt)) < 2:
            continue  # AUC undefined when only one class is sampled
        f1s.append(f1_score(yt, yp))
        aucs.append(roc_auc_score(yt, yr))
        praucs.append(average_precision_score(yt, yr))
    return {
        "f1_lo": float(np.percentile(f1s, 2.5)),
        "f1_hi": float(np.percentile(f1s, 97.5)),
        "roc_auc_lo": float(np.percentile(aucs, 2.5)),
        "roc_auc_hi": float(np.percentile(aucs, 97.5)),
        "pr_auc_lo": float(np.percentile(praucs, 2.5)),
        "pr_auc_hi": float(np.percentile(praucs, 97.5)),
    }


def eval_probe(
    clf: LogisticRegression,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    n_boot: int = 1000,
) -> dict[str, float | int]:
    """Apply a fitted probe to a test set; return metrics (+ optional CIs)."""
    X_te = scaler.transform(X_test)
    proba = clf.predict_proba(X_te)[:, 1]
    pred = clf.predict(X_te)
    metrics: dict[str, float | int] = {
        "n_test": int(len(y_test)),
        "n_test_pos": int(np.sum(y_test == 1)),
        "f1": float(f1_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }
    if n_boot > 0:
        metrics.update(_bootstrap_ci(y_test, pred, proba, n_boot=n_boot, seed=seed))
    return metrics


def fit_and_eval_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    n_boot: int = 1000,
) -> dict[str, float | int]:
    """Convenience wrapper: fit on train, evaluate on test, return metrics."""
    clf, scaler = fit_probe(X_train, y_train, seed=seed)
    out: dict[str, float | int] = {
        "n_train": int(len(y_train)),
        "n_train_pos": int(np.sum(y_train == 1)),
    }
    out.update(eval_probe(clf, scaler, X_test, y_test, seed=seed, n_boot=n_boot))
    return out
