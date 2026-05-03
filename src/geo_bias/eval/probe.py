"""Per-region and pooled logistic-regression linear probes on frozen embeddings.

Standardizes embeddings on the train split only (per CLAUDE.md §8.1), fits
LogisticRegression with class_weight="balanced", and returns a flat metrics
dict on the test split. The standardization-on-train rule prevents any test
information from leaking into the probe via the scaler.
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


def fit_and_eval_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> dict[str, float | int]:
    """Fit a logistic-regression probe; return aggregate test metrics."""
    scaler = StandardScaler().fit(X_train)
    X_tr = scaler.transform(X_train)
    X_te = scaler.transform(X_test)

    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    ).fit(X_tr, y_train)

    proba = clf.predict_proba(X_te)[:, 1]
    pred = clf.predict(X_te)

    return {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_train_pos": int(np.sum(y_train == 1)),
        "n_test_pos": int(np.sum(y_test == 1)),
        "f1": float(f1_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }
