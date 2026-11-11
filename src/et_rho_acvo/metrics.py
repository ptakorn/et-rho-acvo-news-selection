from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, adjusted_rand_score, precision_recall_fscore_support


def agreement_metrics(y_reference: np.ndarray, y_prediction: np.ndarray, average: str = "macro") -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_reference, y_prediction, average=average, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_reference, y_prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def streaming_agreement(y_model: np.ndarray, y_kmeans: np.ndarray) -> dict[str, float]:
    ari = float(adjusted_rand_score(y_kmeans, y_model))
    return {"ari": ari, "ari_percent": 100.0 * ari}


def mean_max_probability(probabilities: np.ndarray) -> float:
    if probabilities.size == 0:
        return float("nan")
    return float(np.mean(np.max(probabilities, axis=1)))
