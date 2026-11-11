from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def percentile_bootstrap_ci(
    values: list[float] | np.ndarray,
    confidence: float = 0.95,
    resamples: int = 10000,
    seed: int = 1000,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in original order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.empty_like(p)
    running_max = 0.0
    total = len(p)
    for rank, original_idx in enumerate(order):
        candidate = min(1.0, (total - rank) * p[original_idx])
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = running_max
    adjusted = np.empty_like(p)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def compare_dataset_methods(
    scores: pd.DataFrame,
    reference_method: str = "ET-RHO",
    metric: str = "f1",
    dataset_column: str = "dataset",
    method_column: str = "method",
) -> pd.DataFrame:
    """Paired two-sided Wilcoxon tests on one score per dataset and method."""
    required = {dataset_column, method_column, metric}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Missing columns in input score file: {sorted(missing)}")
    if scores.duplicated([dataset_column, method_column]).any():
        raise ValueError("Provide exactly one dataset-level score per method and dataset.")
    wide = scores.pivot(index=dataset_column, columns=method_column, values=metric)
    if reference_method not in wide.columns:
        raise ValueError(f"Reference method {reference_method!r} was not found.")
    comparisons = []
    for method in wide.columns:
        if method == reference_method:
            continue
        paired = wide[[reference_method, method]].dropna()
        if len(paired) < 2:
            raise ValueError(f"Need at least two paired datasets to compare with {method!r}.")
        differences = paired[reference_method].to_numpy() - paired[method].to_numpy()
        try:
            p_value = float(wilcoxon(differences, alternative="two-sided", method="auto").pvalue)
        except ValueError:
            # All-zero differences have no evidence of a directional difference.
            p_value = 1.0
        comparisons.append({
            "reference": reference_method,
            "comparison": method,
            "metric": metric,
            "n_datasets": int(len(paired)),
            "median_difference_reference_minus_comparison": float(np.median(differences)),
            "p_raw": p_value,
        })
    adjusted = holm_adjust([row["p_raw"] for row in comparisons])
    for row, p_adj in zip(comparisons, adjusted):
        row["p_holm"] = p_adj
    return pd.DataFrame(comparisons)
