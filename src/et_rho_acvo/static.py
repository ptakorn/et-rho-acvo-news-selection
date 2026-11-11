from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from et_rho_acvo.metrics import agreement_metrics
from et_rho_acvo.modeling import fit_pseudo_label_model, optimize_word2vec
from et_rho_acvo.statistics import percentile_bootstrap_ci


def run_static(
    csv_path: str | Path,
    text_column: str,
    clusters: int,
    config: dict[str, Any],
    output_dir: str | Path,
    seed_start: int | None = None,
    runs: int | None = None,
    trials: int | None = None,
) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    if text_column not in data.columns:
        raise ValueError(f"Column {text_column!r} not found in {csv_path}.")
    documents = data[text_column].fillna("").astype(str).tolist()
    if len(documents) < int(clusters):
        raise ValueError("The number of documents must be at least the number of clusters.")

    exp_cfg = config.get("experiment", {})
    seed_first = int(seed_start if seed_start is not None else exp_cfg.get("random_seed_start", 1000))
    run_count = int(runs if runs is not None else exp_cfg.get("repetitions", 30))
    test_fraction = float(exp_cfg.get("test_fraction", 0.20))
    average = str(exp_cfg.get("metric_average", "macro"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for seed in range(seed_first, seed_first + run_count):
        train_docs, test_docs = train_test_split(
            documents, test_size=test_fraction, random_state=seed, shuffle=True
        )
        best_params = optimize_word2vec(train_docs, config, clusters, seed, trials)
        encoder, kmeans, classifier, x_train = fit_pseudo_label_model(
            train_docs, config, clusters, seed, best_params
        )
        x_test = encoder.transform(test_docs)
        y_reference = kmeans.predict(x_test)
        y_prediction = classifier.predict(x_test)
        scores = agreement_metrics(y_reference, y_prediction, average=average)
        rows.append({"seed": seed, "clusters": clusters, **scores, **{f"w2v_{k}": v for k, v in best_params.items()}})

    result = pd.DataFrame(rows)
    result.to_csv(output / "static_run_scores.csv", index=False)
    metric_cols = ["accuracy", "precision", "recall", "f1"]
    summary_rows = []
    for metric_index, column in enumerate(metric_cols):
        ci_low, ci_high = percentile_bootstrap_ci(
            result[column].to_numpy(),
            confidence=0.95,
            resamples=10000,
            seed=seed_first + metric_index,
        )
        summary_rows.append({
            "metric": column,
            "mean": float(result[column].mean()),
            "sd": float(result[column].std(ddof=1)),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "static_summary.csv", index=False)
    settings = {
        "input_csv": str(Path(csv_path).resolve()),
        "text_column": text_column,
        "clusters": clusters,
        "seed_start": seed_first,
        "runs": run_count,
        "test_fraction": test_fraction,
        "cv_folds": int(exp_cfg.get("cv_folds", 5)),
        "tpe_trials_requested": int(trials if trials is not None else exp_cfg.get("tpe_trials", 1000)),
        "metric_average": average,
        "metric_interpretation": "agreement with K-means pseudo-labels, not human semantic accuracy",
        "config": config,
    }
    (output / "static_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return result
