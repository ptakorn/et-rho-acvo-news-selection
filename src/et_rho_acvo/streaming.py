from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from et_rho_acvo.metrics import mean_max_probability, streaming_agreement
from et_rho_acvo.modeling import fit_pseudo_label_model, make_kmeans, optimize_word2vec
from et_rho_acvo.representation import EnsembleTextRepresentation
from et_rho_acvo.text import tokenize_text


def _vocabulary(documents: list[str], language: str, min_token_length: int) -> set[str]:
    return {
        token
        for document in documents
        for token in tokenize_text(document, language, min_token_length)
    }


def run_stream(
    csv_path: str | Path,
    text_column: str,
    clusters: int,
    config: dict[str, Any],
    output_dir: str | Path,
    seed: int = 1000,
    timestamp_column: str | None = None,
    batches: int | None = None,
    trials: int | None = None,
) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    if text_column not in data.columns:
        raise ValueError(f"Column {text_column!r} not found in {csv_path}.")
    if timestamp_column:
        if timestamp_column not in data.columns:
            raise ValueError(f"Timestamp column {timestamp_column!r} not found in {csv_path}.")
        data = data.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
    else:
        data = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    documents = data[text_column].fillna("").astype(str).tolist()

    exp_cfg = config.get("experiment", {})
    text_cfg = config.get("text", {})
    language = str(text_cfg.get("language", "en"))
    min_token_length = int(text_cfg.get("min_token_length", 1))
    initial_fraction = float(exp_cfg.get("initial_train_fraction", 0.80))
    batch_count = int(batches if batches is not None else exp_cfg.get("stream_batches", 5))
    vocabulary_threshold = float(exp_cfg.get("vocabulary_threshold", 0.10))
    confidence_threshold = float(exp_cfg.get("confidence_threshold", 0.10))
    initial_size = int(len(documents) * initial_fraction)
    if initial_size < clusters or len(documents) - initial_size < batch_count:
        raise ValueError("Not enough rows for the configured initial portion, cluster count, and stream batches.")

    initial_docs = documents[:initial_size]
    stream_docs = documents[initial_size:]
    best_params = optimize_word2vec(initial_docs, config, clusters, seed, trials)
    encoder, kmeans, classifier, x_initial = fit_pseudo_label_model(
        initial_docs, config, clusters, seed, best_params
    )
    confidence_baseline = mean_max_probability(classifier.predict_proba(x_initial))
    vocabulary = _vocabulary(initial_docs, language, min_token_length)
    batch_arrays = np.array_split(np.arange(len(stream_docs)), batch_count)
    rows: list[dict[str, Any]] = []

    for batch_number, local_indices in enumerate(batch_arrays, start=1):
        batch_docs = [stream_docs[int(i)] for i in local_indices]
        if len(batch_docs) < clusters:
            raise ValueError(f"Batch {batch_number} has fewer rows than clusters ({clusters}).")

        x_batch = encoder.transform(batch_docs)
        probabilities = classifier.predict_proba(x_batch)
        predicted_partition = classifier.predict(x_batch)
        batch_reference_model = make_kmeans(config, clusters, seed + batch_number).fit(x_batch)
        batch_reference = batch_reference_model.labels_
        scores = streaming_agreement(predicted_partition, batch_reference)
        current_confidence = mean_max_probability(probabilities)
        batch_vocabulary = _vocabulary(batch_docs, language, min_token_length)
        novelty = len(batch_vocabulary - vocabulary) / max(1, len(vocabulary))
        confidence_drop = confidence_baseline - current_confidence
        vocabulary_trigger = novelty > vocabulary_threshold
        confidence_trigger = confidence_drop > confidence_threshold
        triggered = vocabulary_trigger or confidence_trigger
        reasons = []
        if vocabulary_trigger:
            reasons.append("vocabulary_novelty")
        if confidence_trigger:
            reasons.append("confidence_drop")

        row: dict[str, Any] = {
            "seed": seed,
            "batch": batch_number,
            "n_documents": len(batch_docs),
            "ari": scores["ari"],
            "ari_percent": scores["ari_percent"],
            "mean_max_probability": current_confidence,
            "confidence_baseline_before_batch": confidence_baseline,
            "confidence_drop": confidence_drop,
            "confidence_drop_percent_points": 100.0 * confidence_drop,
            "vocabulary_novelty": novelty,
            "vocabulary_novelty_percent": 100.0 * novelty,
            "vocabulary_trigger": vocabulary_trigger,
            "confidence_trigger": confidence_trigger,
            "retraining_triggered": triggered,
            "trigger_reason": "+".join(reasons) if reasons else "none",
            "updated": False,
        }

        if triggered:
            updated_params = optimize_word2vec(batch_docs, config, clusters, seed + batch_number, trials)
            encoder, kmeans, classifier, x_retrain = fit_pseudo_label_model(
                batch_docs, config, clusters, seed + batch_number, updated_params
            )
            # The manuscript specifies an in-sample confidence baseline after retraining.
            confidence_baseline = mean_max_probability(classifier.predict_proba(x_retrain))
            row["confidence_baseline_after_retraining"] = confidence_baseline
            row["updated"] = True

        rows.append(row)
        vocabulary.update(batch_vocabulary)

    result = pd.DataFrame(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / f"stream_batch_scores_seed_{seed}.csv", index=False)
    settings = {
        "input_csv": str(Path(csv_path).resolve()),
        "text_column": text_column,
        "timestamp_column": timestamp_column,
        "clusters": clusters,
        "seed": seed,
        "initial_train_fraction": initial_fraction,
        "batches": batch_count,
        "vocabulary_threshold": vocabulary_threshold,
        "confidence_threshold": confidence_threshold,
        "initial_confidence_is_in_sample": True,
        "post_retraining_confidence_is_in_sample": True,
        "batch_kmeans_is_independent": True,
        "cluster_ids_aligned_across_batches": False,
        "pseudo_label_filtering": False,
        "config": config,
    }
    (output / f"stream_settings_seed_{seed}.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return result
