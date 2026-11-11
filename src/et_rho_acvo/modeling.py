from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from et_rho_acvo.representation import EnsembleTextRepresentation


def make_kmeans(config: dict[str, Any], clusters: int, seed: int) -> KMeans:
    model_cfg = config.get("models", {})
    return KMeans(
        n_clusters=int(clusters),
        n_init=int(model_cfg.get("kmeans_n_init", 10)),
        max_iter=int(model_cfg.get("kmeans_max_iter", 300)),
        random_state=int(seed),
    )


def make_classifier(config: dict[str, Any], seed: int) -> RandomForestClassifier:
    model_cfg = config.get("models", {})
    return RandomForestClassifier(
        n_estimators=int(model_cfg.get("rf_estimators", 300)),
        max_features=model_cfg.get("rf_max_features", "sqrt"),
        class_weight=model_cfg.get("rf_class_weight", "balanced_subsample"),
        n_jobs=int(model_cfg.get("rf_n_jobs", -1)),
        random_state=int(seed),
    )


def optimize_word2vec(
    documents: list[str], config: dict[str, Any], clusters: int, seed: int, trials: int | None = None
) -> dict[str, Any]:
    """TPE search maximizing CV agreement with fold-specific K-means labels."""
    exp_cfg = config.get("experiment", {})
    rep_cfg = config.get("representation", {})
    n_splits = int(exp_cfg.get("cv_folds", 5))
    if len(documents) < n_splits:
        raise ValueError(f"Need at least {n_splits} documents for the configured CV.")
    n_trials = int(trials if trials is not None else exp_cfg.get("tpe_trials", 1000))
    patience = int(exp_cfg.get("tpe_patience", 10))
    vector_min = int(rep_cfg.get("tpe_vector_size_min", 50))
    vector_max = int(rep_cfg.get("tpe_vector_size_max", 200))
    vector_step = int(rep_cfg.get("tpe_vector_size_step", 10))
    window_min = int(rep_cfg.get("tpe_window_min", 3))
    window_max = int(rep_cfg.get("tpe_window_max", 10))
    sg_values = [int(v) for v in rep_cfg.get("tpe_sg_values", [0, 1])]
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(np.arange(len(documents))))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "vector_size": trial.suggest_int("vector_size", vector_min, vector_max, step=vector_step),
            "window": trial.suggest_int("window", window_min, window_max),
            "sg": trial.suggest_categorical("sg", sg_values),
        }
        fold_scores: list[float] = []
        for fold_id, (train_idx, val_idx) in enumerate(splits):
            tr_docs = [documents[i] for i in train_idx]
            val_docs = [documents[i] for i in val_idx]
            fold_seed = seed + fold_id
            encoder = EnsembleTextRepresentation(config, params, fold_seed).fit(tr_docs)
            x_train = encoder.transform(tr_docs)
            x_val = encoder.transform(val_docs)
            kmeans = make_kmeans(config, clusters, fold_seed).fit(x_train)
            y_train = kmeans.labels_
            y_val_reference = kmeans.predict(x_val)
            classifier = make_classifier(config, fold_seed).fit(x_train, y_train)
            y_val_pred = classifier.predict(x_val)
            fold_scores.append(float(accuracy_score(y_val_reference, y_val_pred)))
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    best_value = -np.inf
    stale = 0

    def stop_after_patience(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        nonlocal best_value, stale
        if trial.value is not None and trial.value > best_value:
            best_value = float(trial.value)
            stale = 0
        else:
            stale += 1
        if patience > 0 and stale >= patience:
            study_.stop()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, callbacks=[stop_after_patience], show_progress_bar=False)
    if not study.best_trial:
        raise RuntimeError("TPE did not complete a successful trial.")
    return dict(study.best_params)


def fit_pseudo_label_model(
    documents: list[str], config: dict[str, Any], clusters: int, seed: int, word2vec_params: dict[str, Any]
) -> tuple[EnsembleTextRepresentation, KMeans, RandomForestClassifier, np.ndarray]:
    encoder = EnsembleTextRepresentation(config, word2vec_params, seed).fit(documents)
    features = encoder.transform(documents)
    kmeans = make_kmeans(config, clusters, seed).fit(features)
    pseudo_labels = kmeans.labels_
    classifier = make_classifier(config, seed).fit(features, pseudo_labels)
    return encoder, kmeans, classifier, features
