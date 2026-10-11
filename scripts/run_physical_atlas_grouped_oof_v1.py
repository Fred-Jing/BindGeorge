#!/usr/bin/env python3
"""Run frozen train-only grouped OOF ablations for physical-atlas features."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from physical_atlas_schema_v1 import FEATURE_GROUPS, FEATURE_NAMES, LEGACY_FEATURES
except ModuleNotFoundError:  # Support importing as scripts.<module> in tests.
    from scripts.physical_atlas_schema_v1 import FEATURE_GROUPS, FEATURE_NAMES, LEGACY_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "features/physical_atlas_v1/ipae_min_le_2p5_known_target_train_validation/features_v1"
DEFAULT_TRAIN = PROJECT_ROOT / "processed/v1/training_10x/cohorts/ipae_min_le_2p5/known_target/train.tsv.gz"
DEFAULT_FOLDS = PROJECT_ROOT / "experiments/dmasif_feature_pruning_folds_v1.tsv"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs/v2/known_target/physical_atlas_grouped_oof_v1"
MODELS = ("logistic", "hist_gradient_boosting")


def read_tsv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def variants() -> dict[str, tuple[str, tuple[str, ...]]]:
    output: dict[str, tuple[str, tuple[str, ...]]] = {
        "raw_legacy11": ("raw", tuple(LEGACY_FEATURES)),
        "raw_all39": ("raw", tuple(FEATURE_NAMES)),
        "response_legacy44": ("response", tuple(LEGACY_FEATURES)),
        "response_all156": ("response", tuple(FEATURE_NAMES)),
    }
    for group, names in FEATURE_GROUPS.items():
        output[f"raw_{group}"] = ("raw", tuple(names))
        output[f"response_{group}"] = ("response", tuple(names))
        retained = tuple(name for name in FEATURE_NAMES if name not in names)
        output[f"raw_without_{group}"] = ("raw", retained)
        output[f"response_without_{group}"] = ("response", retained)
    return output


def columns(view: str, names: tuple[str, ...]) -> np.ndarray:
    indices = [FEATURE_NAMES.index(name) for name in names]
    if view == "raw":
        return np.asarray(indices, dtype=np.int64)
    if view == "response":
        dimension = len(FEATURE_NAMES)
        return np.asarray([block * dimension + index for block in range(4) for index in indices], dtype=np.int64)
    raise ValueError(view)


def estimator(model: str, seed: int) -> Pipeline:
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    if model == "logistic":
        return Pipeline(
            [
                ("imputer", imputer),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000, class_weight="balanced", random_state=seed)),
            ]
        )
    return Pipeline(
        [
            ("imputer", imputer),
            ("classifier", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, early_stopping=False, class_weight="balanced", random_state=seed)),
        ]
    )


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "auprc": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


def bootstrap(labels, probabilities, clusters, comparisons, replicates, seed):
    rng = np.random.default_rng(seed)
    unique = sorted(set(clusters.tolist()))
    cluster_indices = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    values = {(candidate, reference, metric): [] for candidate, reference in comparisons for metric in ("auprc", "auroc", "log_loss_improvement")}
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([cluster_indices[cluster] for cluster in sampled])
        y = labels[indices]
        if len(np.unique(y)) != 2:
            continue
        for candidate, reference in comparisons:
            candidate_metrics = metrics(y, probabilities[candidate][indices])
            reference_metrics = metrics(y, probabilities[reference][indices])
            values[(candidate, reference, "auprc")].append(candidate_metrics["auprc"] - reference_metrics["auprc"])
            values[(candidate, reference, "auroc")].append(candidate_metrics["auroc"] - reference_metrics["auroc"])
            values[(candidate, reference, "log_loss_improvement")].append(reference_metrics["log_loss"] - candidate_metrics["log_loss"])
    rows = []
    for (candidate, reference, metric), samples in values.items():
        array = np.asarray(samples)
        observed_candidate = metrics(labels, probabilities[candidate])
        observed_reference = metrics(labels, probabilities[reference])
        observed = (
            observed_reference["log_loss"] - observed_candidate["log_loss"]
            if metric == "log_loss_improvement"
            else observed_candidate[metric] - observed_reference[metric]
        )
        rows.append({
            "candidate": candidate,
            "reference": reference,
            "metric": metric,
            "observed_delta": observed,
            "bootstrap_mean_delta": float(array.mean()),
            "ci_2p5": float(np.quantile(array, 0.025)),
            "ci_97p5": float(np.quantile(array, 0.975)),
            "probability_delta_gt_0": float(np.mean(array > 0)),
            "valid_replicates": len(array),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output requires review: {args.output_dir}")
    cache_metadata = json.loads((args.feature_cache / "metadata.json").read_text())
    if (
        cache_metadata.get("status") != "complete"
        or cache_metadata.get("sample_count") != 3471
        or cache_metadata.get("sealed_test_accessed") is not False
        or cache_metadata.get("contains_labels") is not False
    ):
        raise ValueError("Unsafe or incomplete feature cache")
    train_rows = read_tsv(args.train)
    fold_rows = read_tsv(args.folds)
    if len(train_rows) != 2881:
        raise ValueError("Expected 2,881 training samples")
    fold_by_sample = {row["sample_id"]: int(row["cv_fold"]) for row in fold_rows}
    if set(fold_by_sample) != {row["sample_id"] for row in train_rows}:
        raise ValueError("Fold manifest does not exactly cover training samples")
    index_rows = read_tsv(args.feature_cache / "index.tsv")
    row_by_sample = {row["sample_id"]: int(row["sample_index"]) for row in index_rows}
    with np.load(args.feature_cache / "features.npz", allow_pickle=False) as archive:
        matrices = {view: np.asarray(archive[view], dtype=np.float64) for view in ("raw", "response")}
    sample_ids = sorted(row["sample_id"] for row in train_rows)
    metadata_by_sample = {row["sample_id"]: row for row in train_rows}
    labels = np.asarray([metadata_by_sample[sample_id]["label"] == "positive" for sample_id in sample_ids], dtype=np.int64)
    folds = np.asarray([fold_by_sample[sample_id] for sample_id in sample_ids], dtype=np.int64)
    clusters = np.asarray([metadata_by_sample[sample_id]["target_cluster_id"] for sample_id in sample_ids])
    leakage_units = np.asarray([metadata_by_sample[sample_id]["leakage_unit_id"] for sample_id in sample_ids])
    cache_rows = np.asarray([row_by_sample[sample_id] for sample_id in sample_ids], dtype=np.int64)

    variant_definitions = variants()
    summary_rows = []
    prediction_rows = []
    probability_map: dict[str, np.ndarray] = {}
    for model in MODELS:
        for variant, (view, feature_names) in variant_definitions.items():
            selected_columns = columns(view, feature_names)
            x = matrices[view][cache_rows][:, selected_columns]
            probabilities = np.full(len(sample_ids), np.nan, dtype=np.float64)
            for outer_fold in range(5):
                train_mask = folds != outer_fold
                outer_mask = folds == outer_fold
                if set(leakage_units[train_mask]) & set(leakage_units[outer_mask]):
                    raise ValueError(f"Leakage unit crosses outer fold {outer_fold}")
                fitted = estimator(model, args.seed + outer_fold)
                fitted.fit(x[train_mask], labels[train_mask])
                probabilities[outer_mask] = fitted.predict_proba(x[outer_mask])[:, 1]
            if not np.all(np.isfinite(probabilities)):
                raise ValueError(f"Non-finite OOF predictions for {model}/{variant}")
            key = f"{model}:{variant}"
            probability_map[key] = probabilities
            result = metrics(labels, probabilities)
            target_values = []
            for target in sorted(set(metadata_by_sample[sample_id]["target_sequence_id"] for sample_id in sample_ids)):
                selected = np.asarray([metadata_by_sample[sample_id]["target_sequence_id"] == target for sample_id in sample_ids])
                if len(np.unique(labels[selected])) == 2:
                    target_values.append(average_precision_score(labels[selected], probabilities[selected]))
            summary_rows.append({"model": model, "variant": variant, "view": view, "base_feature_count": len(feature_names), "model_feature_count": len(selected_columns), **result, "macro_target_auprc": float(np.mean(target_values)), "eligible_targets": len(target_values)})
            for index, sample_id in enumerate(sample_ids):
                prediction_rows.append({"model": model, "variant": variant, "outer_fold": int(folds[index]), "sample_id": sample_id, "label": int(labels[index]), "target_sequence_id": metadata_by_sample[sample_id]["target_sequence_id"], "target_cluster_id": clusters[index], "probability": float(probabilities[index])})

    comparisons = []
    for model in MODELS:
        comparisons.extend([
            (f"{model}:raw_all39", f"{model}:raw_legacy11"),
            (f"{model}:response_all156", f"{model}:response_legacy44"),
            (f"{model}:response_all156", f"{model}:raw_all39"),
        ])
        comparisons.extend((f"{model}:response_all156", f"{model}:response_without_{group}") for group in FEATURE_GROUPS)
    bootstrap_rows = bootstrap(labels, probability_map, clusters, comparisons, args.bootstrap_replicates, args.seed + 1000)
    args.output_dir.mkdir(parents=True)
    write_tsv(args.output_dir / "summary.tsv", summary_rows)
    write_tsv(args.output_dir / "oof_predictions.tsv.gz", prediction_rows)
    write_tsv(args.output_dir / "target_cluster_bootstrap.tsv", bootstrap_rows)
    metadata = {
        "status": "complete",
        "schema_version": "physical_atlas_grouped_oof_v1",
        "sample_count": len(sample_ids),
        "outer_folds": 5,
        "models": list(MODELS),
        "variant_count": len(variant_definitions),
        "comparison_count": len(comparisons),
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "frozen_hyperparameters": True,
        "project_validation_accessed": False,
        "sealed_test_accessed": False,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
