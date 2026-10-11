#!/usr/bin/env python3
"""Confirm the literature-aligned ipSAE_min baseline on frozen grouped folds."""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "processed/v1/training_10x/cohorts/ipae_min_le_2p5/known_target/train.tsv.gz"
COHORT = ROOT / "processed/v1/training_10x/cohorts/ipae_min_le_2p5/cohort.tsv.gz"
FOLDS = ROOT / "experiments/dmasif_feature_pruning_folds_v1.tsv"
FEATURES = ROOT / "features/ipsae_v1/ipae_min_le_2p5_known_target_train/features.tsv"
MULTICHAIN = ROOT / "experiments/ipsae_multichain_audit_v1.tsv"
LEGACY_OOF = ROOT / "runs/v2/known_target/ipsae_v1/oof_predictions.tsv.gz"
OUTPUT = ROOT / "runs/v2/known_target/ipsae_v1/literature_alignment"
REPLICATES = 5000
SEED = 2026090301


def read_tsv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def score_metrics(labels: np.ndarray, values: np.ndarray, probability: bool) -> dict[str, float]:
    return {
        "auprc": float(average_precision_score(labels, values)),
        "auroc": float(roc_auc_score(labels, values)),
        "log_loss": float(log_loss(labels, values, labels=[0, 1])) if probability else float("nan"),
    }


def fit_oof(values: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> np.ndarray:
    predicted = np.zeros(len(labels), dtype=np.float64)
    for fold in range(5):
        held_out = folds == fold
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=20260830),
        )
        model.fit(values[~held_out], labels[~held_out])
        predicted[held_out] = model.predict_proba(values[held_out])[:, 1]
    return predicted


def bootstrap(
    labels: np.ndarray,
    clusters: np.ndarray,
    direct: dict[str, np.ndarray],
    probabilities: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    comparisons = (
        ("ipsae_min", "iptm", "auprc", direct),
        ("ipsae_min", "iptm", "auroc", direct),
        ("confidence3_plus_ipsae_min", "confidence3", "auprc", probabilities),
        ("confidence3_plus_ipsae_min", "confidence3", "auroc", probabilities),
        ("confidence3_plus_ipsae_min", "confidence3", "negative_log_loss", probabilities),
    )
    functions = {"auprc": average_precision_score, "auroc": roc_auc_score}
    unique_clusters = np.asarray(sorted(set(clusters.tolist())))
    cluster_indices = {
        cluster: np.flatnonzero(clusters == cluster) for cluster in unique_clusters
    }
    rng = np.random.default_rng(SEED)
    samples: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for _ in range(REPLICATES):
        sampled = rng.choice(unique_clusters, len(unique_clusters), replace=True)
        indices = np.concatenate([cluster_indices[cluster] for cluster in sampled])
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) != 2:
            continue
        for candidate, reference, metric, source in comparisons:
            if metric == "negative_log_loss":
                delta = log_loss(
                    sampled_labels, source[reference][indices], labels=[0, 1]
                ) - log_loss(
                    sampled_labels, source[candidate][indices], labels=[0, 1]
                )
            else:
                function = functions[metric]
                delta = function(sampled_labels, source[candidate][indices]) - function(
                    sampled_labels, source[reference][indices]
                )
            samples[(candidate, reference, metric)].append(float(delta))

    rows = []
    for candidate, reference, metric, source in comparisons:
        if metric == "negative_log_loss":
            observed = log_loss(labels, source[reference], labels=[0, 1]) - log_loss(
                labels, source[candidate], labels=[0, 1]
            )
        else:
            function = functions[metric]
            observed = function(labels, source[candidate]) - function(labels, source[reference])
        values = np.asarray(samples[(candidate, reference, metric)])
        rows.append(
            {
                "candidate": candidate,
                "reference": reference,
                "metric": metric,
                "observed_delta": float(observed),
                "ci_2p5": float(np.quantile(values, 0.025)),
                "ci_97p5": float(np.quantile(values, 0.975)),
                "probability_delta_gt_0": float(np.mean(values > 0)),
                "replicates": len(values),
            }
        )
    return rows


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Output requires review: {OUTPUT}")
    train = read_tsv(TRAIN)
    sample_ids = [row["sample_id"] for row in train]
    labels = np.asarray([row["label"] == "positive" for row in train], dtype=np.int8)
    clusters = np.asarray([row["target_cluster_id"] for row in train])
    features = {row["sample_id"]: row for row in read_tsv(FEATURES)}
    confidence = {
        sample_id: row
        for row in read_tsv(COHORT)
        for sample_id in row["sample_ids"].split("|")
    }
    folds_by_id = {row["sample_id"]: int(row["cv_fold"]) for row in read_tsv(FOLDS)}
    legacy = {row["sample_id"]: row for row in read_tsv(LEGACY_OOF)}
    if len(train) != 2881 or int(labels.sum()) != 756:
        raise ValueError("Frozen cohort count mismatch")
    if set(sample_ids) != set(features) or set(sample_ids) != set(folds_by_id):
        raise ValueError("Feature or fold coverage mismatch")

    ipsae_min_by_id = {
        sample_id: min(
            float(features[sample_id]["ipsae_target_to_binder"]),
            float(features[sample_id]["ipsae_binder_to_target"]),
        )
        for sample_id in sample_ids
    }
    multichain_rows = read_tsv(MULTICHAIN)
    grouped_multichain: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in multichain_rows:
        grouped_multichain[row["sample_id"]].append(row)
    for sample_id, rows in grouped_multichain.items():
        selected = max(rows, key=lambda row: float(row["weaker_link"]))
        if selected["aggregation_selected"] != "1":
            raise ValueError(f"Multichain selection flag mismatch: {sample_id}")
        ipsae_min_by_id[sample_id] = float(selected["weaker_link"])

    folds = np.asarray([folds_by_id[sample_id] for sample_id in sample_ids])
    ipsae_min = np.asarray([ipsae_min_by_id[sample_id] for sample_id in sample_ids])
    iptm = np.asarray([float(confidence[sample_id]["iptm"]) for sample_id in sample_ids])
    confidence3_values = np.column_stack(
        [
            [float(confidence[sample_id]["iptm"]) for sample_id in sample_ids],
            [-float(confidence[sample_id]["interface_pae_min"]) for sample_id in sample_ids],
            [float(confidence[sample_id]["mean_atom_plddt"]) for sample_id in sample_ids],
        ]
    )
    confidence3 = fit_oof(confidence3_values, labels, folds)
    confidence3_plus_ipsae = fit_oof(
        np.column_stack([confidence3_values, ipsae_min]), labels, folds
    )
    legacy_confidence3 = np.asarray(
        [float(legacy[sample_id]["probability_legacy_confidence3_logistic"]) for sample_id in sample_ids]
    )
    maximum_reproduction_difference = float(np.max(np.abs(confidence3 - legacy_confidence3)))
    if maximum_reproduction_difference > 1e-12:
        raise ValueError(
            f"confidence3 OOF reproduction mismatch: {maximum_reproduction_difference}"
        )

    direct = {"ipsae_min": ipsae_min, "iptm": iptm}
    probabilities = {
        "confidence3": confidence3,
        "confidence3_plus_ipsae_min": confidence3_plus_ipsae,
    }
    summary = []
    for method, values in direct.items():
        summary.append(
            {
                "method": method,
                "evaluation": "direct_score",
                **score_metrics(labels, values, False),
            }
        )
    for method, values in probabilities.items():
        summary.append(
            {
                "method": method,
                "evaluation": "five_fold_grouped_oof",
                **score_metrics(labels, values, True),
            }
        )

    OUTPUT.mkdir(parents=True)
    write_tsv(OUTPUT / "summary.tsv", summary)
    write_tsv(OUTPUT / "target_cluster_bootstrap.tsv", bootstrap(labels, clusters, direct, probabilities))
    prediction_rows = [
        {
            "sample_id": sample_id,
            "label": int(labels[index]),
            "target_cluster_id": clusters[index],
            "score_ipsae_min": ipsae_min[index],
            "score_iptm": iptm[index],
            "probability_confidence3": confidence3[index],
            "probability_confidence3_plus_ipsae_min": confidence3_plus_ipsae[index],
        }
        for index, sample_id in enumerate(sample_ids)
    ]
    with gzip.open(OUTPUT / "oof_predictions.tsv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(prediction_rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(prediction_rows)
    (OUTPUT / "metadata.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "sample_count": len(labels),
                "positive_count": int(labels.sum()),
                "target_cluster_count": len(set(clusters.tolist())),
                "bootstrap_replicates": REPLICATES,
                "bootstrap_seed": SEED,
                "multichain_sample_count": len(grouped_multichain),
                "confidence3_maximum_reproduction_difference": maximum_reproduction_difference,
                "project_validation_accessed": False,
                "sealed_test_accessed": False,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
