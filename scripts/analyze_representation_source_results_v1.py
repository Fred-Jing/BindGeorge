#!/usr/bin/env python3
"""Audit and compare completed AF2/AF3 representation-source experiments."""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "runs/v2/representation_source_v1"
OUTPUT = RUN_ROOT / "summary"
FILTERED_SPLIT = (
    PROJECT_ROOT
    / "processed/v1/training_10x/cohorts/ipae_min_le_2p5/known_target/train.tsv.gz"
)
FULL_VALIDATION = (
    PROJECT_ROOT / "processed/v1/training_10x/splits/known_target/validation.tsv.gz"
)
CONFIDENCE_TABLE = PROJECT_ROOT / "runs/af3_v1/full/quality/confidence_by_complex.tsv.gz"
FILTERED_VARIANTS = ("af3_single", "af3_pair", "af3_single_pair")
FULL_VARIANTS = (
    "confidence",
    "af3_single",
    "af3_pair",
    "af3_single_pair",
    "af3_pair_confidence",
)
FILTERED_SEEDS = (20260823, 20260824)
FULL_SEEDS = (20260727, 20260728, 20260729, 20260730, 20260731)
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 2026090201


def read_tsv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def binary_label(value: str) -> int:
    if value in {"1", "1.0", "positive"}:
        return 1
    if value in {"0", "0.0", "negative"}:
        return 0
    raise ValueError(f"Unexpected binary label: {value}")


def metric_values(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "auprc": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def load_split(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    rows = read_tsv(path)
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return (
        sample_ids,
        np.asarray([binary_label(row["label"]) for row in rows], dtype=np.int8),
        np.asarray([row["target_sequence_id"] for row in rows]),
        np.asarray([row["target_cluster_id"] for row in rows]),
        {row["target_sequence_id"]: row["target_name"] for row in rows},
    )


def read_predictions(path: Path) -> dict[str, dict[str, str]]:
    rows = read_tsv(path)
    records = {row["sample_id"]: row for row in rows}
    if len(records) != len(rows):
        raise ValueError(f"Duplicate prediction sample IDs in {path}")
    return records


def audit_config(run_dir: Path, project_validation_expected: bool) -> None:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    for payload_name, payload in (("config", config), ("metrics", metrics)):
        if payload.get("sealed_test_accessed") is not False:
            raise ValueError(f"Sealed-test access violation in {run_dir} {payload_name}")
        if payload.get("project_validation_accessed", False) is not project_validation_expected:
            raise ValueError(
                f"Project-validation access mismatch in {run_dir} {payload_name}"
            )
    if config.get("evaluate_test") is not False:
        raise ValueError(f"evaluate_test must be false: {run_dir}")


def load_filtered(
    sample_ids: list[str], labels: np.ndarray
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    expected = set(sample_ids)
    label_by_sample = dict(zip(sample_ids, labels.tolist(), strict=True))
    ensembles: dict[str, np.ndarray] = {}
    seed_rows: list[dict[str, object]] = []
    for source in ("af3", "af2"):
        for variant in FILTERED_VARIANTS:
            seed_arrays = []
            for seed in FILTERED_SEEDS:
                records: dict[str, dict[str, str]] = {}
                for fold in range(5):
                    run_dir = (
                        RUN_ROOT
                        / "filtered_oof"
                        / f"af2_af3_repr_v1_{source}_{variant}_fold{fold}_seed{seed}"
                    )
                    audit_config(run_dir, project_validation_expected=False)
                    fold_records = read_predictions(
                        run_dir / "outer_holdout_predictions.tsv.gz"
                    )
                    overlap = set(records) & set(fold_records)
                    if overlap:
                        raise ValueError(
                            f"Duplicate OOF coverage for {source}/{variant}/{seed}"
                        )
                    records.update(fold_records)
                if set(records) != expected:
                    raise ValueError(
                        f"OOF sample mismatch for {source}/{variant}/{seed}: "
                        f"{len(records)} != {len(expected)}"
                    )
                observed_labels = np.asarray(
                    [binary_label(records[sample_id]["label"]) for sample_id in sample_ids]
                )
                if not np.array_equal(observed_labels, labels):
                    raise ValueError(f"OOF label mismatch for {source}/{variant}/{seed}")
                probabilities = np.asarray(
                    [float(records[sample_id]["probability"]) for sample_id in sample_ids]
                )
                seed_arrays.append(probabilities)
                seed_rows.append(
                    {
                        "branch": "filtered_oof",
                        "source": source,
                        "variant": variant,
                        "seed": seed,
                        "count": len(labels),
                        "positive": int(labels.sum()),
                        **metric_values(labels, probabilities),
                    }
                )
                if any(
                    binary_label(records[sample_id]["label"]) != label_by_sample[sample_id]
                    for sample_id in sample_ids
                ):
                    raise ValueError(f"Split-label mismatch for {source}/{variant}/{seed}")
            ensembles[f"{source}_{variant.removeprefix('af3_')}"] = np.mean(
                np.stack(seed_arrays), axis=0
            )
    return ensembles, seed_rows


def load_full(
    sample_ids: list[str], labels: np.ndarray
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    expected = set(sample_ids)
    ensembles: dict[str, np.ndarray] = {}
    seed_rows: list[dict[str, object]] = []
    for variant in FULL_VARIANTS:
        seed_arrays = []
        for seed in FULL_SEEDS:
            run_dir = (
                RUN_ROOT / "full_af3" / f"af3_full_repr_v1_{variant}_seed{seed}"
            )
            audit_config(run_dir, project_validation_expected=True)
            records = read_predictions(run_dir / "validation_predictions.tsv.gz")
            if set(records) != expected:
                raise ValueError(
                    f"Full-validation sample mismatch for {variant}/{seed}: "
                    f"{len(records)} != {len(expected)}"
                )
            observed_labels = np.asarray(
                [binary_label(records[sample_id]["label"]) for sample_id in sample_ids]
            )
            if not np.array_equal(observed_labels, labels):
                raise ValueError(f"Full-validation label mismatch for {variant}/{seed}")
            probabilities = np.asarray(
                [float(records[sample_id]["probability"]) for sample_id in sample_ids]
            )
            seed_arrays.append(probabilities)
            seed_rows.append(
                {
                    "branch": "full_af3_validation",
                    "source": "af3",
                    "variant": variant,
                    "seed": seed,
                    "count": len(labels),
                    "positive": int(labels.sum()),
                    **metric_values(labels, probabilities),
                }
            )
        ensembles[variant.removeprefix("af3_")] = np.mean(
            np.stack(seed_arrays), axis=0
        )
    return ensembles, seed_rows


def bootstrap_deltas(
    branch: str,
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    cluster_ids: np.ndarray,
    comparisons: list[tuple[str, str]],
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    clusters = np.asarray(sorted(set(cluster_ids.tolist())))
    cluster_indices = {cluster: np.flatnonzero(cluster_ids == cluster) for cluster in clusters}
    metric_names = ("auprc", "auroc", "log_loss", "brier_score")
    samples: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate([cluster_indices[cluster] for cluster in sampled_clusters])
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        replicate_metrics = {
            model: metric_values(sampled_labels, values[indices])
            for model, values in probabilities.items()
        }
        for candidate, reference in comparisons:
            for metric in metric_names:
                candidate_value = replicate_metrics[candidate][metric]
                reference_value = replicate_metrics[reference][metric]
                delta = (
                    reference_value - candidate_value
                    if metric in {"log_loss", "brier_score"}
                    else candidate_value - reference_value
                )
                samples[(candidate, reference, metric)].append(delta)

    observed_metrics = {
        model: metric_values(labels, values) for model, values in probabilities.items()
    }
    rows = []
    for candidate, reference in comparisons:
        for metric in metric_names:
            candidate_value = observed_metrics[candidate][metric]
            reference_value = observed_metrics[reference][metric]
            observed = (
                reference_value - candidate_value
                if metric in {"log_loss", "brier_score"}
                else candidate_value - reference_value
            )
            values = np.asarray(samples[(candidate, reference, metric)])
            rows.append(
                {
                    "branch": branch,
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    "delta_definition": (
                        "reference_minus_candidate"
                        if metric in {"log_loss", "brier_score"}
                        else "candidate_minus_reference"
                    ),
                    "positive_delta_favors_candidate": True,
                    "observed_delta": observed,
                    "bootstrap_mean_delta": float(values.mean()),
                    "ci_2p5": float(np.quantile(values, 0.025)),
                    "ci_97p5": float(np.quantile(values, 0.975)),
                    "probability_delta_gt_0": float(np.mean(values > 0.0)),
                    "target_clusters": len(clusters),
                    "valid_replicates": len(values),
                }
            )
    return rows


def target_macro_rows(
    branch: str,
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    target_ids: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for model, values in probabilities.items():
        by_metric: dict[str, list[float]] = defaultdict(list)
        eligible = 0
        for target_id in sorted(set(target_ids.tolist())):
            indices = np.flatnonzero(target_ids == target_id)
            target_labels = labels[indices]
            if len(np.unique(target_labels)) < 2:
                continue
            eligible += 1
            for metric, value in metric_values(target_labels, values[indices]).items():
                by_metric[metric].append(value)
        rows.append(
            {
                "branch": branch,
                "model": model,
                "targets_total": len(set(target_ids.tolist())),
                "targets_with_both_labels": eligible,
                **{
                    f"macro_{metric}": float(np.mean(metric_values_))
                    for metric, metric_values_ in by_metric.items()
                },
            }
        )
    return rows


def prediction_rows(
    sample_ids: list[str],
    labels: np.ndarray,
    target_ids: np.ndarray,
    cluster_ids: np.ndarray,
    target_names: dict[str, str],
    probabilities: dict[str, np.ndarray],
    extra: dict[str, np.ndarray] | None = None,
) -> list[dict[str, object]]:
    rows = []
    for index, sample_id in enumerate(sample_ids):
        row: dict[str, object] = {
            "sample_id": sample_id,
            "target_name": target_names[target_ids[index]],
            "target_sequence_id": target_ids[index],
            "target_cluster_id": cluster_ids[index],
            "label": int(labels[index]),
        }
        if extra:
            row.update({name: values[index] for name, values in extra.items()})
        row.update(
            {
                f"probability_{model}": float(values[index])
                for model, values in probabilities.items()
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    filtered = load_split(FILTERED_SPLIT)
    full = load_split(FULL_VALIDATION)
    filtered_probabilities, filtered_seed_rows = load_filtered(filtered[0], filtered[1])
    full_probabilities, full_seed_rows = load_full(full[0], full[1])

    confidence = {
        row["sample_ids"]: float(row["interface_pae_min"])
        for row in read_tsv(CONFIDENCE_TABLE)
    }
    full_ipae = np.asarray([confidence[sample_id] for sample_id in full[0]])
    gated = full_ipae <= 2.5

    filtered_comparisons = [
        ("af2_single", "af3_single"),
        ("af2_pair", "af3_pair"),
        ("af2_single_pair", "af3_single_pair"),
        ("af3_pair", "af3_single"),
        ("af3_single_pair", "af3_pair"),
        ("af2_pair", "af2_single"),
        ("af2_single_pair", "af2_pair"),
    ]
    full_comparisons = [
        ("single", "confidence"),
        ("pair", "confidence"),
        ("single_pair", "confidence"),
        ("pair", "single"),
        ("single_pair", "single"),
        ("single_pair", "pair"),
        ("pair_confidence", "pair"),
    ]
    bootstrap_rows = bootstrap_deltas(
        "filtered_oof",
        filtered[1],
        filtered_probabilities,
        filtered[3],
        filtered_comparisons,
        BOOTSTRAP_SEED,
    )
    bootstrap_rows.extend(
        bootstrap_deltas(
            "full_af3_validation",
            full[1],
            full_probabilities,
            full[3],
            full_comparisons,
            BOOTSTRAP_SEED + 1,
        )
    )
    bootstrap_rows.extend(
        bootstrap_deltas(
            "full_af3_validation_ipae_le_2p5",
            full[1][gated],
            {name: values[gated] for name, values in full_probabilities.items()},
            full[3][gated],
            full_comparisons,
            BOOTSTRAP_SEED + 2,
        )
    )

    write_tsv(OUTPUT / "individual_seed_metrics.tsv", filtered_seed_rows + full_seed_rows)
    write_tsv(OUTPUT / "paired_target_cluster_bootstrap.tsv", bootstrap_rows)
    write_tsv(
        OUTPUT / "target_macro_metrics.tsv",
        target_macro_rows(
            "filtered_oof", filtered[1], filtered_probabilities, filtered[2]
        )
        + target_macro_rows(
            "full_af3_validation", full[1], full_probabilities, full[2]
        )
        + target_macro_rows(
            "full_af3_validation_ipae_le_2p5",
            full[1][gated],
            {name: values[gated] for name, values in full_probabilities.items()},
            full[2][gated],
        ),
    )
    write_tsv(
        OUTPUT / "filtered_oof_predictions.tsv.gz",
        prediction_rows(*filtered, probabilities=filtered_probabilities),
    )
    write_tsv(
        OUTPUT / "full_af3_validation_predictions.tsv.gz",
        prediction_rows(
            *full,
            probabilities=full_probabilities,
            extra={"interface_pae_min": full_ipae},
        ),
    )

    metadata = {
        "status": "complete",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "target_cluster_id",
        "filtered_models": len(filtered_probabilities),
        "filtered_samples": len(filtered[0]),
        "filtered_positive": int(filtered[1].sum()),
        "filtered_target_sequences": len(set(filtered[2].tolist())),
        "filtered_target_clusters": len(set(filtered[3].tolist())),
        "full_models": len(full_probabilities),
        "full_validation_samples": len(full[0]),
        "full_validation_positive": int(full[1].sum()),
        "full_validation_target_sequences": len(set(full[2].tolist())),
        "full_validation_target_clusters": len(set(full[3].tolist())),
        "full_validation_ipae_le_2p5_samples": int(gated.sum()),
        "full_validation_ipae_le_2p5_positive": int(full[1][gated].sum()),
        "project_validation_accessed_by_filtered_oof": False,
        "project_validation_accessed_by_full_experiment": True,
        "sealed_test_accessed": False,
    }
    (OUTPUT / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
