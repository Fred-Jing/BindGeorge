#!/usr/bin/env python3
"""Compare AF3 confidence-refined and mature representations by grouped OOF."""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT = PROJECT_ROOT / "processed/v1/training_10x/cohorts/ipae_min_le_2p5/known_target/train.tsv.gz"
MATURE_ROOT = PROJECT_ROOT / "runs/v2/representation_source_v1/filtered_oof"
REFINED_ROOT = PROJECT_ROOT / "runs/v2/af3_refined_representation_v1/oof"
OUTPUT = PROJECT_ROOT / "runs/v2/af3_refined_representation_v1/summary"
VARIANTS = ("af3_single", "af3_pair", "af3_single_pair")
SEEDS = (20260823, 20260824)
FOLDS = range(5)
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 2026090301


def read_tsv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def binary_label(value: str) -> int:
    if value in {"1", "1.0", "positive"}:
        return 1
    if value in {"0", "0.0", "negative"}:
        return 0
    raise ValueError(f"Unexpected label: {value}")


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "auprc": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def audit_run(run_dir: Path) -> None:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    for name, payload in (("config", config), ("metrics", result)):
        if payload.get("sealed_test_accessed") is not False:
            raise ValueError(f"Sealed-test access in {run_dir} {name}")
        if payload.get("project_validation_accessed") is not False:
            raise ValueError(f"Project-validation access in {run_dir} {name}")
    if config.get("evaluate_test") is not False:
        raise ValueError(f"evaluate_test enabled in {run_dir}")


def run_dir(source: str, variant: str, fold: int, seed: int) -> Path:
    if source == "mature":
        name = f"af2_af3_repr_v1_af3_{variant}_fold{fold}_seed{seed}"
        return MATURE_ROOT / name
    name = f"af3_refined_{variant}_fold{fold}_seed{seed}"
    return REFINED_ROOT / name


def load_oof(
    source: str, variant: str, sample_ids: list[str], labels: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    expected = set(sample_ids)
    seed_probabilities = []
    seed_rows = []
    for seed in SEEDS:
        records: dict[str, dict[str, str]] = {}
        for fold in FOLDS:
            directory = run_dir(source, variant, fold, seed)
            audit_run(directory)
            fold_rows = read_tsv(directory / "outer_holdout_predictions.tsv.gz")
            overlap = set(records) & {row["sample_id"] for row in fold_rows}
            if overlap:
                raise ValueError(f"Duplicate OOF samples for {source}/{variant}/{seed}")
            records.update({row["sample_id"]: row for row in fold_rows})
        if set(records) != expected:
            raise ValueError(f"OOF coverage mismatch for {source}/{variant}/{seed}: {len(records)}")
        observed_labels = np.asarray([binary_label(records[sample]["label"]) for sample in sample_ids])
        if not np.array_equal(observed_labels, labels):
            raise ValueError(f"OOF label mismatch for {source}/{variant}/{seed}")
        probabilities = np.asarray([float(records[sample]["probability"]) for sample in sample_ids])
        seed_probabilities.append(probabilities)
        seed_rows.append({"source": source, "variant": variant, "seed": seed, **metrics(labels, probabilities)})
    return np.mean(np.stack(seed_probabilities), axis=0), seed_rows


def macro_metrics(
    labels: np.ndarray, probabilities: np.ndarray, targets: np.ndarray,
) -> dict[str, float | int]:
    values: defaultdict[str, list[float]] = defaultdict(list)
    eligible = 0
    for target in sorted(set(targets.tolist())):
        indices = np.flatnonzero(targets == target)
        if len(np.unique(labels[indices])) < 2:
            continue
        eligible += 1
        for key, value in metrics(labels[indices], probabilities[indices]).items():
            values[key].append(value)
    return {
        "targets_with_both_labels": eligible,
        **{f"macro_{key}": float(np.mean(items)) for key, items in values.items()},
    }


def bootstrap(
    labels: np.ndarray, candidate: np.ndarray, reference: np.ndarray,
    clusters: np.ndarray, variant: str,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + VARIANTS.index(variant))
    unique = np.asarray(sorted(set(clusters.tolist())))
    indices = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    values: defaultdict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([indices[cluster] for cluster in sampled])
        sampled_labels = labels[selected]
        if len(np.unique(sampled_labels)) < 2:
            continue
        candidate_metrics = metrics(sampled_labels, candidate[selected])
        reference_metrics = metrics(sampled_labels, reference[selected])
        for metric in candidate_metrics:
            delta = (
                reference_metrics[metric] - candidate_metrics[metric]
                if metric in {"log_loss", "brier_score"}
                else candidate_metrics[metric] - reference_metrics[metric]
            )
            values[metric].append(delta)
    candidate_metrics = metrics(labels, candidate)
    reference_metrics = metrics(labels, reference)
    rows = []
    for metric, samples in values.items():
        observed = (
            reference_metrics[metric] - candidate_metrics[metric]
            if metric in {"log_loss", "brier_score"}
            else candidate_metrics[metric] - reference_metrics[metric]
        )
        array = np.asarray(samples)
        rows.append({
            "variant": variant,
            "candidate": "refined",
            "reference": "mature",
            "metric": metric,
            "positive_delta_favors_refined": True,
            "observed_delta": observed,
            "ci_2p5": float(np.quantile(array, 0.025)),
            "ci_97p5": float(np.quantile(array, 0.975)),
            "probability_delta_gt_0": float(np.mean(array > 0)),
            "target_clusters": len(unique),
            "valid_replicates": len(array),
        })
    return rows


def main() -> None:
    split_rows = read_tsv(SPLIT)
    sample_ids = [row["sample_id"] for row in split_rows]
    labels = np.asarray([binary_label(row["label"]) for row in split_rows])
    targets = np.asarray([row["target_sequence_id"] for row in split_rows])
    clusters = np.asarray([row["target_cluster_id"] for row in split_rows])
    summary_rows = []
    seed_rows = []
    bootstrap_rows = []
    probabilities: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for variant in VARIANTS:
        for source in ("mature", "refined"):
            ensemble, source_seed_rows = load_oof(source, variant, sample_ids, labels)
            probabilities[variant][source] = ensemble
            seed_rows.extend(source_seed_rows)
            summary_rows.append({
                "source": source,
                "variant": variant,
                "count": len(labels),
                "positive": int(labels.sum()),
                **metrics(labels, ensemble),
                **macro_metrics(labels, ensemble, targets),
            })
        bootstrap_rows.extend(bootstrap(
            labels, probabilities[variant]["refined"], probabilities[variant]["mature"],
            clusters, variant,
        ))
    write_tsv(OUTPUT / "ensemble_metrics.tsv", summary_rows)
    write_tsv(OUTPUT / "seed_metrics.tsv", seed_rows)
    write_tsv(OUTPUT / "bootstrap_deltas.tsv", bootstrap_rows)
    payload = {
        "status": "complete",
        "question": "AF3 confidence-refined versus mature representation on matched grouped OOF",
        "samples": len(labels),
        "positive": int(labels.sum()),
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "folds": len(tuple(FOLDS)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "sealed_test_accessed": False,
        "project_validation_accessed": False,
    }
    write_json(OUTPUT / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
