#!/usr/bin/env python3
"""Audit complete matched OOF runs and compare with the frozen AF3 pair anchor."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_representation_benchmark_completion_v1 import (
    ADAPTERS, BASE, FOLDS, RUN, SEEDS, read_tsv, sha256, unique, verify, write_tsv,
)
from scripts.run_representation_benchmark_completion_v1 import valid_receipt
from scripts.summarize_representation_source_v1 import metrics


def collapse(members, original, fold_by_id):
    values = {sid: {} for sid in original}
    for fold, seed, rows in members:
        indexed = unique(rows)
        expected = {sid for sid in original if fold_by_id[sid] == fold}
        if set(indexed) != expected:
            raise ValueError(f"Wrong OOF member coverage: fold={fold} seed={seed}")
        for sid, record in indexed.items():
            if seed in values[sid] or int(float(record["label"])) != int(original[sid]["label"] == "positive"):
                raise ValueError(f"Wrong OOF identity or duplicate seed: {sid}")
            if record["target_sequence_id"] != original[sid]["target_sequence_id"]:
                raise ValueError(f"Wrong OOF target: {sid}")
            p = float(record["probability"])
            if not np.isfinite(p) or not 0 <= p <= 1:
                raise ValueError(f"Invalid OOF probability: {sid}")
            values[sid][seed] = p
    if any(set(v) != set(SEEDS) for v in values.values()):
        raise ValueError("Each sample must have exactly the two frozen seeds")
    return [{"sample_id": sid, "label": int(original[sid]["label"] == "positive"),
             "target_name": original[sid]["target_name"],
             "target_sequence_id": original[sid]["target_sequence_id"],
             "target_cluster_id": original[sid]["target_cluster_id"],
             "outer_fold": fold_by_id[sid], "probability": float(np.mean(list(values[sid].values())))}
            for sid in sorted(original)]


def macro_metrics(rows):
    targets = sorted({r["target_sequence_id"] for r in rows})
    per_target = []
    for target in targets:
        selected = [r for r in rows if r["target_sequence_id"] == target]
        if len({r["label"] for r in selected}) == 2:
            per_target.append({"target_sequence_id": target, **metrics(selected)})
    return {"macro_target_auprc": float(np.mean([r["auprc"] for r in per_target])),
            "eligible_targets": len(per_target)}, per_target


def bootstrap_pair(arguments):
    name, rows, reference = arguments
    y = np.asarray([r["label"] for r in rows])
    p = np.asarray([r["probability"] for r in rows])
    q = np.asarray([r["probability"] for r in reference])
    groups = np.asarray([r["target_cluster_id"] for r in rows])
    unique_groups = sorted(set(groups))
    blocks = [np.flatnonzero(groups == group) for group in unique_groups]
    rng = np.random.default_rng(20260905)
    deltas = []
    for _ in range(5000):
        indices = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        labels = y[indices]
        if len(np.unique(labels)) != 2:
            continue
        deltas.append([average_precision_score(labels, p[indices]) - average_precision_score(labels, q[indices]),
                       roc_auc_score(labels, p[indices]) - roc_auc_score(labels, q[indices])])
    values = np.asarray(deltas)
    result = []
    for index, (metric, score) in enumerate((("auprc", average_precision_score), ("auroc", roc_auc_score))):
        result.append({"candidate": name, "reference": "af3_mature_pair", "metric": metric,
                       "observed_delta": float(score(y, p) - score(y, q)),
                       "ci_2p5": float(np.quantile(values[:, index], 0.025)),
                       "ci_97p5": float(np.quantile(values[:, index], 0.975)),
                       "valid_replicates": len(values), "cluster_count": len(unique_groups)})
    return result


def main():
    verify()
    output = RUN / "summary"
    if output.exists():
        raise FileExistsError(f"Summary already exists; review before replacing: {output}")
    original = unique(read_tsv(BASE / "train.tsv.gz"))
    folds = {r["sample_id"]: int(r["cv_fold"]) for r in read_tsv(FOLDS)}
    tasks = read_tsv(RUN / "manifest.tsv")
    if len(tasks) != 60:
        raise ValueError("Expected 60 model fits")
    for row in tasks:
        valid_receipt(row)
    ensembles, per_run, parameter_counts = {}, [], {}
    for adapter in ADAPTERS:
        members, counts = [], set()
        for row in tasks:
            if row["adapter"] != adapter:
                continue
            path = RUN / "models" / row["run_name"]
            predictions = read_tsv(path / "outer_holdout_predictions.tsv.gz")
            members.append((int(row["fold"]), int(row["seed"]), predictions))
            per_run.append({"adapter": adapter, "fold": row["fold"], "seed": row["seed"], **metrics(predictions)})
            counts.add(json.loads((path / "config.json").read_text()).get("trainable_parameters"))
        if len(counts) != 1:
            raise ValueError(f"Adapter capacity changed across runs: {adapter}")
        parameter_counts[adapter] = counts.pop()
        ensembles[adapter] = collapse(members, original, folds)
    reference_members, reference_hashes = [], {}
    for fold in range(5):
        for seed in SEEDS:
            directory = ROOT / "runs/v2/representation_source_v1/filtered_oof" / f"af2_af3_repr_v1_af3_af3_pair_fold{fold}_seed{seed}"
            path = directory / "outer_holdout_predictions.tsv.gz"
            reference_members.append((fold, seed, read_tsv(path)))
            reference_hashes[str(path)] = sha256(path)
    ensembles["af3_mature_pair"] = collapse(reference_members, original, folds)
    parameter_counts["af3_mature_pair"] = 17153
    ensembles["dmasif_multiview"] = [
        {**a, "probability": (a["probability"] + b["probability"]) / 2}
        for a, b in zip(ensembles["dmasif_global"], ensembles["dmasif_region"], strict=True)
    ]
    parameter_counts["dmasif_multiview"] = parameter_counts["dmasif_global"] + parameter_counts["dmasif_region"]
    summary, target_rows, prediction_rows = [], [], []
    for adapter, rows in ensembles.items():
        macro, targets = macro_metrics(rows)
        summary.append({"adapter": adapter, "trainable_parameters_per_seed": parameter_counts[adapter],
                        "ensemble_members_per_outer_fold": 4 if adapter == "dmasif_multiview" else 2,
                        **metrics(rows), **macro})
        target_rows.extend({"adapter": adapter, **r} for r in targets)
        prediction_rows.extend({"adapter": adapter, **r} for r in rows)
    reference = ensembles["af3_mature_pair"]
    arguments = [(name, rows, reference) for name, rows in ensembles.items() if name != "af3_mature_pair"]
    with ProcessPoolExecutor(max_workers=7) as pool:
        bootstrap_rows = [row for result in pool.map(bootstrap_pair, arguments) for row in result]
    output.mkdir()
    write_tsv(output / "metrics.tsv", summary)
    write_tsv(output / "per_run_metrics.tsv", per_run)
    write_tsv(output / "per_target_metrics.tsv", target_rows)
    write_tsv(output / "oof_predictions.tsv.gz", prediction_rows)
    write_tsv(output / "paired_target_cluster_bootstrap.tsv", bootstrap_rows)
    metadata = {"status": "complete", "sample_count": len(original), "fits": 60,
                "protocol_sha256": sha256(RUN / "protocol.json"), "reference_sha256": reference_hashes,
                "project_validation_accessed": False, "sealed_test_accessed": False,
                "manuscript_automatically_changed": False}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
