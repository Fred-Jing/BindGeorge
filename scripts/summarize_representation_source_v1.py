#!/usr/bin/env python3
"""Summarize matched AF2/AF3 OOF and full-cohort AF3 representation runs."""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "runs/v2/representation_source_v1"
OUTPUT = RUN_ROOT / "summary"
FILTERED_VARIANTS = ("af3_single", "af3_pair", "af3_single_pair")
FULL_VARIANTS = ("confidence", "af3_single", "af3_pair", "af3_single_pair", "af3_pair_confidence")
FILTERED_SEEDS = (20260823, 20260824)
FULL_SEEDS = (20260727, 20260728, 20260729, 20260730, 20260731)


def read_tsv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def metrics(records: list[dict[str, object]]) -> dict[str, object]:
    labels = np.asarray([int(row["label"]) for row in records], dtype=np.int64)
    probabilities = np.asarray(
        [float(row["probability"]) for row in records], dtype=np.float64
    )
    output: dict[str, object] = {
        "count": len(records),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "positive_rate": float(labels.mean()) if len(labels) else math.nan,
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }
    if len(np.unique(labels)) == 2:
        output["auprc"] = float(average_precision_score(labels, probabilities))
        output["auroc"] = float(roc_auc_score(labels, probabilities))
    else:
        output["auprc"] = output["auroc"] = math.nan
    return output


def ensemble(paths: list[Path]) -> list[dict[str, object]]:
    values: dict[str, list[float]] = defaultdict(list)
    identity: dict[str, dict[str, object]] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in read_tsv(path):
            sample_id = row["sample_id"]
            label = int(float(row["label"]))
            current = {
                "sample_id": sample_id,
                "target_name": row["target_name"],
                "target_sequence_id": row["target_sequence_id"],
                "label": label,
            }
            if sample_id in identity and identity[sample_id] != current:
                raise ValueError(f"Prediction identity mismatch: {sample_id}")
            identity[sample_id] = current
            values[sample_id].append(float(row["probability"]))
    expected = len(paths)
    if any(len(item) != expected for item in values.values()):
        raise ValueError("Ensemble members do not cover identical sample IDs")
    return [
        {**identity[sample_id], "probability": float(np.mean(values[sample_id]))}
        for sample_id in sorted(values)
    ]


def filtered_summary() -> list[dict[str, object]]:
    rows = []
    for source in ("af3", "af2"):
        for variant in FILTERED_VARIANTS:
            paths = [
                RUN_ROOT
                / "filtered_oof"
                / f"af2_af3_repr_v1_{source}_{variant}_fold{fold}_seed{seed}"
                / "outer_holdout_predictions.tsv.gz"
                for seed in FILTERED_SEEDS
                for fold in range(5)
            ]
            # Each sample appears in one outer fold for each seed, not in all ten files.
            by_sample: dict[str, list[dict[str, object]]] = defaultdict(list)
            for path in paths:
                for record in read_tsv(path):
                    by_sample[record["sample_id"]].append(
                        {
                            "sample_id": record["sample_id"],
                            "target_name": record["target_name"],
                            "target_sequence_id": record["target_sequence_id"],
                            "label": int(float(record["label"])),
                            "probability": float(record["probability"]),
                        }
                    )
            records = []
            for sample_id in sorted(by_sample):
                members = by_sample[sample_id]
                if len(members) != len(FILTERED_SEEDS):
                    raise ValueError(f"OOF seed coverage mismatch: {sample_id}")
                records.append(
                    {
                        **{key: members[0][key] for key in ("sample_id", "target_name", "target_sequence_id", "label")},
                        "probability": float(np.mean([row["probability"] for row in members])),
                    }
                )
            rows.append({"source": source, "variant": variant, **metrics(records)})
    return rows


def full_summary() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    confidence = {
        row["sample_ids"]: float(row["interface_pae_min"])
        for row in read_tsv(
            PROJECT_ROOT / "runs/af3_v1/full/quality/confidence_by_complex.tsv.gz"
        )
    }
    summary = []
    strata = []
    bins = (
        ("le_1p5", -math.inf, 1.5),
        ("gt_1p5_le_2p0", 1.5, 2.0),
        ("gt_2p0_le_2p5", 2.0, 2.5),
        ("gt_2p5_le_3p0", 2.5, 3.0),
        ("gt_3p0_le_5p0", 3.0, 5.0),
        ("gt_5p0", 5.0, math.inf),
    )
    for variant in FULL_VARIANTS:
        paths = [
            RUN_ROOT
            / "full_af3"
            / f"af3_full_repr_v1_{variant}_seed{seed}"
            / "validation_predictions.tsv.gz"
            for seed in FULL_SEEDS
        ]
        records = ensemble(paths)
        summary.append({"variant": variant, **metrics(records)})
        gated = [row for row in records if confidence[row["sample_id"]] <= 2.5]
        strata.append(
            {"variant": variant, "stratum": "ipae_le_2p5", **metrics(gated)}
        )
        for name, lower, upper in bins:
            selected = [
                row
                for row in records
                if confidence[row["sample_id"]] > lower
                and confidence[row["sample_id"]] <= upper
            ]
            if selected:
                strata.append(
                    {"variant": variant, "stratum": name, **metrics(selected)}
                )
    return summary, strata


def filtered_source_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    indexed = {(str(row["source"]), str(row["variant"])): row for row in rows}
    output = []
    for variant in FILTERED_VARIANTS:
        af2 = indexed[("af2", variant)]
        af3 = indexed[("af3", variant)]
        if af2["count"] != af3["count"] or af2["positive"] != af3["positive"]:
            raise ValueError(f"AF2/AF3 OOF cohort mismatch: {variant}")
        output.append(
            {
                "variant": variant,
                "count": af2["count"],
                "delta_auprc_af2_minus_af3": float(af2["auprc"]) - float(af3["auprc"]),
                "delta_auroc_af2_minus_af3": float(af2["auroc"]) - float(af3["auroc"]),
                "delta_log_loss_af2_minus_af3": float(af2["log_loss"]) - float(af3["log_loss"]),
                "delta_brier_af2_minus_af3": float(af2["brier_score"]) - float(af3["brier_score"]),
            }
        )
    return output


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    filtered = filtered_summary()
    full, strata = full_summary()
    write_tsv(OUTPUT / "filtered_oof_ensemble.tsv", filtered)
    write_tsv(
        OUTPUT / "filtered_oof_source_delta.tsv",
        filtered_source_deltas(filtered),
    )
    write_tsv(OUTPUT / "full_af3_validation_ensemble.tsv", full)
    write_tsv(OUTPUT / "full_af3_validation_confidence_strata.tsv", strata)
    payload = {
        "status": "complete",
        "filtered_oof_models": len(filtered),
        "full_af3_models": len(full),
        "sealed_test_accessed": False,
        "project_validation_accessed_by_full_experiment": True,
        "project_validation_accessed_by_filtered_oof": False,
    }
    temporary = OUTPUT / f"metadata.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT / "metadata.json")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
