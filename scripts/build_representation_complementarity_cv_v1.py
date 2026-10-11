#!/usr/bin/env python3
"""Build immutable train/early-stop/outer files for AF3-pair OOF training."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = (
    PROJECT_ROOT
    / "processed/v1/training_10x/cohorts/ipae_min_le_2p5/known_target/train.tsv.gz"
)
DEFAULT_FOLDS = PROJECT_ROOT / "experiments/dmasif_feature_pruning_folds_v1.tsv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "processed/v1/training_10x/cohorts/ipae_min_le_2p5/known_target/"
    "representation_complementarity_cv_v1"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "experiments/representation_complementarity_cv_v1.tsv"
SEEDS = (20260823, 20260824)


def read_tsv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    if args.output_dir.exists() or args.manifest.exists():
        raise FileExistsError("CV split package or manifest already exists")
    rows = read_tsv(args.train)
    folds = read_tsv(args.folds)
    fold_by_sample = {row["sample_id"]: int(row["cv_fold"]) for row in folds}
    if len(rows) != 2881 or set(fold_by_sample) != {row["sample_id"] for row in rows}:
        raise ValueError("Fold manifest does not exactly cover 2,881 training samples")

    manifest_rows = []
    metadata_folds = []
    for outer_fold in range(5):
        early_stop_fold = (outer_fold + 1) % 5
        selected = {
            "train": [
                row
                for row in rows
                if fold_by_sample[row["sample_id"]]
                not in {outer_fold, early_stop_fold}
            ],
            "validation": [
                row
                for row in rows
                if fold_by_sample[row["sample_id"]] == early_stop_fold
            ],
            "outer_holdout": [
                row
                for row in rows
                if fold_by_sample[row["sample_id"]] == outer_fold
            ],
        }
        group_sets = [
            {row["leakage_unit_id"] for row in split_rows}
            for split_rows in selected.values()
        ]
        if any(
            group_sets[i] & group_sets[j]
            for i in range(3)
            for j in range(i + 1, 3)
        ):
            raise ValueError(f"Leakage group crosses roles for outer fold {outer_fold}")
        fold_dir = args.output_dir / f"fold{outer_fold}"
        fold_dir.mkdir(parents=True)
        role_metadata = {}
        for role, split_rows in selected.items():
            path = fold_dir / f"{role}.tsv.gz"
            write_tsv(path, split_rows)
            role_metadata[role] = {
                "path": str(path.resolve()),
                "count": len(split_rows),
                "positive": sum(row["label"] == "positive" for row in split_rows),
                "negative": sum(row["label"] == "negative" for row in split_rows),
                "sha256": sha256(path),
            }
        metadata_folds.append(
            {
                "outer_fold": outer_fold,
                "early_stop_fold": early_stop_fold,
                "roles": role_metadata,
            }
        )
        for seed in SEEDS:
            manifest_rows.append(
                {
                    "outer_fold": str(outer_fold),
                    "early_stop_fold": str(early_stop_fold),
                    "seed": str(seed),
                    "run_name": f"representation_complementarity_cv_v1_af3_pair_fold{outer_fold}_seed{seed}",
                    "train_file": role_metadata["train"]["path"],
                    "early_stop_file": role_metadata["validation"]["path"],
                    "outer_holdout_file": role_metadata["outer_holdout"]["path"],
                }
            )
    write_tsv(args.manifest, manifest_rows)
    metadata = {
        "schema_version": "representation_complementarity_cv_v1",
        "source_train": str(args.train.resolve()),
        "source_train_sha256": sha256(args.train),
        "fold_manifest": str(args.folds.resolve()),
        "fold_manifest_sha256": sha256(args.folds),
        "seeds": list(SEEDS),
        "folds": metadata_folds,
        "project_validation_accessed": False,
        "sealed_test_accessed": False,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "runs": len(manifest_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
