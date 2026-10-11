#!/usr/bin/env python3
"""Freeze and validate the matched, development-only completion experiment."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/v2/representation_benchmark_completion_v1"
BASE = ROOT / "processed/v1/training_10x/cohorts/ipae_min_le_2p5/known_target"
SPLITS = BASE / "representation_complementarity_cv_v1"
FOLDS = ROOT / "experiments/dmasif_feature_pruning_folds_v1.tsv"
SEEDS = (20260823, 20260824)
ADAPTERS = ("esmc", "dmasif_global", "dmasif_region", "geometry_g4", "atom_gnn", "rosetta39")
CACHES = {
    "esmc": ROOT / "features/esmc6b_v1",
    "atom": ROOT / "features/af3_v1/cohorts/ipae_min_le_2p5_interface_atom_v1",
    "geometry": ROOT / "features/af3_v1/cohorts/ipae_min_le_2p5_interface_geometry_descriptor_v1",
    "point": ROOT / "features/dmasif_v1/ipae_min_le_2p5_atom_patch_point82_v1",
    "knn": ROOT / "features/dmasif_v1/ipae_min_le_2p5_knn_partner_v1",
    "paired": ROOT / "features/dmasif_v1/ipae_min_le_2p5_paired_patch_v1",
    "rosetta": ROOT / "features/physical_atlas_v1/ipae_min_le_2p5_known_target_train_validation/features_v1",
}


def read_tsv(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, rows):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique(rows, key="sample_id"):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key}")
    return result


def audit_splits():
    original = unique(read_tsv(BASE / "train.tsv.gz"))
    folds = unique(read_tsv(FOLDS))
    if len(original) != 2881 or sum(r["label"] == "positive" for r in original.values()) != 756:
        raise ValueError("Unexpected development population")
    if set(original) != set(folds):
        raise ValueError("Frozen fold coverage mismatch")
    for sid, row in original.items():
        for key in ("label", "target_sequence_id", "leakage_unit_id"):
            if folds[sid][key] != row[key]:
                raise ValueError(f"Fold identity mismatch: {sid}/{key}")
    paths = [BASE / "train.tsv.gz", FOLDS]
    counts = {}
    for fold in range(5):
        partitions = {}
        for split in ("train", "validation", "outer_holdout"):
            path = SPLITS / f"fold{fold}" / f"{split}.tsv.gz"
            rows = unique(read_tsv(path))
            expected = {sid for sid in original if (
                int(folds[sid]["cv_fold"]) == fold if split == "outer_holdout" else
                int(folds[sid]["cv_fold"]) == (fold + 1) % 5 if split == "validation" else
                int(folds[sid]["cv_fold"]) not in (fold, (fold + 1) % 5)
            )}
            if set(rows) != expected:
                raise ValueError(f"Wrong split membership: {path}")
            for sid, row in rows.items():
                for key in ("label", "binder_sequence", "target_sequence", "leakage_unit_id", "target_cluster_id"):
                    if row[key] != original[sid][key]:
                        raise ValueError(f"Changed sample identity: {path}/{sid}/{key}")
            partitions[split] = rows
            paths.append(path)
        for key in ("leakage_unit_id", "binder_sequence", "binder_cluster_id"):
            sets = [{r[key] for r in rows.values() if r[key]} for rows in partitions.values()]
            if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
                raise ValueError(f"Fold {fold}: cross-partition {key} leakage")
        counts[str(fold)] = {split: len(rows) for split, rows in partitions.items()}
    return original, counts, paths


def prepare():
    if RUN.exists():
        raise FileExistsError(f"Refusing to replace experiment: {RUN}")
    original, counts, paths = audit_splits()
    cache_audit = {}
    for name, cache in CACHES.items():
        metadata_path = cache / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        paths.append(metadata_path)
        if name == "esmc":
            if metadata.get("complete") is not True:
                raise ValueError("Incomplete ESMC cache")
            index_path = cache / "sequence_manifest.tsv"
            index = unique(read_tsv(index_path), "sequence_key")
            def has_sequence(sequence):
                key = "SEQ_" + hashlib.sha256(sequence.encode()).hexdigest()[:24]
                return key in index and int(index[key]["length"]) == len(sequence)
            for row in original.values():
                chains = json.loads(row["target_chain_sequences"])
                if not has_sequence(row["binder_sequence"]) or not (
                    has_sequence(row["target_sequence"]) or
                    ("".join(chains) == row["target_sequence"] and all(has_sequence(c) for c in chains))
                ):
                    raise ValueError(f"Missing ESMC sequence: {row['sample_id']}")
        else:
            if metadata.get("status") != "complete":
                raise ValueError(f"Incomplete cache: {name}")
            index_path = cache / "index.tsv"
            index = unique(read_tsv(index_path))
            missing = set(original) - set(index)
            if missing:
                raise ValueError(f"Missing {name} samples: {len(missing)}")
        paths.append(index_path)
        cache_audit[name] = {"path": str(cache), "index_count": len(index), "development_coverage": 2881}
    fields = ("sample_ids", "sampling_weight", "interface_pae_min", "iptm", "mean_atom_plddt")
    # This shared auxiliary source is only a feature/weight lookup, not labels.
    source = BASE.parent / "cohort.tsv.gz"
    auxiliary = [{key: row[key] for key in fields} for row in read_tsv(source) if row["sample_ids"] in original]
    if set(unique(auxiliary, "sample_ids")) != set(original):
        raise ValueError("Auxiliary feature coverage mismatch")
    paths.append(source)
    source_paths = [
        ROOT / "experiments/representation_benchmark_completion_v1.md",
        *[ROOT / "scripts" / f for f in (
            "prepare_representation_benchmark_completion_v1.py",
            "train_representation_benchmark_completion_v1.py",
            "run_representation_benchmark_completion_v1.py",
            "submit_representation_benchmark_completion_v1.py",
            "summarize_representation_benchmark_completion_v1.py",
            "train_multimodal_v2.py", "train_interface_atom_e3nn_v1.py",
            "train_interface_geometry_descriptor_v1.py", "train_dmasif_knn_region_factorial_v1.py",
            "train_dmasif_region_pool_v1.py", "train_dmasif_knn_partner_v1.py",
            "run_physical_atlas_grouped_oof_v1.py", "physical_atlas_schema_v1.py",
        )],
        *[ROOT / "training" / f for f in (
            "data.py", "multimodal_data.py", "multimodal_model.py",
            "interface_atom_data.py", "interface_atom_e3nn.py", "interface_geometry_descriptor.py",
        )],
        ROOT / "slurm/run_representation_benchmark_completion_v1.sbatch",
        ROOT / "slurm/run_representation_benchmark_completion_cpu_v1.sbatch",
    ]
    for fold in range(5):
        for seed in SEEDS:
            reference = ROOT / "runs/v2/representation_source_v1/filtered_oof" / f"af2_af3_repr_v1_af3_af3_pair_fold{fold}_seed{seed}"
            paths.extend((reference / "config.json", reference / "outer_holdout_predictions.tsv.gz"))
    RUN.mkdir(parents=True)
    (RUN / "logs").mkdir()
    manifest = [{"adapter": adapter, "fold": fold, "seed": seed,
                 "device": "cpu" if adapter == "rosetta39" else "cuda",
                 "run_name": f"{adapter}_fold{fold}_seed{seed}"}
                for fold in range(5) for adapter in ADAPTERS for seed in SEEDS]
    write_tsv(RUN / "manifest.tsv", manifest)
    write_tsv(RUN / "auxiliary.tsv.gz", auxiliary)
    paths.extend((RUN / "manifest.tsv", RUN / "auxiliary.tsv.gz"))
    protocol = {
        "schema": "representation_benchmark_completion_v1", "status": "prepared",
        "sample_count": 2881, "positive": 756, "seeds": SEEDS,
        "split_counts": counts, "cache_audit": cache_audit,
        "input_sha256": {str(p): sha256(p) for p in paths},
        "source_sha256": {str(p): sha256(p) for p in source_paths},
        "project_validation_used_for_fit_or_evaluation": False, "sealed_test_accessed": False,
        "auxiliary_source_usage": "development-ID-filtered features and sampling weights only; labels unused",
    }
    (RUN / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(protocol, indent=2))


def verify():
    protocol = json.loads((RUN / "protocol.json").read_text())
    for group in ("input_sha256", "source_sha256"):
        for path, digest in protocol[group].items():
            if sha256(path) != digest:
                raise ValueError(f"Frozen experiment input changed: {path}")
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
        print("status=ok frozen_protocol_verified=true")
    else:
        prepare()
