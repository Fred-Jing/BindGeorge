#!/usr/bin/env python3
"""Train a controlled KNN-descriptor by core/rim-pooling factorial."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_dmasif_knn_partner_v1 import replace_partner_descriptor  # noqa: E402
from scripts.train_dmasif_point_feature_engineering_v1 import (  # noqa: E402
    evaluate,
    move_batch,
    read_tsv,
    state_hash,
)
from scripts.train_dmasif_region_pool_v1 import (  # noqa: E402
    FEATURE_DIMENSION,
    REGION_SHUFFLE_SEED,
    RegionPoolClassifier,
    collate,
    shuffled_regions,
)
from scripts.train_multimodal_v2 import (  # noqa: E402
    best_f1_threshold,
    macro_target_metrics,
    ranking_metrics,
    set_seed,
    write_json_atomic,
    write_predictions,
    write_target_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_ROOT = (
    PROJECT_ROOT / "processed/v1/training_10x/cohorts/ipae_min_le_2p5/known_target"
)
DEFAULT_POINT_CACHE = (
    PROJECT_ROOT / "features/dmasif_v1/ipae_min_le_2p5_atom_patch_point82_v1"
)
DEFAULT_KNN_CACHE = (
    PROJECT_ROOT / "features/dmasif_v1/ipae_min_le_2p5_knn_partner_v1"
)
DEFAULT_PAIRED_CACHE = (
    PROJECT_ROOT / "features/dmasif_v1/ipae_min_le_2p5_paired_patch_v1"
)
VARIANT_SPECS = {
    "knn8_global_capacity": {
        "descriptor_key": "knn8",
        "region_mode": "real",
        "model_mode": "global_capacity",
    },
    "knn8_region_pool": {
        "descriptor_key": "knn8",
        "region_mode": "real",
        "model_mode": "region_pool",
    },
    "knn8_region_shuffled": {
        "descriptor_key": "knn8",
        "region_mode": "shuffled",
        "model_mode": "region_pool",
    },
    "knn8_descriptor_shuffled_region_pool": {
        "descriptor_key": "knn8_shuffled",
        "region_mode": "real",
        "model_mode": "region_pool",
    },
}


class FactorialDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        point_cache: Path,
        knn_cache: Path,
        paired_cache: Path,
        variant: str,
    ):
        self.rows = rows
        self.point_cache = point_cache
        self.knn_cache = knn_cache
        self.paired_cache = paired_cache
        self.spec = VARIANT_SPECS[variant]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        sample_id = row["sample_id"]
        with np.load(
            self.point_cache / "samples" / sample_id / "point82.npz",
            allow_pickle=False,
        ) as arrays:
            base_target = np.asarray(arrays["target_features"], dtype=np.float32)
            base_binder = np.asarray(arrays["binder_features"], dtype=np.float32)
        descriptor_key = self.spec["descriptor_key"]
        with np.load(
            self.knn_cache
            / "samples"
            / sample_id
            / "partner_descriptors.npz",
            allow_pickle=False,
        ) as arrays:
            target_partner = np.asarray(
                arrays[f"target_{descriptor_key}"], dtype=np.float32
            )
            binder_partner = np.asarray(
                arrays[f"binder_{descriptor_key}"], dtype=np.float32
            )
        with np.load(
            self.paired_cache / "samples" / sample_id / "paired_patch.npz",
            allow_pickle=False,
        ) as arrays:
            target_region = np.asarray(arrays["target_region"], dtype=np.int64)
            binder_region = np.asarray(arrays["binder_region"], dtype=np.int64)

        target = replace_partner_descriptor(base_target, target_partner)
        binder = replace_partner_descriptor(base_binder, binder_partner)
        if len(target_region) != len(target) or len(binder_region) != len(binder):
            raise ValueError(f"Point/region count mismatch for {sample_id}")
        if self.spec["region_mode"] == "shuffled":
            target_region = shuffled_regions(target_region, sample_id, "target")
            binder_region = shuffled_regions(binder_region, sample_id, "binder")
        return {
            "sample_id": sample_id,
            "target_name": row["target_name"],
            "target_sequence_id": row["target_sequence_id"],
            "target": target,
            "binder": binder,
            "target_region": target_region,
            "binder_region": binder_region,
            "label": float(row["label"] == "positive"),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANT_SPECS, required=True)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--point-cache", type=Path, default=DEFAULT_POINT_CACHE)
    parser.add_argument("--knn-cache", type=Path, default=DEFAULT_KNN_CACHE)
    parser.add_argument("--paired-cache", type=Path, default=DEFAULT_PAIRED_CACHE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output requires review: {args.output_dir}")

    point_metadata = json.loads((args.point_cache / "metadata.json").read_text())
    knn_metadata = json.loads((args.knn_cache / "metadata.json").read_text())
    paired_metadata = json.loads((args.paired_cache / "metadata.json").read_text())
    for name, metadata in (
        ("point", point_metadata),
        ("knn", knn_metadata),
        ("paired", paired_metadata),
    ):
        if (
            metadata.get("status") != "complete"
            or metadata.get("sample_count") != 3471
            or metadata.get("sealed_test_accessed") is not False
        ):
            raise ValueError(f"Invalid or unsafe {name} cache")
    if (
        point_metadata.get("feature_dimension") != FEATURE_DIMENSION
        or knn_metadata.get("neighbor_counts") != [3, 8]
        or paired_metadata.get("region_encoding")
        != "0=core/core,1=core/rim,2=rim/core,3=rim/rim"
    ):
        raise ValueError("Unexpected cache schema")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_tsv(args.split_root / "train.tsv.gz")
    validation_rows = read_tsv(args.split_root / "validation.tsv.gz")
    dataset_args = (
        args.point_cache,
        args.knn_cache,
        args.paired_cache,
        args.variant,
    )
    train_dataset = FactorialDataset(train_rows, *dataset_args)
    validation_dataset = FactorialDataset(validation_rows, *dataset_args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = VARIANT_SPECS[args.variant]
    model = RegionPoolClassifier(spec["model_mode"]).to(device)
    initial_state_sha256 = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    best = -math.inf
    stale = 0
    best_epoch = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch["label"])
            loss.backward()
            optimizer.step()
        probability, _, _, _, labels = evaluate(model, validation_loader, device)
        score = float(average_precision_score(labels, probability))
        print(
            f"epoch={epoch} validation_auprc={score} "
            f"elapsed_seconds={time.time() - started}",
            flush=True,
        )
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            torch.save(model.state_dict(), args.output_dir / "best_model.pt.tmp")
            (args.output_dir / "best_model.pt.tmp").replace(
                args.output_dir / "best_model.pt"
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(
        torch.load(
            args.output_dir / "best_model.pt", map_location=device, weights_only=True
        )
    )
    train_probability, _, _, _, train_labels = evaluate(model, train_loader, device)
    probability, sample_ids, target_names, target_ids, labels = evaluate(
        model, validation_loader, device
    )
    threshold = best_f1_threshold(train_labels.astype(np.int64), train_probability)
    records = [
        {
            "sample_id": sample_id,
            "target_name": target_name,
            "target_sequence_id": target_id,
            "label": int(label),
            "probability": float(probability_value),
            "sampling_weight": 1.0,
        }
        for sample_id, target_name, target_id, label, probability_value in zip(
            sample_ids,
            target_names,
            target_ids,
            labels,
            probability,
            strict=True,
        )
    ]
    target_metrics = macro_target_metrics(records)
    fit = {
        "device": str(device),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_auprc_during_training": best,
        "initial_state_sha256": initial_state_sha256,
        "fit_seconds": time.time() - started,
    }
    metrics = {
        "variant": args.variant,
        "descriptor_key": spec["descriptor_key"],
        "region_mode": spec["region_mode"],
        "pooling": spec["model_mode"],
        "seed": args.seed,
        "feature_dimension": FEATURE_DIMENSION,
        "region_shuffle_seed": REGION_SHUFFLE_SEED,
        "fit": fit,
        "selected_train_f1_threshold": threshold,
        "validation": {
            **ranking_metrics(records, threshold),
            "log_loss": float(log_loss(labels, probability)),
            **target_metrics["summary"],
        },
        "split_sizes": {"train": len(train_rows), "validation": len(validation_rows)},
        "point_cap": None,
        "sealed_test_accessed": False,
    }
    write_predictions(args.output_dir / "validation_predictions.tsv.gz", records)
    write_target_rows(
        args.output_dir / "per_target_metrics.tsv",
        {"validation": target_metrics["rows"]},
    )
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    write_json_atomic(
        args.output_dir / "config.json",
        {
            **config,
            **spec,
            "feature_dimension": FEATURE_DIMENSION,
            "trainable_parameters": fit["trainable_parameters"],
            "region_shuffle_seed": REGION_SHUFFLE_SEED,
            "point_cap": None,
            "sealed_test_accessed": False,
        },
    )
    write_json_atomic(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("status=ok sealed_test_accessed=false")


if __name__ == "__main__":
    main()
