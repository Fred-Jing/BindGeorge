#!/usr/bin/env python3
"""Train point-only dMaSIF feature-engineering controls."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
DEFAULT_CACHE = (
    PROJECT_ROOT / "features/dmasif_v1/ipae_min_le_2p5_point_feature_engineering_v1"
)
FEATURE_DIMS = {
    "base82": 82,
    "symmetric130": 130,
    "geometry92": 92,
    "region_augmented84": 84,
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class FeatureDataset(Dataset):
    def __init__(self, rows, cache: Path, variant: str):
        self.rows = rows
        self.cache = cache
        self.variant = variant

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        with np.load(
            self.cache / "samples" / row["sample_id"] / "point_features.npz",
            allow_pickle=False,
        ) as arrays:
            target = np.asarray(
                arrays[f"target_{self.variant}"], dtype=np.float32
            )
            binder = np.asarray(
                arrays[f"binder_{self.variant}"], dtype=np.float32
            )
        expected = FEATURE_DIMS[self.variant]
        if target.shape[1] != expected or binder.shape[1] != expected:
            raise ValueError(f"Invalid {self.variant} shape for {row['sample_id']}")
        return {
            "sample_id": row["sample_id"],
            "target_name": row["target_name"],
            "target_sequence_id": row["target_sequence_id"],
            "target": target,
            "binder": binder,
            "label": float(row["label"] == "positive"),
        }


def collate(items: list[dict[str, object]], feature_dim: int) -> dict[str, object]:
    target, binder, target_batch, binder_batch = [], [], [], []
    for batch_index, item in enumerate(items):
        target.append(item["target"])
        binder.append(item["binder"])
        target_batch.append(
            np.full(len(item["target"]), batch_index, dtype=np.int64)
        )
        binder_batch.append(
            np.full(len(item["binder"]), batch_index, dtype=np.int64)
        )
    return {
        "target": torch.from_numpy(np.concatenate(target)),
        "binder": torch.from_numpy(np.concatenate(binder)),
        "target_batch": torch.from_numpy(np.concatenate(target_batch)),
        "binder_batch": torch.from_numpy(np.concatenate(binder_batch)),
        "batch_size": len(items),
        "feature_dim": feature_dim,
        "label": torch.tensor([item["label"] for item in items], dtype=torch.float32),
        "sample_id": [item["sample_id"] for item in items],
        "target_name": [item["target_name"] for item in items],
        "target_sequence_id": [item["target_sequence_id"] for item in items],
    }


def segment_mean_max(
    values: torch.Tensor, groups: torch.Tensor, group_count: int
) -> torch.Tensor:
    dimension = values.shape[1]
    sums = torch.zeros(group_count, dimension, device=values.device, dtype=values.dtype)
    counts = torch.zeros(group_count, 1, device=values.device, dtype=values.dtype)
    sums.index_add_(0, groups, values)
    counts.index_add_(
        0,
        groups,
        torch.ones(len(values), 1, device=values.device, dtype=values.dtype),
    )
    means = sums / counts.clamp_min(1.0)
    maxima = torch.full(
        (group_count, dimension),
        -torch.inf,
        device=values.device,
        dtype=values.dtype,
    )
    maxima.scatter_reduce_(
        0,
        groups[:, None].expand(-1, dimension),
        values,
        reduce="amax",
        include_self=True,
    )
    maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
    return torch.cat([means, maxima], dim=1)


class FeaturePointClassifier(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.target_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.binder_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, batch: dict[str, object]) -> torch.Tensor:
        target_values = self.target_encoder(batch["target"])
        binder_values = self.binder_encoder(batch["binder"])
        target = segment_mean_max(
            target_values, batch["target_batch"], batch["batch_size"]
        )
        binder = segment_mean_max(
            binder_values, batch["binder_batch"], batch["batch_size"]
        )
        interaction = torch.cat(
            [target, binder, torch.abs(target - binder), target * binder], dim=1
        )
        return self.classifier(interaction).squeeze(-1)


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def evaluate(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    sample_ids, target_names, target_ids = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            probabilities.append(torch.sigmoid(model(batch)).cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
            sample_ids.extend(batch["sample_id"])
            target_names.extend(batch["target_name"])
            target_ids.extend(batch["target_sequence_id"])
    return (
        np.concatenate(probabilities),
        sample_ids,
        target_names,
        target_ids,
        np.concatenate(labels),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=FEATURE_DIMS, required=True)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output requires review: {args.output_dir}")
    metadata = json.loads((args.cache / "metadata.json").read_text())
    if (
        metadata.get("status") != "complete"
        or metadata.get("sample_count") != 3471
        or metadata.get("sealed_test_accessed") is not False
        or metadata.get("feature_variants", {}).get(args.variant)
        != FEATURE_DIMS[args.variant]
    ):
        raise ValueError("Invalid feature-engineering cache")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_tsv(args.split_root / "train.tsv.gz")
    validation_rows = read_tsv(args.split_root / "validation.tsv.gz")
    feature_dim = FEATURE_DIMS[args.variant]
    train_dataset = FeatureDataset(train_rows, args.cache, args.variant)
    validation_dataset = FeatureDataset(validation_rows, args.cache, args.variant)
    collate_fn = lambda items: collate(items, feature_dim)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeaturePointClassifier(feature_dim).to(device)
    initial_state_sha256 = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
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
        torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=True)
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
            sample_ids, target_names, target_ids, labels, probability, strict=True
        )
    ]
    target_metrics = macro_target_metrics(records)
    fit = {
        "device": str(device),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_auprc_during_training": best,
        "initial_state_sha256": initial_state_sha256,
        "fit_seconds": time.time() - started,
    }
    metrics = {
        "variant": args.variant,
        "seed": args.seed,
        "feature_dimension": feature_dim,
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
        args.output_dir / "per_target_metrics.tsv", {"validation": target_metrics["rows"]}
    )
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    write_json_atomic(
        args.output_dir / "config.json",
        {
            **config,
            "feature_dimension": feature_dim,
            "trainable_parameters": fit["trainable_parameters"],
            "point_cap": None,
            "sealed_test_accessed": False,
        },
    )
    write_json_atomic(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("status=ok sealed_test_accessed=false")


if __name__ == "__main__":
    main()
