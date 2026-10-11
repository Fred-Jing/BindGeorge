#!/usr/bin/env python3
"""Train frozen-feature binder classifiers on the AF3-confidence cohort."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.multimodal_data import (  # noqa: E402
    CONFIDENCE_FIELDS,
    MultimodalPairDataset,
    PairCellBatchSampler,
    collate_multimodal_pairs,
)
from training.multimodal_model import (  # noqa: E402
    MODEL_VARIANTS,
    GEOMETRY_POOLINGS,
    PAIR_FUSIONS,
    PAIR_POOLINGS,
    PAIR_READOUT_MODES,
    SINGLE_READOUTS,
    PairConditionedBinderClassifier,
)


DEFAULT_SPLIT_ROOT = (
    PROJECT_ROOT
    / "processed/v1/training_10x/cohorts/ipae_min_le_2p5/known_target"
)
DEFAULT_ESMC_CACHE = PROJECT_ROOT / "features/esmc6b_v1"
DEFAULT_AF3_CACHE = PROJECT_ROOT / "features/af3_v1/cohorts/ipae_min_le_2p5_cross"
DEFAULT_GEOMETRY_CACHE = (
    PROJECT_ROOT / "features/af3_v1/cohorts/ipae_min_le_2p5_geometry"
)
DEFAULT_GEOMETRY_V2_CACHE = (
    PROJECT_ROOT / "features/af3_v1/cohorts/ipae_min_le_2p5_geometry_v2"
)
DEFAULT_GEOMETRY_CELL_CACHE = (
    PROJECT_ROOT / "features/af3_v1/cohorts/ipae_min_le_2p5_geometry_cells_v1"
)
DEFAULT_INTERFACE_DESCRIPTOR_CACHE = (
    PROJECT_ROOT
    / "features/af3_v1/cohorts/ipae_min_le_2p5_interface_geometry_descriptor_v1"
)
DEFAULT_CONFIDENCE = (
    PROJECT_ROOT
    / "processed/v1/training_10x/cohorts/ipae_min_le_2p5/cohort.tsv.gz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(MODEL_VARIANTS), default="full")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--development-holdout-file", type=Path)
    parser.add_argument("--development-holdout-name", default="outer_holdout")
    parser.add_argument("--esmc-cache", type=Path, default=DEFAULT_ESMC_CACHE)
    parser.add_argument("--af3-cache", type=Path, default=DEFAULT_AF3_CACHE)
    parser.add_argument(
        "--representation-source", choices=("af3", "af2"), default="af3"
    )
    parser.add_argument(
        "--representation-cache",
        type=Path,
        help="Generic single/pair cache; overrides --af3-cache when provided",
    )
    parser.add_argument(
        "--geometry-cache", type=Path, default=DEFAULT_GEOMETRY_CACHE
    )
    parser.add_argument(
        "--geometry-v2-cache", type=Path, default=DEFAULT_GEOMETRY_V2_CACHE
    )
    parser.add_argument(
        "--geometry-cell-cache", type=Path, default=DEFAULT_GEOMETRY_CELL_CACHE
    )
    parser.add_argument(
        "--interface-descriptor-cache",
        type=Path,
        default=DEFAULT_INTERFACE_DESCRIPTOR_CACHE,
    )
    parser.add_argument(
        "--interface-descriptor-variant", default="g1_aa_without_size"
    )
    parser.add_argument("--confidence-table", type=Path, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--residue-dim", type=int, default=128)
    parser.add_argument("--pair-dim", type=int, default=32)
    parser.add_argument(
        "--pair-fusion", choices=PAIR_FUSIONS, default="projected_sum"
    )
    parser.add_argument(
        "--pair-pooling", choices=PAIR_POOLINGS, default="attention"
    )
    parser.add_argument("--pair-pooling-queries", type=int, default=1)
    parser.add_argument(
        "--single-readout", choices=SINGLE_READOUTS, default="legacy_all"
    )
    parser.add_argument("--raw-branch-dim", type=int, default=32)
    parser.add_argument("--matching-branch-dim", type=int, default=32)
    parser.add_argument("--single-fusion-dim", type=int, default=64)
    parser.add_argument("--geometry-rbf-bins", type=int, default=16)
    parser.add_argument("--geometry-rbf-minimum", type=float, default=2.0)
    parser.add_argument("--geometry-rbf-maximum", type=float, default=32.0)
    parser.add_argument("--pair-distance-cutoff", type=float, default=10.0)
    parser.add_argument("--interface-descriptor-dim", type=int, default=118)
    parser.add_argument("--descriptor-branch-dim", type=int, default=8)
    parser.add_argument(
        "--pair-readout-mode", choices=PAIR_READOUT_MODES, default="none"
    )
    parser.add_argument("--pair-readout-hidden-dim", type=int, default=32)
    parser.add_argument("--pair-readout-depth", type=int, default=1)
    parser.add_argument("--random-mask-reference-cutoff", type=float, default=10.0)
    parser.add_argument("--pair-mask-seed", type=int, default=0)
    parser.add_argument(
        "--geometry-pooling", choices=GEOMETRY_POOLINGS, default="mean_max"
    )
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--projection-dropout", type=float, default=0.1)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--classifier-dropout", type=float, default=0.2)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--max-padded-pair-cells", type=int, default=300_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--evaluate-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def make_loader(
    dataset: MultimodalPairDataset,
    *,
    max_batch_size: int,
    max_padded_pair_cells: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> tuple[PairCellBatchSampler, DataLoader]:
    sampler = PairCellBatchSampler(
        dataset.lengths,
        max_batch_size=max_batch_size,
        max_padded_pair_cells=max_padded_pair_cells,
        shuffle=shuffle,
        seed=seed,
    )
    options: dict[str, object] = {}
    if num_workers > 0:
        options["prefetch_factor"] = 2
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_multimodal_pairs,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        **options,
    )
    return sampler, loader


def confidence_normalization(dataset: MultimodalPairDataset) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(
        [
            [float(dataset.confidence[row["sample_id"]][field]) for field in CONFIDENCE_FIELDS]
            for row in dataset.rows
        ],
        dtype=np.float64,
    )
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0)
    standard_deviation[standard_deviation < 1e-8] = 1.0
    return mean, standard_deviation


def interface_descriptor_normalization(
    dataset: MultimodalPairDataset,
) -> tuple[np.ndarray, np.ndarray]:
    store = dataset.interface_descriptor_store
    if store is None:
        raise ValueError("Interface descriptor normalization requires a descriptor store")
    values = np.stack(
        [
            store.get(
                dataset.interface_descriptor_source_by_sample[row["sample_id"]]
                if dataset.interface_descriptor_source_by_sample is not None
                else row["sample_id"]
            ).numpy()
            for row in dataset.rows
        ]
    ).astype(np.float64)
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0)
    standard_deviation[standard_deviation < 1e-8] = 1.0
    return mean, standard_deviation


def move_batch(
    batch: dict[str, object],
    device: torch.device,
    confidence_mean: torch.Tensor,
    confidence_std: torch.Tensor,
    interface_descriptor_mean: torch.Tensor | None = None,
    interface_descriptor_std: torch.Tensor | None = None,
) -> dict[str, object]:
    output = dict(batch)
    tensor_keys = (
        "target_mask",
        "binder_mask",
        "target_esmc",
        "binder_esmc",
        "target_single",
        "binder_single",
        "target_binder_pair",
        "binder_target_pair",
        "target_binder_distance",
        "target_binder_selection_mask",
        "target_binder_geometry_summary",
        "target_binder_geometry",
        "interface_descriptor",
        "confidence",
        "label",
        "sampling_weight",
    )
    for key in tensor_keys:
        if key in batch:
            output[key] = batch[key].to(device, non_blocking=True)
    output["confidence"] = (
        output["confidence"].float() - confidence_mean
    ) / confidence_std
    if "interface_descriptor" in output:
        if interface_descriptor_mean is None or interface_descriptor_std is None:
            raise ValueError("Interface descriptor normalization tensors are missing")
        output["interface_descriptor"] = (
            output["interface_descriptor"].float() - interface_descriptor_mean
        ) / interface_descriptor_std
    return output


def model_forward(model: nn.Module, batch: dict[str, object]) -> torch.Tensor:
    keys = (
        "target_mask",
        "binder_mask",
        "target_esmc",
        "binder_esmc",
        "target_single",
        "binder_single",
        "target_binder_pair",
        "binder_target_pair",
        "target_binder_distance",
        "target_binder_selection_mask",
        "target_binder_geometry_summary",
        "target_binder_geometry",
        "interface_descriptor",
        "confidence",
    )
    return model(**{key: batch[key] for key in keys if key in batch})


def safe_metric(function, labels, values, **kwargs) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(function(labels, values, **kwargs))


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(f1))])


def ranking_metrics(
    records: list[dict[str, object]], threshold: float
) -> dict[str, object]:
    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    probabilities = np.asarray(
        [record["probability"] for record in records], dtype=np.float64
    )
    weights = np.asarray(
        [record["sampling_weight"] for record in records], dtype=np.float64
    )
    predictions = probabilities >= threshold
    positive_rate = float(labels.mean())
    weighted_positive_rate = float(np.sum(weights * labels) / np.sum(weights))
    output: dict[str, object] = {
        "count": len(records),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "positive_rate": positive_rate,
        "weighted_positive_rate": weighted_positive_rate,
        "auroc": safe_metric(roc_auc_score, labels, probabilities),
        "auprc": safe_metric(average_precision_score, labels, probabilities),
        "weighted_auroc": safe_metric(
            roc_auc_score, labels, probabilities, sample_weight=weights
        ),
        "weighted_auprc": safe_metric(
            average_precision_score, labels, probabilities, sample_weight=weights
        ),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "threshold": threshold,
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
    }
    order = np.argsort(-probabilities, kind="stable")
    for requested_k in (20, 50, 100):
        k = min(requested_k, len(records))
        top_labels = labels[order[:k]]
        precision_at_k = float(top_labels.mean())
        output[f"precision_at_{requested_k}"] = precision_at_k
        output[f"recall_at_{requested_k}"] = float(top_labels.sum() / max(1, labels.sum()))
        output[f"enrichment_at_{requested_k}"] = (
            precision_at_k / positive_rate if positive_rate else None
        )
    return output


def macro_target_metrics(records: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[str(record["target_name"])].append(record)
    rows = []
    for target_name, target_records in sorted(groups.items()):
        labels = np.asarray([row["label"] for row in target_records], dtype=np.int64)
        if len(np.unique(labels)) < 2:
            continue
        probabilities = np.asarray(
            [row["probability"] for row in target_records], dtype=np.float64
        )
        weights = np.asarray(
            [row["sampling_weight"] for row in target_records], dtype=np.float64
        )
        rows.append(
            {
                "target_name": target_name,
                "count": len(target_records),
                "positive": int(labels.sum()),
                "negative": int(len(labels) - labels.sum()),
                "auroc": float(roc_auc_score(labels, probabilities)),
                "auprc": float(average_precision_score(labels, probabilities)),
                "weighted_auroc": float(
                    roc_auc_score(labels, probabilities, sample_weight=weights)
                ),
                "weighted_auprc": float(
                    average_precision_score(labels, probabilities, sample_weight=weights)
                ),
            }
        )
    summary = {"eligible_targets": len(rows)}
    for key in ("auroc", "auprc", "weighted_auroc", "weighted_auprc"):
        summary[f"macro_target_{key}"] = (
            float(np.mean([row[key] for row in rows])) if rows else None
        )
    return {"summary": summary, "rows": rows}


def macro_target_sequence_metrics(records: list[dict[str, object]]) -> dict[str, object]:
    sequence_records = [
        {**record, "target_name": record["target_sequence_id"]} for record in records
    ]
    result = macro_target_metrics(sequence_records)
    summary = {
        (
            "eligible_target_sequences"
            if key == "eligible_targets"
            else key.replace("macro_target_", "macro_target_sequence_", 1)
        ): value
        for key, value in result["summary"].items()
    }
    rows = [
        {
            "target_sequence_id": row["target_name"],
            **{key: value for key, value in row.items() if key != "target_name"},
        }
        for row in result["rows"]
    ]
    return {"summary": summary, "rows": rows}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    confidence_mean: torch.Tensor,
    confidence_std: torch.Tensor,
    interface_descriptor_mean: torch.Tensor | None = None,
    interface_descriptor_std: torch.Tensor | None = None,
) -> tuple[float, list[dict[str, object]]]:
    model.eval()
    records = []
    total_loss = 0.0
    total_count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(
                batch,
                device,
                confidence_mean,
                confidence_std,
                interface_descriptor_mean,
                interface_descriptor_std,
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = model_forward(model, batch)
                loss = criterion(logits, batch["label"])
            probabilities = torch.sigmoid(logits).float().cpu().numpy()
            labels = batch["label"].float().cpu().numpy()
            weights = batch["sampling_weight"].float().cpu().numpy()
            total_loss += float(loss.item()) * len(labels)
            total_count += len(labels)
            for index, probability in enumerate(probabilities):
                records.append(
                    {
                        "sample_id": batch["sample_id"][index],
                        "target_name": batch["target_name"][index],
                        "target_sequence_id": batch["target_sequence_id"][index],
                        "label": int(labels[index]),
                        "sampling_weight": float(weights[index]),
                        "probability": float(probability),
                    }
                )
    return total_loss / total_count, records


def write_predictions(path: Path, records: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(records[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def write_target_rows(path: Path, split_rows: dict[str, list[dict[str, object]]]) -> None:
    fields = (
        "split",
        "target_name",
        "count",
        "positive",
        "negative",
        "auroc",
        "auprc",
        "weighted_auroc",
        "weighted_auprc",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for split, rows in split_rows.items():
            for row in rows:
                writer.writerow({"split": split, **row})


def write_target_sequence_rows(
    path: Path, split_rows: dict[str, list[dict[str, object]]]
) -> None:
    fields = (
        "split", "target_sequence_id", "count", "positive", "negative",
        "auroc", "auprc", "weighted_auroc", "weighted_auprc",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for split, rows in split_rows.items():
            for row in rows:
                writer.writerow({"split": split, **row})


def main() -> None:
    args = parse_args()
    if args.representation_source == "af2" and args.representation_cache is None:
        raise ValueError("--representation-cache is required for AF2 representations")
    representation_cache = args.representation_cache or args.af3_cache
    if (args.output_dir / "best_model.pt").exists():
        raise FileExistsError(f"Run already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.development_holdout_name in {"train", "validation", "test"}:
        raise ValueError("Development holdout name collides with a reserved split")
    split_paths = {
        "train": args.train_file or args.split_root / "train.tsv.gz",
        "validation": args.validation_file or args.split_root / "validation.tsv.gz",
    }
    if args.development_holdout_file is not None:
        split_paths[args.development_holdout_name] = args.development_holdout_file
    active_splits = ["train", "validation"]
    if args.development_holdout_file is not None:
        active_splits.append(args.development_holdout_name)
    if args.evaluate_test:
        active_splits.append("test")
        split_paths["test"] = args.split_root / "test.tsv.gz"
    datasets = {
        split: MultimodalPairDataset(
            split_path=split_paths[split],
            esmc_cache_dir=args.esmc_cache,
            af3_cache_dir=representation_cache,
            confidence_path=args.confidence_table,
            variant=args.variant,
            geometry_cache_dir=args.geometry_cache,
            geometry_v2_cache_dir=args.geometry_v2_cache,
            geometry_cell_cache_dir=args.geometry_cell_cache,
            interface_descriptor_cache_dir=args.interface_descriptor_cache,
            interface_descriptor_variant=args.interface_descriptor_variant,
            random_mask_reference_cutoff=args.random_mask_reference_cutoff,
            pair_mask_seed=args.pair_mask_seed,
        )
        for split in active_splits
    }
    samplers = {}
    loaders = {}
    for offset, split in enumerate(active_splits):
        samplers[split], loaders[split] = make_loader(
            datasets[split],
            max_batch_size=args.max_batch_size,
            max_padded_pair_cells=args.max_padded_pair_cells,
            num_workers=args.num_workers,
            shuffle=split == "train",
            seed=args.seed + offset,
        )

    confidence_mean_np, confidence_std_np = confidence_normalization(datasets["train"])
    confidence_mean = torch.tensor(confidence_mean_np, dtype=torch.float32, device=device)
    confidence_std = torch.tensor(confidence_std_np, dtype=torch.float32, device=device)
    descriptor_store = datasets["train"].interface_descriptor_store
    if descriptor_store is not None:
        if descriptor_store.width != args.interface_descriptor_dim:
            raise ValueError(
                "Interface descriptor width mismatch: "
                f"cache={descriptor_store.width} model={args.interface_descriptor_dim}"
            )
        descriptor_mean_np, descriptor_std_np = interface_descriptor_normalization(
            datasets["train"]
        )
        interface_descriptor_mean = torch.tensor(
            descriptor_mean_np, dtype=torch.float32, device=device
        )
        interface_descriptor_std = torch.tensor(
            descriptor_std_np, dtype=torch.float32, device=device
        )
    else:
        descriptor_mean_np = descriptor_std_np = None
        interface_descriptor_mean = interface_descriptor_std = None
    model = PairConditionedBinderClassifier(
        variant=args.variant,
        residue_dimension=args.residue_dim,
        projected_pair_dimension=args.pair_dim,
        num_heads=args.num_heads,
        projection_dropout=args.projection_dropout,
        attention_dropout=args.attention_dropout,
        classifier_dropout=args.classifier_dropout,
        pair_fusion=args.pair_fusion,
        pair_pooling=args.pair_pooling,
        pair_pooling_queries=args.pair_pooling_queries,
        single_readout=args.single_readout,
        raw_branch_dimension=args.raw_branch_dim,
        matching_branch_dimension=args.matching_branch_dim,
        single_fusion_dimension=args.single_fusion_dim,
        geometry_rbf_bins=args.geometry_rbf_bins,
        geometry_rbf_minimum=args.geometry_rbf_minimum,
        geometry_rbf_maximum=args.geometry_rbf_maximum,
        geometry_pooling=args.geometry_pooling,
        pair_distance_cutoff=args.pair_distance_cutoff,
        interface_descriptor_dimension=args.interface_descriptor_dim,
        descriptor_branch_dimension=args.descriptor_branch_dim,
        pair_readout_mode=args.pair_readout_mode,
        pair_readout_hidden_dimension=args.pair_readout_hidden_dim,
        pair_readout_depth=args.pair_readout_depth,
    ).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device": str(device),
        "representation_required": bool(
            MODEL_VARIANTS[args.variant]["use_af3_single"]
            or MODEL_VARIANTS[args.variant]["use_af3_pair"]
        ),
        "effective_representation_cache": (
            str(representation_cache)
            if MODEL_VARIANTS[args.variant]["use_af3_single"]
            or MODEL_VARIANTS[args.variant]["use_af3_pair"]
            else None
        ),
        "trainable_parameters": trainable_parameters,
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "batches_per_epoch": {split: len(sampler) for split, sampler in samplers.items()},
        "confidence_fields": list(CONFIDENCE_FIELDS),
        "confidence_train_mean": confidence_mean_np.tolist(),
        "confidence_train_std": confidence_std_np.tolist(),
        "geometry_shuffle_statistics": {
            split: dataset.geometry_shuffle_statistics
            for split, dataset in datasets.items()
            if dataset.geometry_shuffle_statistics is not None
        },
        "interface_descriptor_shuffle_statistics": {
            split: dataset.interface_descriptor_shuffle_statistics
            for split, dataset in datasets.items()
            if dataset.interface_descriptor_shuffle_statistics is not None
        },
        "interface_descriptor_standardization_fit_split": (
            "train" if descriptor_store is not None else None
        ),
        "interface_descriptor_train_mean": (
            descriptor_mean_np.tolist() if descriptor_mean_np is not None else None
        ),
        "interface_descriptor_train_std": (
            descriptor_std_np.tolist() if descriptor_std_np is not None else None
        ),
        "interface_descriptor_feature_names": (
            descriptor_store.selected_feature_names if descriptor_store is not None else None
        ),
        "sealed_test_accessed": bool(args.evaluate_test),
        "project_validation_accessed": (
            args.validation_file is None or args.development_holdout_file is None
        ),
        "development_holdout_name": (
            args.development_holdout_name
            if args.development_holdout_file is not None
            else None
        ),
    }
    write_json_atomic(args.output_dir / "config.json", config)
    print(json.dumps(config, indent=2, sort_keys=True), flush=True)

    best_auprc = -math.inf
    best_epoch = 0
    stale_epochs = 0
    history = []
    started_at = time.time()
    for epoch in range(1, args.epochs + 1):
        samplers["train"].set_epoch(epoch)
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        accumulated_samples = 0
        optimizer.zero_grad(set_to_none=True)
        for batch in loaders["train"]:
            batch = move_batch(
                batch,
                device,
                confidence_mean,
                confidence_std,
                interface_descriptor_mean,
                interface_descriptor_std,
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = model_forward(model, batch)
                loss_sum = nn.functional.binary_cross_entropy_with_logits(
                    logits, batch["label"], reduction="sum"
                )
            loss_sum.backward()
            batch_samples = len(batch["label"])
            accumulated_samples += batch_samples
            train_loss_sum += float(loss_sum.item())
            train_count += batch_samples
            if accumulated_samples >= args.max_batch_size:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_samples)
                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_samples = 0
        if accumulated_samples:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_samples)
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        validation_loss, validation_records = evaluate(
            model,
            loaders["validation"],
            criterion,
            device,
            confidence_mean,
            confidence_std,
            interface_descriptor_mean,
            interface_descriptor_std,
        )
        validation_labels = np.asarray(
            [row["label"] for row in validation_records], dtype=np.int64
        )
        validation_probabilities = np.asarray(
            [row["probability"] for row in validation_records], dtype=np.float64
        )
        validation_auprc = float(
            average_precision_score(validation_labels, validation_probabilities)
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_count,
            "validation_loss": validation_loss,
            "validation_auprc": validation_auprc,
            "validation_auroc": float(
                roc_auc_score(validation_labels, validation_probabilities)
            ),
            "elapsed_seconds": time.time() - started_at,
        }
        history.append(row)
        with (args.output_dir / "history.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(row), delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(history)
        print(" ".join(f"{key}={value}" for key, value in row.items()), flush=True)

        if validation_auprc > best_auprc:
            best_auprc = validation_auprc
            best_epoch = epoch
            stale_epochs = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "validation_auprc": validation_auprc,
            }
            temporary = args.output_dir / "best_model.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(args.output_dir / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early_stopping_epoch={epoch}", flush=True)
                break

    checkpoint = torch.load(
        args.output_dir / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    split_records = {}
    split_losses = {}
    evaluation_splits = ["validation"]
    if args.development_holdout_file is not None:
        evaluation_splits.append(args.development_holdout_name)
    if args.evaluate_test:
        evaluation_splits.append("test")
    for split in evaluation_splits:
        split_losses[split], split_records[split] = evaluate(
            model,
            loaders[split],
            criterion,
            device,
            confidence_mean,
            confidence_std,
            interface_descriptor_mean,
            interface_descriptor_std,
        )
    validation_labels = np.asarray(
        [row["label"] for row in split_records["validation"]], dtype=np.int64
    )
    validation_probabilities = np.asarray(
        [row["probability"] for row in split_records["validation"]], dtype=np.float64
    )
    selected_threshold = best_f1_threshold(
        validation_labels, validation_probabilities
    )
    target_results = {
        split: macro_target_metrics(records)
        for split, records in split_records.items()
    }
    target_sequence_results = {
        split: macro_target_sequence_metrics(records)
        for split, records in split_records.items()
    }
    metrics = {
        "best_epoch": best_epoch,
        "best_validation_auprc_during_training": best_auprc,
        "selected_validation_f1_threshold": selected_threshold,
    }
    for split, records in split_records.items():
        split_metrics = ranking_metrics(records, selected_threshold)
        split_metrics["loss"] = split_losses[split]
        split_metrics.update(target_results[split]["summary"])
        split_metrics.update(target_sequence_results[split]["summary"])
        metrics[split] = split_metrics
        write_predictions(args.output_dir / f"{split}_predictions.tsv.gz", records)
    write_target_rows(
        args.output_dir / "per_target_metrics.tsv",
        {split: result["rows"] for split, result in target_results.items()},
    )
    write_target_sequence_rows(
        args.output_dir / "per_target_sequence_metrics.tsv",
        {
            split: result["rows"]
            for split, result in target_sequence_results.items()
        },
    )
    metrics["sealed_test_accessed"] = bool(args.evaluate_test)
    metrics["project_validation_accessed"] = config["project_validation_accessed"]
    write_json_atomic(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print("status=ok", flush=True)


if __name__ == "__main__":
    main()
