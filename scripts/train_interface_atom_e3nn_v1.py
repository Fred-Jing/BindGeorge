#!/usr/bin/env python3
"""Train an interface-centered atom-level e3nn classifier on train/validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import e3nn
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_multimodal_v2 import (  # noqa: E402
    best_f1_threshold,
    macro_target_metrics,
    ranking_metrics,
    set_seed,
    write_json_atomic,
    write_predictions,
    write_target_rows,
)
from training.interface_atom_data import (  # noqa: E402
    ATOM_VARIANTS,
    PAIR_GATE_MODES,
    GraphBudgetBatchSampler,
    InterfaceAtomDataset,
    collate_interface_atoms,
)
from training.interface_atom_e3nn import (  # noqa: E402
    INTERACTION_MODES,
    InterfaceAtomE3NNClassifier,
)


DEFAULT_SPLIT_ROOT = (
    PROJECT_ROOT
    / "processed/v1/training_10x/cohorts/ipae_min_le_2p5/known_target"
)
DEFAULT_CACHE = (
    PROJECT_ROOT / "features/af3_v1/cohorts/ipae_min_le_2p5_interface_atom_v1"
)
DEFAULT_CONFIDENCE = (
    PROJECT_ROOT
    / "processed/v1/training_10x/cohorts/ipae_min_le_2p5/cohort.tsv.gz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=ATOM_VARIANTS, required=True)
    parser.add_argument("--interaction-mode", choices=INTERACTION_MODES, default="full")
    parser.add_argument("--pair-gate-mode", choices=PAIR_GATE_MODES, default="none")
    parser.add_argument("--pair-gate-alpha", type=float, default=1.0)
    parser.add_argument("--shuffle-train-labels", action="store_true")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--atom-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--confidence-table", type=Path, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-scalars", type=int, default=16)
    parser.add_argument("--hidden-vectors", type=int, default=4)
    parser.add_argument("--message-layers", type=int, default=2)
    parser.add_argument("--rbf-bins", type=int, default=16)
    parser.add_argument("--classifier-dropout", type=float, default=0.2)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-message-edges", type=int, default=160_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def make_loader(
    dataset: InterfaceAtomDataset,
    *,
    max_batch_size: int,
    max_message_edges: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> tuple[GraphBudgetBatchSampler, DataLoader]:
    sampler = GraphBudgetBatchSampler(
        dataset.sizes,
        max_batch_size=max_batch_size,
        max_message_edges=max_message_edges,
        shuffle=shuffle,
        seed=seed,
    )
    options: dict[str, object] = {}
    if num_workers > 0:
        options.update(prefetch_factor=2, persistent_workers=True)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_interface_atoms,
        pin_memory=torch.cuda.is_available(),
        **options,
    )
    return sampler, loader


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def model_forward(model: nn.Module, batch: dict[str, object]) -> torch.Tensor:
    keys = (
        "node_position",
        "node_element",
        "node_residue_type",
        "node_atom_name",
        "node_backbone",
        "node_role",
        "gate_node_element",
        "gate_node_atom_name",
        "gate_node_backbone",
        "node_graph_index",
        "message_source",
        "message_destination",
        "message_relation",
        "core_target_node",
        "core_binder_node",
        "core_graph_index",
        "graph_count",
    )
    return model(**{key: batch[key] for key in keys})


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, list[dict[str, object]]]:
    model.eval()
    records = []
    loss_sum = 0.0
    count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model_forward(model, batch)
            loss = criterion(logits, batch["label"])
            probabilities = torch.sigmoid(logits).cpu().numpy()
            labels = batch["label"].cpu().numpy()
            weights = batch["sampling_weight"].cpu().numpy()
            loss_sum += float(loss.item()) * len(labels)
            count += len(labels)
            for index, probability in enumerate(probabilities):
                records.append(
                    {
                        "sample_id": batch["sample_id"][index],
                        "target_name": batch["target_name"][index],
                        "label": int(labels[index]),
                        "sampling_weight": float(weights[index]),
                        "probability": float(probability),
                    }
                )
    return loss_sum / count, records


def main() -> None:
    args = parse_args()
    if (args.output_dir / "best_model.pt").exists():
        raise FileExistsError(f"Run already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {
        split: InterfaceAtomDataset(
            split_path=args.split_root / f"{split}.tsv.gz",
            cache_dir=args.atom_cache,
            confidence_path=args.confidence_table,
            variant=args.variant,
            pair_gate_mode=args.pair_gate_mode,
            label_permutation_seed=args.seed if split == "train" and args.shuffle_train_labels else None,
        )
        for split in ("train", "validation")
    }
    samplers = {}
    loaders = {}
    for offset, split in enumerate(("train", "validation")):
        samplers[split], loaders[split] = make_loader(
            datasets[split],
            max_batch_size=args.max_batch_size,
            max_message_edges=args.max_message_edges,
            num_workers=args.num_workers,
            shuffle=split == "train",
            seed=args.seed + offset,
        )
    model = InterfaceAtomE3NNClassifier(
        hidden_scalars=args.hidden_scalars,
        hidden_vectors=args.hidden_vectors,
        message_layers=args.message_layers,
        rbf_bins=args.rbf_bins,
        classifier_dropout=args.classifier_dropout,
        interaction_mode=args.interaction_mode,
        pair_gate_mode=args.pair_gate_mode,
        pair_gate_alpha=args.pair_gate_alpha,
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
        "torch_version": torch.__version__,
        "e3nn_version": e3nn.__version__,
        "precision": "float32",
        "trainable_parameters": trainable_parameters,
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "batches_per_epoch": {split: len(sampler) for split, sampler in samplers.items()},
        "sealed_test_accessed": False,
        "validation_labels_shuffled": False,
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
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model_forward(model, batch)
            loss = criterion(logits, batch["label"])
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(batch["label"])
            train_count += len(batch["label"])
        validation_loss, validation_records = evaluate(
            model, loaders["validation"], criterion, device
        )
        labels = np.asarray([row["label"] for row in validation_records])
        probabilities = np.asarray(
            [row["probability"] for row in validation_records]
        )
        validation_auprc = float(average_precision_score(labels, probabilities))
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_count,
            "validation_loss": validation_loss,
            "validation_auprc": validation_auprc,
            "validation_auroc": float(roc_auc_score(labels, probabilities)),
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
    validation_loss, records = evaluate(
        model, loaders["validation"], criterion, device
    )
    labels = np.asarray([row["label"] for row in records])
    probabilities = np.asarray([row["probability"] for row in records])
    threshold = best_f1_threshold(labels, probabilities)
    target_result = macro_target_metrics(records)
    metrics = {
        "best_epoch": best_epoch,
        "best_validation_auprc_during_training": best_auprc,
        "selected_validation_f1_threshold": threshold,
        "validation": {
            **ranking_metrics(records, threshold),
            "loss": validation_loss,
            **target_result["summary"],
        },
        "sealed_test_accessed": False,
    }
    write_predictions(args.output_dir / "validation_predictions.tsv.gz", records)
    write_target_rows(
        args.output_dir / "per_target_metrics.tsv",
        {"validation": target_result["rows"]},
    )
    write_json_atomic(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print("status=ok sealed_test_accessed=false", flush=True)


if __name__ == "__main__":
    main()
