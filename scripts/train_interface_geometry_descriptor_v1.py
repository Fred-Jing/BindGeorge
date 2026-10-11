#!/usr/bin/env python3
"""Train small classifiers on deterministic interface geometry descriptors."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
from training.interface_geometry_descriptor import (  # noqa: E402
    DESCRIPTOR_VARIANTS,
    FeatureDefinition,
    feature_indices,
)


DEFAULT_SPLIT_ROOT = (
    PROJECT_ROOT / "processed/v1/training_10x/cohorts/ipae_min_le_2p5/known_target"
)
DEFAULT_CACHE = (
    PROJECT_ROOT
    / "features/af3_v1/cohorts/ipae_min_le_2p5_interface_geometry_descriptor_v1"
)
DEFAULT_CONFIDENCE = (
    PROJECT_ROOT / "processed/v1/training_10x/cohorts/ipae_min_le_2p5/cohort.tsv.gz"
)
MODEL_TYPES = ("logistic", "hist_gradient_boosting", "tiny_mlp")


class TinyMLP(nn.Module):
    def __init__(self, input_dimension: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def state_dict_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_TYPES, required=True)
    parser.add_argument("--variant", choices=DESCRIPTOR_VARIANTS, required=True)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--descriptor-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--confidence-table", type=Path, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--control-seed",
        type=int,
        help="Independent seed for feature shuffles; defaults to --seed",
    )
    parser.add_argument("--smoke-samples-per-class", type=int)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--hist-iterations", type=int, default=200)
    parser.add_argument("--hist-learning-rate", type=float, default=0.05)
    parser.add_argument("--hist-max-leaf-nodes", type=int, default=15)
    parser.add_argument("--hist-l2", type=float, default=0.1)
    parser.add_argument("--mlp-dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_definitions(path: Path) -> list[FeatureDefinition]:
    rows = read_tsv(path)
    for expected, row in enumerate(rows):
        if int(row["feature_index"]) != expected:
            raise ValueError("Non-contiguous feature indices")
    return [FeatureDefinition(row["name"], row["group"], row["subgroup"]) for row in rows]


def apply_control(
    values: np.ndarray,
    definitions: list[FeatureDefinition],
    variant: str,
    seed: int,
    group_ids: list[str] | None = None,
) -> np.ndarray:
    output = np.array(values, dtype=np.float32, copy=True)
    atomic_surface_components = {"clearance", "gap", "normal"}
    atomic_surface_active = {
        "asp_pose_edges_zero": set(),
        "asp_pose_edges_uniform_clearance": {"clearance"},
        "asp_pose_edges_uniform_gap": {"gap"},
        "asp_pose_edges_uniform_normal": {"normal"},
        "asp_pose_edges_uniform_all": atomic_surface_components,
        "asp_pose_edges_vdw_clearance": {"clearance"},
        "asp_pose_edges_vdw_gap": {"gap"},
        "asp_pose_edges_vdw_normal": {"normal"},
        "asp_pose_edges_vdw_all": atomic_surface_components,
        "asp_clearance_confirm_zero": set(),
        "asp_clearance_confirm_real": {"clearance"},
        "asp_clearance_confirm_shuffled": {"clearance"},
    }
    if variant in atomic_surface_active:
        mode = "p_vdw" if "vdw" in variant else "p_uniform"
        inactive = atomic_surface_components - atomic_surface_active[variant]
        columns = np.asarray(
            [
                index
                for index, definition in enumerate(definitions)
                if definition.group == mode and definition.subgroup in inactive
            ],
            dtype=np.int64,
        )
        if columns.size:
            output[:, columns] = 0.0
    if variant in {
        "asp_pose_edges_uniform_shuffled",
        "asp_pose_edges_vdw_shuffled",
        "asp_clearance_confirm_shuffled",
    }:
        if group_ids is None or len(group_ids) != len(output):
            raise ValueError("Within-target shuffle requires one group ID per sample")
        mode = "p_vdw" if "vdw" in variant else "p_uniform"
        columns = np.asarray(
            [
                index
                for index, definition in enumerate(definitions)
                if definition.group == mode
                and (
                    variant != "asp_clearance_confirm_shuffled"
                    or definition.subgroup == "clearance"
                )
            ],
            dtype=np.int64,
        )
        rng = np.random.default_rng(seed)
        group_array = np.asarray(group_ids)
        for group_id in sorted(set(group_ids)):
            rows = np.flatnonzero(group_array == group_id)
            output[np.ix_(rows, columns)] = output[
                np.ix_(rng.permutation(rows), columns)
            ]
    if variant == "asp_clearance_confirm_capacity":
        capacity_columns = np.asarray(
            [
                index
                for index, definition in enumerate(definitions)
                if definition.group == "p_capacity"
            ],
            dtype=np.int64,
        )
        if capacity_columns.size != 36:
            raise ValueError("Clearance capacity control requires 36 capacity slots")
        output[:, capacity_columns[12:]] = 0.0
    contact_axis_groups = {"c0_separation", "c1_footprint", "c2_relief"}
    contact_axis_active = {
        "cai_pose_zero": set(),
        "cai_pose_separation": {"c0_separation"},
        "cai_pose_footprint": {"c1_footprint"},
        "cai_pose_relief": {"c2_relief"},
        "cai_pose_all": contact_axis_groups,
        "cai_pose_edges_zero": set(),
        "cai_pose_edges_all": contact_axis_groups,
    }
    if variant in contact_axis_active:
        inactive = contact_axis_groups - contact_axis_active[variant]
        columns = np.asarray(
            [
                index
                for index, definition in enumerate(definitions)
                if definition.group in inactive
            ],
            dtype=np.int64,
        )
        if columns.size:
            output[:, columns] = 0.0
    if variant in {"cai_pose_all_shuffled", "cai_pose_edges_all_shuffled"}:
        if group_ids is None or len(group_ids) != len(output):
            raise ValueError("Within-target shuffle requires one group ID per sample")
        columns = np.asarray(
            [
                index
                for index, definition in enumerate(definitions)
                if definition.group in contact_axis_groups
            ]
        )
        rng = np.random.default_rng(seed)
        group_array = np.asarray(group_ids)
        for group_id in sorted(set(group_ids)):
            rows = np.flatnonzero(group_array == group_id)
            output[np.ix_(rows, columns)] = output[
                np.ix_(rng.permutation(rows), columns)
            ]
    shuffled_group = {
        "g2_frame_shuffled": "g2_pose",
        "g4_identity_shuffled": "g4_identity",
        "g2_identity_without_size_pose_shuffled": "g2_pose",
        "g2_identity_without_size_identity_shuffled": "g4_identity",
    }.get(variant)
    if shuffled_group is not None:
        columns = np.asarray(
            [index for index, definition in enumerate(definitions) if definition.group == shuffled_group]
        )
        permutation = np.random.default_rng(seed).permutation(len(output))
        output[:, columns] = output[permutation][:, columns]
    if variant == "g1_identity_without_size_within_target_shuffled":
        if group_ids is None or len(group_ids) != len(output):
            raise ValueError("Within-target shuffle requires one group ID per sample")
        columns = np.asarray(
            [index for index, definition in enumerate(definitions) if definition.group == "g4_identity"]
        )
        rng = np.random.default_rng(seed)
        for group_id in sorted(set(group_ids)):
            rows = np.asarray(
                [index for index, candidate in enumerate(group_ids) if candidate == group_id]
            )
            output[np.ix_(rows, columns)] = output[np.ix_(rng.permutation(rows), columns)]
    if variant in {
        "surface_full_normals_within_target_shuffled",
        "surface_full_within_target_shuffled",
    }:
        if group_ids is None or len(group_ids) != len(output):
            raise ValueError("Within-target shuffle requires one group ID per sample")
        if variant == "surface_full_normals_within_target_shuffled":
            shuffled_groups = {"s2_normals", "s3_complementarity"}
        else:
            shuffled_groups = {
                "s0_distance", "s1_local_shape", "s2_normals", "s3_complementarity"
            }
        columns = np.asarray(
            [
                index
                for index, definition in enumerate(definitions)
                if definition.group in shuffled_groups
            ]
        )
        rng = np.random.default_rng(seed)
        group_array = np.asarray(group_ids)
        for group_id in sorted(set(group_ids)):
            rows = np.flatnonzero(group_array == group_id)
            output[np.ix_(rows, columns)] = output[
                np.ix_(rng.permutation(rows), columns)
            ]
    if variant == "dcg_full_shuffled":
        if group_ids is None or len(group_ids) != len(output):
            raise ValueError("Within-target shuffle requires one group ID per sample")
        columns = np.asarray(
            [
                index for index, definition in enumerate(definitions)
                if definition.group in {"d1_conditioned_pose", "d2_spectral"}
            ]
        )
        rng = np.random.default_rng(seed)
        group_array = np.asarray(group_ids)
        for group_id in sorted(set(group_ids)):
            rows = np.flatnonzero(group_array == group_id)
            output[np.ix_(rows, columns)] = output[
                np.ix_(rng.permutation(rows), columns)
            ]
    if variant in {
        "dcg_pose_spectral_all_shuffled",
        "dcg_pose_spectral_normalized_shuffled",
    }:
        if group_ids is None or len(group_ids) != len(output):
            raise ValueError("Within-target shuffle requires one group ID per sample")
        columns = np.asarray(
            [
                index
                for index, definition in enumerate(definitions)
                if definition.group == "d2_spectral"
                and (
                    variant == "dcg_pose_spectral_all_shuffled"
                    or "singular_value_" not in definition.name
                )
            ]
        )
        rng = np.random.default_rng(seed)
        group_array = np.asarray(group_ids)
        for group_id in sorted(set(group_ids)):
            rows = np.flatnonzero(group_array == group_id)
            output[np.ix_(rows, columns)] = output[
                np.ix_(rng.permutation(rows), columns)
            ]
    if variant in {
        "dcg_pose_occupancy_shuffled",
        "dcg_pose_occupancy_variable_shuffled",
    }:
        if group_ids is None or len(group_ids) != len(output):
            raise ValueError("Within-target shuffle requires one group ID per sample")
        include_constant_12a = variant == "dcg_pose_occupancy_shuffled"
        columns = np.asarray(
            [
                index
                for index, definition in enumerate(definitions)
                if definition.group == "d2_spectral"
                and (
                    definition.name.endswith("edge_density")
                    or definition.name.endswith(
                        ("active_target_fraction", "active_binder_fraction")
                    )
                    and (include_constant_12a or definition.subgroup != "le_12a")
                )
            ]
        )
        rng = np.random.default_rng(seed)
        group_array = np.asarray(group_ids)
        for group_id in sorted(set(group_ids)):
            rows = np.flatnonzero(group_array == group_id)
            output[np.ix_(rows, columns)] = output[
                np.ix_(rng.permutation(rows), columns)
            ]
    if variant == "dcg_pose_edges_matched_shuffled":
        if group_ids is None or len(group_ids) != len(output):
            raise ValueError("Within-target shuffle requires one group ID per sample")
        columns = np.asarray(
            [
                index
                for index, definition in enumerate(definitions)
                if definition.group == "d2_spectral"
                and definition.name.endswith("edge_density")
            ]
        )
        rng = np.random.default_rng(seed)
        group_array = np.asarray(group_ids)
        for group_id in sorted(set(group_ids)):
            rows = np.flatnonzero(group_array == group_id)
            output[np.ix_(rows, columns)] = output[
                np.ix_(rng.permutation(rows), columns)
            ]
    if variant == "dcg_pose_edges_matched_zero":
        columns = np.asarray(
            [
                index
                for index, definition in enumerate(definitions)
                if definition.group == "d2_spectral"
                and definition.name.endswith("edge_density")
            ]
        )
        output[:, columns] = 0.0
    return output


def load_data(args: argparse.Namespace) -> tuple[dict[str, dict[str, object]], list[FeatureDefinition], np.ndarray]:
    metadata = json.loads((args.descriptor_cache / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") not in {
        "interface_geometry_descriptor_v1",
        "contact_axis_interface_v1",
        "atomic_surface_packing_v1",
    }:
        raise ValueError("Unexpected descriptor cache schema")
    if metadata.get("status") != "complete":
        raise ValueError("Descriptor cache is incomplete")
    if metadata.get("sealed_test_accessed") is not False:
        raise ValueError("Descriptor cache does not certify sealed-test isolation")
    full_features = np.load(args.descriptor_cache / "features.float32.npy", mmap_mode="r")
    definitions = load_definitions(args.descriptor_cache / "feature_names.tsv")
    if full_features.shape[1] != len(definitions):
        raise ValueError("Feature matrix/schema width mismatch")
    cache_rows = read_tsv(args.descriptor_cache / "index.tsv")
    cache_index = {row["sample_id"]: int(row["cache_index"]) for row in cache_rows}
    if len(cache_index) != len(cache_rows):
        raise ValueError("Duplicate descriptor cache sample IDs")
    confidence = {row["sample_ids"]: row for row in read_tsv(args.confidence_table)}
    selected_columns = feature_indices(definitions, args.variant)
    datasets = {}
    for split_offset, split in enumerate(("train", "validation")):
        rows = read_tsv(args.split_root / f"{split}.tsv.gz")
        if args.smoke_samples_per_class is not None:
            if args.smoke_samples_per_class < 1:
                raise ValueError("--smoke-samples-per-class must be positive")
            rows = [
                row
                for label in ("negative", "positive")
                for row in [entry for entry in rows if entry["label"] == label][
                    : args.smoke_samples_per_class
                ]
            ]
        missing = [row["sample_id"] for row in rows if row["sample_id"] not in cache_index]
        if missing:
            raise ValueError(f"Missing {split} descriptors: {len(missing)}")
        indices = np.asarray([cache_index[row["sample_id"]] for row in rows])
        unselected = np.asarray(full_features[indices], dtype=np.float32)
        control_seed = args.seed if args.control_seed is None else args.control_seed
        controlled = apply_control(
            unselected,
            definitions,
            args.variant,
            control_seed + 1009 * split_offset,
            group_ids=[row["target_sequence_id"] for row in rows],
        )
        labels = np.asarray([row["label"] == "positive" for row in rows], dtype=np.float32)
        weights = np.asarray(
            [float(confidence[row["sample_id"]]["sampling_weight"]) for row in rows],
            dtype=np.float32,
        )
        datasets[split] = {
            "rows": rows,
            "features": controlled[:, selected_columns],
            "labels": labels,
            "weights": weights,
        }
    return datasets, definitions, selected_columns


def records_for(
    dataset: dict[str, object], probabilities: np.ndarray
) -> list[dict[str, object]]:
    rows = dataset["rows"]
    labels = dataset["labels"]
    weights = dataset["weights"]
    return [
        {
            "sample_id": row["sample_id"],
            "target_name": row["target_name"],
            "target_sequence_id": row["target_sequence_id"],
            "label": int(labels[index]),
            "sampling_weight": float(weights[index]),
            "probability": float(probabilities[index]),
        }
        for index, row in enumerate(rows)
    ]


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
        {"target_sequence_id": row["target_name"], **{key: value for key, value in row.items() if key != "target_name"}}
        for row in result["rows"]
    ]
    return {"summary": summary, "rows": rows}


def write_target_sequence_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "target_sequence_id", "count", "positive", "negative", "auroc", "auprc",
        "weighted_auroc", "weighted_auprc",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def train_sklearn(
    args: argparse.Namespace,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if args.model == "logistic":
        model = LogisticRegression(
            C=args.logistic_c, max_iter=2000, random_state=args.seed, solver="lbfgs"
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=args.hist_iterations,
            learning_rate=args.hist_learning_rate,
            max_leaf_nodes=args.hist_max_leaf_nodes,
            l2_regularization=args.hist_l2,
            early_stopping=False,
            random_state=args.seed,
        )
    started = time.time()
    model.fit(train_x, train_y)
    joblib.dump(model, args.output_dir / "model.joblib")
    iteration_value = np.asarray(getattr(model, "n_iter_", 1)).max()
    details = {
        "fit_seconds": time.time() - started,
        "iterations": int(iteration_value),
    }
    return model.predict_proba(train_x)[:, 1], model.predict_proba(validation_x)[:, 1], details


def probabilities(model: nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(values), 1024):
            tensor = torch.from_numpy(values[start : start + 1024]).to(device)
            output.append(torch.sigmoid(model(tensor)).cpu().numpy())
    return np.concatenate(output)


def train_mlp(
    args: argparse.Namespace,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMLP(train_x.shape[1], args.mlp_dropout).to(device)
    initial_state_sha256 = state_dict_sha256(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_auprc = -math.inf
    best_epoch = 0
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch_y)
        validation_probability = probabilities(model, validation_x, device)
        validation_auprc = float(average_precision_score(validation_y, validation_probability))
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / len(train_y),
            "validation_auprc": validation_auprc,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        if validation_auprc > best_auprc:
            best_auprc = validation_auprc
            best_epoch = epoch
            stale = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "validation_auprc": validation_auprc},
                args.output_dir / "best_model.pt.tmp",
            )
            (args.output_dir / "best_model.pt.tmp").replace(args.output_dir / "best_model.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    with (args.output_dir / "history.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return (
        probabilities(model, train_x, device),
        probabilities(model, validation_x, device),
        {
            "device": str(device), "trainable_parameters": parameter_count,
            "initial_state_sha256": initial_state_sha256,
            "best_epoch": best_epoch, "best_validation_auprc_during_training": best_auprc,
            "fit_seconds": time.time() - started,
        },
    )


def main() -> None:
    args = parse_args()
    completion_file = args.output_dir / "metrics.json"
    if completion_file.exists():
        raise FileExistsError(f"Run already complete: {args.output_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Incomplete non-empty run requires review: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    datasets, definitions, columns = load_data(args)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(datasets["train"]["features"]).astype(np.float32)
    validation_x = scaler.transform(datasets["validation"]["features"]).astype(np.float32)
    np.savez(
        args.output_dir / "scaler.npz",
        mean=scaler.mean_, scale=scaler.scale_, selected_columns=columns,
    )
    train_y = datasets["train"]["labels"]
    validation_y = datasets["validation"]["labels"]
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "feature_count": len(columns),
        "selected_feature_names": [definitions[index].name for index in columns],
        "split_sizes": {split: len(data["labels"]) for split, data in datasets.items()},
        "standardization_fit_split": "train",
        "sealed_test_accessed": False,
        "validation_labels_shuffled": False,
    }
    write_json_atomic(args.output_dir / "config.json", config)
    if args.model == "tiny_mlp":
        train_probability, validation_probability, fit_details = train_mlp(
            args, train_x, train_y, validation_x, validation_y
        )
    else:
        train_probability, validation_probability, fit_details = train_sklearn(
            args, train_x, train_y, validation_x
        )
    train_records = records_for(datasets["train"], train_probability)
    validation_records = records_for(datasets["validation"], validation_probability)
    threshold = best_f1_threshold(train_y.astype(np.int64), train_probability)
    validation_target = macro_target_metrics(validation_records)
    validation_target_sequence = macro_target_sequence_metrics(validation_records)
    metrics = {
        "model": args.model,
        "variant": args.variant,
        "seed": args.seed,
        "fit": fit_details,
        "selected_train_f1_threshold": threshold,
        "train": ranking_metrics(train_records, threshold),
        "validation": {
            **ranking_metrics(validation_records, threshold),
            "log_loss": float(log_loss(validation_y, validation_probability)),
            **validation_target["summary"],
            **validation_target_sequence["summary"],
        },
        "sealed_test_accessed": False,
    }
    write_predictions(args.output_dir / "validation_predictions.tsv.gz", validation_records)
    write_target_rows(args.output_dir / "per_target_metrics.tsv", {"validation": validation_target["rows"]})
    write_target_sequence_rows(
        args.output_dir / "per_target_sequence_metrics.tsv", validation_target_sequence["rows"]
    )
    write_json_atomic(completion_file, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print("status=ok sealed_test_accessed=false", flush=True)


if __name__ == "__main__":
    main()
