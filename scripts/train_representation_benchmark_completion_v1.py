#!/usr/bin/env python3
"""Train frozen non-AlphaFold adapters with a protected outer holdout."""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_representation_benchmark_completion_v1 import (
    ADAPTERS, CACHES, RUN, SEEDS, SPLITS, read_tsv, sha256, write_tsv,
)
from scripts.train_multimodal_v2 import set_seed, write_json_atomic, write_predictions
from scripts.summarize_representation_source_v1 import metrics as record_metrics


class DescriptorDataset(Dataset):
    def __init__(self, rows, values):
        self.rows = rows
        self.values = torch.from_numpy(np.asarray(values, dtype=np.float32))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {"sample_id": row["sample_id"], "values": self.values[index],
                "label": float(row["label"] == "positive")}


def descriptor_values(rows, cache, kind):
    index_key = "cache_index" if kind == "geometry_g4" else "sample_index"
    indices = {r["sample_id"]: int(r[index_key]) for r in read_tsv(cache / "index.tsv")}
    selected = [indices[r["sample_id"]] for r in rows]
    if kind == "geometry_g4":
        from scripts.train_interface_geometry_descriptor_v1 import load_definitions
        from training.interface_geometry_descriptor import feature_indices
        definitions = load_definitions(cache / "feature_names.tsv")
        columns = feature_indices(definitions, "g4")
        values = np.load(cache / "features.float32.npy", mmap_mode="r")[selected][:, columns]
        if values.shape[1] != 249 or not np.isfinite(values).all():
            raise ValueError("Invalid g4 descriptor matrix")
    else:
        with np.load(cache / "features.npz", allow_pickle=False) as archive:
            values = archive["raw"][selected]
        if values.shape[1] != 39 or np.isinf(values).any():
            raise ValueError("Invalid raw39 physical matrix")
    return np.asarray(values, dtype=np.float32)


def data_loader(adapter, path, rows, seed, workers, shuffle, scaler=None):
    if adapter == "atom_gnn":
        from training.interface_atom_data import InterfaceAtomDataset
        from scripts.train_interface_atom_e3nn_v1 import make_loader
        dataset = InterfaceAtomDataset(
            split_path=path, cache_dir=CACHES["atom"], confidence_path=RUN / "auxiliary.tsv.gz",
            variant="atom_chemistry", pair_gate_mode="none", label_permutation_seed=None,
        )
        weight_by_id = {r["sample_id"]: w for r, w in zip(dataset.rows, dataset.sampling_weights, strict=True)}
        dataset.rows = rows
        dataset.labels = [float(r["label"] == "positive") for r in rows]
        dataset.sampling_weights = [weight_by_id[r["sample_id"]] for r in rows]
        # Recreate size metadata after a smoke-test subset, without changing full runs.
        dataset.sizes = [dataset.store.sizes(r["sample_id"]) for r in rows]
        _, loader = make_loader(dataset, max_batch_size=8, max_message_edges=160000,
                                num_workers=workers, shuffle=shuffle, seed=seed)
        return loader
    options = {"num_workers": workers, "shuffle": shuffle,
               "generator": torch.Generator().manual_seed(seed),
               "pin_memory": torch.cuda.is_available()}
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    if adapter == "geometry_g4":
        values = descriptor_values(rows, CACHES["geometry"], adapter)
        return DataLoader(DescriptorDataset(rows, scaler.transform(values)), batch_size=128, **options)
    from scripts.train_dmasif_knn_region_factorial_v1 import FactorialDataset
    from scripts.train_dmasif_region_pool_v1 import collate
    variant = "knn8_global_capacity" if adapter == "dmasif_global" else "knn8_region_pool"
    dataset = FactorialDataset(rows, CACHES["point"], CACHES["knn"], CACHES["paired"], variant)
    return DataLoader(dataset, batch_size=8, collate_fn=collate, **options)


def make_model(adapter):
    if adapter == "geometry_g4":
        from scripts.train_interface_geometry_descriptor_v1 import TinyMLP
        return TinyMLP(249, 0.15)
    if adapter == "atom_gnn":
        from training.interface_atom_e3nn import InterfaceAtomE3NNClassifier
        return InterfaceAtomE3NNClassifier(
            hidden_scalars=16, hidden_vectors=4, message_layers=2, rbf_bins=16,
            classifier_dropout=0.2, interaction_mode="full", pair_gate_mode="none", pair_gate_alpha=1.0,
        )
    from scripts.train_dmasif_region_pool_v1 import RegionPoolClassifier
    return RegionPoolClassifier("global_capacity" if adapter == "dmasif_global" else "region_pool")


def move(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def forward(model, batch, adapter):
    if adapter == "atom_gnn":
        from scripts.train_interface_atom_e3nn_v1 import model_forward
        return model_forward(model, batch)
    return model(batch["values"] if adapter == "geometry_g4" else batch)


def predictions(model, loader, adapter, device, rows):
    model.eval()
    values = {}
    with torch.inference_mode():
        for batch in loader:
            probability = torch.sigmoid(forward(model, move(batch, device), adapter)).float().cpu().numpy()
            for sid, p in zip(batch["sample_id"], probability, strict=True):
                if sid in values or not np.isfinite(p):
                    raise ValueError(f"Duplicate or nonfinite prediction: {sid}")
                values[sid] = float(p)
    if set(values) != {r["sample_id"] for r in rows}:
        raise ValueError("Incomplete prediction coverage")
    return records_for(rows, [values[r["sample_id"]] for r in rows])


def records_for(rows, probabilities):
    return [{"sample_id": r["sample_id"], "label": int(r["label"] == "positive"),
             "target_name": r["target_name"], "target_sequence_id": r["target_sequence_id"],
             "target_cluster_id": r["target_cluster_id"], "probability": float(p)}
            for r, p in zip(rows, probabilities, strict=True)]


def train(args):
    paths = {split: SPLITS / f"fold{args.fold}" / f"{split}.tsv.gz"
             for split in ("train", "validation", "outer_holdout")}
    rows = {split: read_tsv(path) for split, path in paths.items()}
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no silent CPU fallback")
    if args.smoke:
        rows = {split: [next(r for r in rs if r["label"] == label)
                        for label in ("positive", "negative")] for split, rs in rows.items()}
        args.num_workers = 0
    elif args.output_dir.exists():
        raise FileExistsError(f"Incomplete/existing run requires review: {args.output_dir}")
    if not args.smoke:
        args.output_dir.mkdir(parents=True)
    started = time.time()
    config = {
        "adapter": args.adapter, "seed": args.seed, "outer_fold": args.fold,
        "early_stop_fold": (args.fold + 1) % 5, "device": str(device),
        "torch_version": torch.__version__, "python": sys.executable,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "precision": "float32", "matmul_precision": "high", "num_workers": args.num_workers,
        "split_sizes": {key: len(value) for key, value in rows.items()},
        "split_sha256": {key: sha256(value) for key, value in paths.items()},
        "epochs": 50, "patience": 8, "learning_rate": 3e-4,
        "weight_decay": 1e-3, "gradient_clip": 1.0,
        "project_validation_accessed": False, "sealed_test_accessed": False,
        "outer_used_for_selection": False, "smoke": args.smoke,
    }
    if args.adapter == "rosetta39":
        from scripts.run_physical_atlas_grouped_oof_v1 import estimator
        x = descriptor_values(rows["train"], CACHES["rosetta"], args.adapter)
        y = np.asarray([r["label"] == "positive" for r in rows["train"]], dtype=int)
        model = estimator("hist_gradient_boosting", args.seed)
        if args.smoke:
            model.set_params(classifier__max_iter=2, classifier__min_samples_leaf=1)
        model.fit(x, y)
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        loaded = joblib.load(buffer)
        np.testing.assert_allclose(loaded.predict_proba(x), model.predict_proba(x))
        if args.smoke:
            print("smoke=ok adapter=rosetta39 fit_predict_roundtrip=true", flush=True)
            return
        config.update(precision="float64_estimator", epochs=None, patience=None,
                      optimizer="hist_gradient_boosting", class_weight="balanced",
                      estimator_parameters=str(model.get_params()), trainable_parameters=None)
        joblib.dump(model, args.output_dir / "best_model.joblib")
        results = {}
        for split in ("validation", "outer_holdout"):
            values = descriptor_values(rows[split], CACHES["rosetta"], args.adapter)
            records = records_for(rows[split], model.predict_proba(values)[:, 1])
            write_predictions(args.output_dir / f"{split}_predictions.tsv.gz", records)
            results[split] = record_metrics(records)
        write_json_atomic(args.output_dir / "config.json", config)
        write_json_atomic(args.output_dir / "metrics.json", {**results, "fit_seconds": time.time() - started})
        return
    scaler = None
    if args.adapter == "geometry_g4":
        scaler = StandardScaler().fit(descriptor_values(rows["train"], CACHES["geometry"], args.adapter))
        if not args.smoke:
            joblib.dump(scaler, args.output_dir / "preprocessor.joblib")
    loaders = {split: data_loader(args.adapter, paths[split], rows[split], args.seed + offset,
                                 args.num_workers, split == "train", scaler)
               for offset, split in enumerate(("train", "validation"))}
    model = make_model(args.adapter).to(device)
    config["trainable_parameters"] = sum(p.numel() for p in model.parameters())
    config["batch_size"] = 128 if args.adapter == "geometry_g4" else 8
    config["max_message_edges"] = 160000 if args.adapter == "atom_gnn" else None
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    if args.smoke:
        batch = move(next(iter(loaders["train"])), device)
        loss = criterion(forward(model, batch, args.adapter), batch["label"].float())
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError("Nonfinite smoke loss/gradient")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = forward(model, batch, args.adapter)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        loaded = make_model(args.adapter).to(device)
        loaded.load_state_dict(torch.load(buffer, map_location=device, weights_only=True))
        loaded.eval()
        with torch.no_grad():
            torch.testing.assert_close(forward(loaded, batch, args.adapter), expected)
        print(f"smoke=ok adapter={args.adapter} parameters={config['trainable_parameters']} loss={loss.item()}", flush=True)
        return
    write_json_atomic(args.output_dir / "config.json", config)
    print(json.dumps(config, sort_keys=True), flush=True)
    best, stale, best_epoch = -math.inf, 0, 0
    history = []
    for epoch in range(1, 51):
        sampler = loaders["train"].batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        total, count = 0.0, 0
        for batch in loaders["train"]:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(forward(model, batch, args.adapter), batch["label"].float())
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(batch["label"])
            count += len(batch["label"])
        records = predictions(model, loaders["validation"], args.adapter, device, rows["validation"])
        score = average_precision_score([r["label"] for r in records], [r["probability"] for r in records])
        history.append({"epoch": epoch, "train_loss": total / count, "validation_auprc": float(score),
                        "elapsed_seconds": time.time() - started})
        write_tsv(args.output_dir / "history.tsv", history)
        print(json.dumps(history[-1]), flush=True)
        if score > best:
            best, stale, best_epoch = float(score), 0, epoch
            temporary = args.output_dir / "best_model.pt.tmp"
            torch.save({"model_state_dict": model.state_dict(), "config": config,
                        "epoch": epoch, "validation_auprc": best}, temporary)
            temporary.replace(args.output_dir / "best_model.pt")
        else:
            stale += 1
            if stale >= 8:
                break
    model.load_state_dict(torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)["model_state_dict"])
    # Outer features/predictions are first evaluated after checkpoint selection.
    loaders["outer_holdout"] = data_loader(args.adapter, paths["outer_holdout"], rows["outer_holdout"],
                                           args.seed + 2, args.num_workers, False, scaler)
    results = {}
    for split in ("validation", "outer_holdout"):
        records = predictions(model, loaders[split], args.adapter, device, rows[split])
        write_predictions(args.output_dir / f"{split}_predictions.tsv.gz", records)
        results[split] = record_metrics(records)
    write_json_atomic(args.output_dir / "metrics.json", {
        **results, "best_epoch": best_epoch, "best_early_stop_auprc": best,
        "fit_seconds": time.time() - started, "sealed_test_accessed": False,
    })
    print("status=ok outer_predictions_written=true", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=ADAPTERS[1:], required=True)
    parser.add_argument("--fold", choices=range(5), type=int, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    train(parser.parse_args())
