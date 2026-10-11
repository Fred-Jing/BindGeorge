from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


SCHEMA_VERSION = "af3_interface_atom_graph_v1"
ATOM_VARIANTS = (
    "atom_geometry_only",
    "atom_identity_only",
    "residue_identity_only",
    "atom_chemistry",
    "atom_chemistry_shuffled",
)
PAIR_GATE_MODES = ("none", "unknown", "real", "shuffled")
NODE_FIELDS = (
    "node_position",
    "node_element",
    "node_residue_type",
    "node_atom_name",
    "node_backbone",
    "node_role",
)
GATE_NODE_FIELDS = (
    "gate_node_element",
    "gate_node_atom_name",
    "gate_node_backbone",
)
MESSAGE_FIELDS = (
    "message_source",
    "message_destination",
    "message_relation",
)
CORE_FIELDS = ("core_target_node", "core_binder_node")


def read_tsv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def chemistry_permutation(sample_id: str, role: int, count: int) -> np.ndarray:
    digest = hashlib.sha256(
        f"interface-atom-chemistry-v1:{sample_id}:{role}".encode()
    ).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return generator.permutation(count)


class InterfaceAtomStore:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.metadata = json.loads(
            (self.cache_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if self.metadata.get("status") != "complete":
            raise ValueError(f"Interface atom cache is incomplete: {self.cache_dir}")
        if self.metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unexpected interface atom schema: {self.cache_dir}")
        self.index: dict[str, dict[str, int | str]] = {}
        with (self.cache_dir / "index.tsv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                sample_id = row["sample_id"]
                if sample_id in self.index:
                    raise ValueError(f"Duplicate interface atom sample: {sample_id}")
                self.index[sample_id] = {
                    "complex_feature_id": row["complex_feature_id"],
                    **{
                        key: int(row[key])
                        for key in (
                            "node_offset",
                            "node_count",
                            "message_offset",
                            "message_count",
                            "core_offset",
                            "core_count",
                            "target_interface_residue_count",
                            "binder_interface_residue_count",
                            "target_selected_residue_count",
                            "binder_selected_residue_count",
                        )
                    },
                }
        self._arrays: dict[str, np.ndarray] = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_arrays"] = {}
        return state

    def _array(self, name: str) -> np.ndarray:
        if name not in self._arrays:
            self._arrays[name] = np.load(
                Path(self.metadata["arrays"][name]["path"]), mmap_mode="r"
            )
        return self._arrays[name]

    def sizes(self, sample_id: str) -> tuple[int, int, int]:
        row = self.index[sample_id]
        return (
            int(row["node_count"]),
            int(row["message_count"]),
            int(row["core_count"]),
        )

    def get(self, sample_id: str) -> dict[str, torch.Tensor]:
        row = self.index[sample_id]
        node_offset = int(row["node_offset"])
        message_offset = int(row["message_offset"])
        core_offset = int(row["core_offset"])
        slices = {
            **{
                name: slice(node_offset, node_offset + int(row["node_count"]))
                for name in NODE_FIELDS
            },
            **{
                name: slice(
                    message_offset, message_offset + int(row["message_count"])
                )
                for name in MESSAGE_FIELDS
            },
            **{
                name: slice(core_offset, core_offset + int(row["core_count"]))
                for name in CORE_FIELDS
            },
        }
        output = {}
        for name, array_slice in slices.items():
            dtype = np.float32 if name == "node_position" else np.int64
            output[name] = torch.from_numpy(
                np.array(self._array(name)[array_slice], dtype=dtype, copy=True)
            )
        return output


def apply_chemistry_control(
    graph: dict[str, torch.Tensor], sample_id: str, variant: str
) -> None:
    chemistry_fields = (
        "node_element",
        "node_residue_type",
        "node_atom_name",
        "node_backbone",
    )
    if variant == "atom_geometry_only":
        for field in chemistry_fields:
            graph[field].zero_()
        return
    if variant == "atom_identity_only":
        graph["node_residue_type"].zero_()
        return
    if variant == "residue_identity_only":
        for field in ("node_element", "node_atom_name", "node_backbone"):
            graph[field].zero_()
        return
    if variant != "atom_chemistry_shuffled":
        return
    role = graph["node_role"]
    for role_id in (0, 1):
        indices = torch.nonzero(role == role_id, as_tuple=False).flatten()
        permutation = chemistry_permutation(sample_id, role_id, len(indices))
        source = indices[torch.from_numpy(permutation)]
        for field in chemistry_fields:
            graph[field][indices] = graph[field][source].clone()


def add_pair_gate_chemistry(
    graph: dict[str, torch.Tensor], sample_id: str, mode: str
) -> None:
    if mode not in PAIR_GATE_MODES:
        raise ValueError(f"Unsupported pair gate mode: {mode}")
    for field in ("node_element", "node_atom_name", "node_backbone"):
        graph[f"gate_{field}"] = graph[field].clone()
    if mode in {"none", "unknown"}:
        for field in GATE_NODE_FIELDS:
            graph[field].zero_()
        return
    if mode == "real":
        return
    role = graph["node_role"]
    for role_id in (0, 1):
        indices = torch.nonzero(role == role_id, as_tuple=False).flatten()
        permutation = chemistry_permutation(sample_id, role_id, len(indices))
        source = indices[torch.from_numpy(permutation)]
        for field in GATE_NODE_FIELDS:
            graph[field][indices] = graph[field][source].clone()


class InterfaceAtomDataset(Dataset):
    def __init__(
        self,
        *,
        split_path: Path,
        cache_dir: Path,
        confidence_path: Path,
        variant: str,
        pair_gate_mode: str = "none",
        label_permutation_seed: int | None = None,
    ):
        if variant not in ATOM_VARIANTS:
            raise ValueError(f"Unsupported interface atom variant: {variant}")
        if pair_gate_mode not in PAIR_GATE_MODES:
            raise ValueError(f"Unsupported pair gate mode: {pair_gate_mode}")
        self.variant = variant
        self.pair_gate_mode = pair_gate_mode
        self.rows = read_tsv(Path(split_path))
        self.labels = [1.0 if row["label"] == "positive" else 0.0 for row in self.rows]
        if label_permutation_seed is not None:
            permutation = np.random.default_rng(label_permutation_seed).permutation(
                len(self.labels)
            )
            self.labels = [self.labels[index] for index in permutation]
        self.store = InterfaceAtomStore(cache_dir)
        missing = [
            row["sample_id"]
            for row in self.rows
            if row["sample_id"] not in self.store.index
        ]
        if missing:
            raise ValueError(f"Split samples missing from atom cache: {len(missing)}")
        confidence = {row["sample_ids"]: row for row in read_tsv(confidence_path)}
        self.sampling_weights = []
        for row in self.rows:
            if row["sample_id"] not in confidence:
                raise ValueError(f"Missing sampling weight: {row['sample_id']}")
            self.sampling_weights.append(
                float(confidence[row["sample_id"]]["sampling_weight"])
            )
        self.sizes = [self.store.sizes(row["sample_id"]) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        graph = self.store.get(row["sample_id"])
        add_pair_gate_chemistry(graph, row["sample_id"], self.pair_gate_mode)
        apply_chemistry_control(graph, row["sample_id"], self.variant)
        return {
            "sample_id": row["sample_id"],
            "target_name": row["target_name"],
            "label": self.labels[index],
            "sampling_weight": self.sampling_weights[index],
            **graph,
        }


class GraphBudgetBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        sizes: list[tuple[int, int, int]],
        *,
        max_batch_size: int,
        max_message_edges: int,
        shuffle: bool,
        seed: int,
    ):
        if max_batch_size < 1 or max_message_edges < 1:
            raise ValueError("Batch limits must be positive")
        self.sizes = sizes
        self.max_batch_size = max_batch_size
        self.max_message_edges = max_message_edges
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        indices = list(range(len(self.sizes)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        batches: list[list[int]] = []
        current: list[int] = []
        edge_total = 0
        for index in indices:
            edge_count = self.sizes[index][1]
            if edge_count > self.max_message_edges:
                raise ValueError(
                    f"Graph {index} has {edge_count} message edges, exceeding "
                    f"budget {self.max_message_edges}"
                )
            if current and (
                len(current) >= self.max_batch_size
                or edge_total + edge_count > self.max_message_edges
            ):
                batches.append(current)
                current = []
                edge_total = 0
            current.append(index)
            edge_total += edge_count
        if current:
            batches.append(current)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())


def collate_interface_atoms(batch: list[dict[str, object]]) -> dict[str, object]:
    node_counts = [len(item["node_position"]) for item in batch]
    message_counts = [len(item["message_source"]) for item in batch]
    core_counts = [len(item["core_target_node"]) for item in batch]
    node_offsets = np.cumsum([0, *node_counts[:-1]])
    output: dict[str, object] = {
        "sample_id": [item["sample_id"] for item in batch],
        "target_name": [item["target_name"] for item in batch],
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "sampling_weight": torch.tensor(
            [item["sampling_weight"] for item in batch], dtype=torch.float32
        ),
        "graph_count": len(batch),
        "node_graph_index": torch.repeat_interleave(
            torch.arange(len(batch)), torch.tensor(node_counts)
        ),
        "message_graph_index": torch.repeat_interleave(
            torch.arange(len(batch)), torch.tensor(message_counts)
        ),
        "core_graph_index": torch.repeat_interleave(
            torch.arange(len(batch)), torch.tensor(core_counts)
        ),
    }
    for field in NODE_FIELDS:
        output[field] = torch.cat([item[field] for item in batch])
    for field in GATE_NODE_FIELDS:
        output[field] = torch.cat([item[field] for item in batch])
    for field in MESSAGE_FIELDS:
        if field == "message_relation":
            output[field] = torch.cat([item[field] for item in batch])
        else:
            output[field] = torch.cat(
                [
                    item[field] + int(node_offsets[index])
                    for index, item in enumerate(batch)
                ]
            )
    for field in CORE_FIELDS:
        output[field] = torch.cat(
            [
                item[field] + int(node_offsets[index])
                for index, item in enumerate(batch)
            ]
        )
    return output
