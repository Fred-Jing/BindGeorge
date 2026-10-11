from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from .data import SequenceEmbeddingStore
from .interface_geometry_descriptor import FeatureDefinition, feature_indices
from .multimodal_model import MODEL_VARIANTS


CONFIDENCE_FIELDS = ("interface_pae_min", "iptm", "mean_atom_plddt")


def read_tsv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def make_within_target_geometry_shuffle(
    rows: list[dict[str, str]],
) -> tuple[dict[str, str], dict[str, int]]:
    sample_ids_by_target: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        sample_ids_by_target[row["target_name"]].append(row["sample_id"])
    source_by_sample = {}
    singleton_targets = 0
    singleton_samples = 0
    for sample_ids in sample_ids_by_target.values():
        sample_ids.sort()
        if len(sample_ids) == 1:
            singleton_targets += 1
            singleton_samples += 1
        for sample_index, sample_id in enumerate(sample_ids):
            source_by_sample[sample_id] = sample_ids[
                (sample_index + 1) % len(sample_ids)
            ]
    statistics = {
        "target_count": len(sample_ids_by_target),
        "sample_count": len(rows),
        "singleton_target_count": singleton_targets,
        "singleton_sample_count": singleton_samples,
        "nonself_sample_count": len(rows) - singleton_samples,
    }
    return source_by_sample, statistics


def make_within_target_sequence_descriptor_shuffle(
    rows: list[dict[str, str]],
) -> tuple[dict[str, str], dict[str, int]]:
    sample_ids_by_target: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        sample_ids_by_target[row["target_sequence_id"]].append(row["sample_id"])
    source_by_sample = {}
    singleton_targets = 0
    singleton_samples = 0
    for sample_ids in sample_ids_by_target.values():
        sample_ids.sort()
        if len(sample_ids) == 1:
            singleton_targets += 1
            singleton_samples += 1
        for sample_index, sample_id in enumerate(sample_ids):
            source_by_sample[sample_id] = sample_ids[
                (sample_index + 1) % len(sample_ids)
            ]
    statistics = {
        "target_sequence_count": len(sample_ids_by_target),
        "sample_count": len(rows),
        "singleton_target_sequence_count": singleton_targets,
        "singleton_sample_count": singleton_samples,
        "nonself_sample_count": len(rows) - singleton_samples,
    }
    return source_by_sample, statistics


def geometry_axis_shifts(
    sample_id: str, target_length: int, binder_length: int
) -> tuple[int, int]:
    digest = hashlib.sha256(f"geometry-axis-v1:{sample_id}".encode()).digest()
    target_shift = (
        1 + int.from_bytes(digest[:8], "little") % (target_length - 1)
        if target_length > 1
        else 0
    )
    binder_shift = (
        1 + int.from_bytes(digest[8:16], "little") % (binder_length - 1)
        if binder_length > 1
        else 0
    )
    return target_shift, binder_shift


def deterministic_random_pair_mask(
    sample_id: str, shape: tuple[int, int], selected_count: int
) -> torch.Tensor:
    total = shape[0] * shape[1]
    if not 1 <= selected_count <= total:
        raise ValueError(
            f"Invalid random pair selection count for {sample_id}: {selected_count}/{total}"
        )
    digest = hashlib.sha256(f"pair-random-count-v1:{sample_id}".encode()).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    selected = generator.choice(total, size=selected_count, replace=False)
    mask = torch.zeros(total, dtype=torch.bool)
    mask[torch.from_numpy(selected.astype(np.int64, copy=False))] = True
    return mask.reshape(shape)


def deterministic_degree_preserving_pair_mask(
    sample_id: str,
    reference_mask: torch.Tensor,
    mask_seed: int,
    *,
    swap_attempts_per_edge: int = 20,
) -> torch.Tensor:
    """Rewire a bipartite mask while preserving every row and column degree."""
    if reference_mask.ndim != 2 or reference_mask.dtype != torch.bool:
        raise ValueError("Reference pair mask must be a two-dimensional boolean tensor")
    if swap_attempts_per_edge < 1:
        raise ValueError("Swap attempts per edge must be positive")
    edge_array = torch.nonzero(reference_mask, as_tuple=False).cpu().numpy()
    edge_count = edge_array.shape[0]
    if edge_count < 2:
        return reference_mask.clone()

    binder_length = reference_mask.shape[1]
    flat_edges = {
        int(target_index) * binder_length + int(binder_index)
        for target_index, binder_index in edge_array
    }
    digest = hashlib.sha256(
        f"pair-degree-rewire-v1:{mask_seed}:{sample_id}".encode()
    ).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    attempts = max(100, swap_attempts_per_edge * edge_count)
    for _ in range(attempts):
        first, second = generator.integers(0, edge_count, size=2)
        if first == second:
            continue
        first_target, first_binder = edge_array[first]
        second_target, second_binder = edge_array[second]
        if first_target == second_target or first_binder == second_binder:
            continue
        first_rewired = int(first_target) * binder_length + int(second_binder)
        second_rewired = int(second_target) * binder_length + int(first_binder)
        if first_rewired in flat_edges or second_rewired in flat_edges:
            continue
        first_flat = int(first_target) * binder_length + int(first_binder)
        second_flat = int(second_target) * binder_length + int(second_binder)
        flat_edges.remove(first_flat)
        flat_edges.remove(second_flat)
        flat_edges.add(first_rewired)
        flat_edges.add(second_rewired)
        edge_array[first, 1] = second_binder
        edge_array[second, 1] = first_binder

    output = torch.zeros(reference_mask.numel(), dtype=torch.bool)
    output[torch.tensor(sorted(flat_edges), dtype=torch.long)] = True
    return output.reshape(reference_mask.shape)


class AF3CrossStore:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        metadata = json.loads(
            (self.cache_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if metadata.get("status") != "complete":
            raise ValueError(f"Representation cache is incomplete: {self.cache_dir}")
        self.metadata = metadata
        self.index: dict[str, dict[str, int | str]] = {}
        with (self.cache_dir / "index.tsv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                sample_id = row["sample_id"]
                if sample_id in self.index:
                    raise ValueError(f"Duplicate representation cache sample: {sample_id}")
                self.index[sample_id] = {
                    "cache_index": int(row["cache_index"]),
                    "complex_feature_id": row["complex_feature_id"],
                    "target_length": int(row["target_length"]),
                    "binder_length": int(row["binder_length"]),
                    "target_single_offset": int(row["target_single_offset"]),
                    "binder_single_offset": int(row["binder_single_offset"]),
                    "pair_offset": int(row["pair_offset"]),
                    "pair_cells": int(row["pair_cells"]),
                }
        self._arrays: dict[str, np.ndarray] = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_arrays"] = {}
        return state

    def _array(self, key: str) -> np.ndarray:
        if key not in self._arrays:
            path = Path(self.metadata["arrays"][key]["path"])
            self._arrays[key] = np.load(path, mmap_mode="r")
        return self._arrays[key]

    def lengths(self, sample_id: str) -> tuple[int, int]:
        row = self.index[sample_id]
        return int(row["target_length"]), int(row["binder_length"])

    def get(
        self,
        sample_id: str,
        *,
        load_single: bool,
        load_pair: bool,
    ) -> dict[str, torch.Tensor]:
        row = self.index[sample_id]
        target_length = int(row["target_length"])
        binder_length = int(row["binder_length"])
        output: dict[str, torch.Tensor] = {}
        if load_single:
            target_offset = int(row["target_single_offset"])
            binder_offset = int(row["binder_single_offset"])
            output["target_single"] = torch.from_numpy(
                np.array(
                    self._array("target_single")[
                        target_offset : target_offset + target_length
                    ],
                    copy=True,
                )
            )
            output["binder_single"] = torch.from_numpy(
                np.array(
                    self._array("binder_single")[
                        binder_offset : binder_offset + binder_length
                    ],
                    copy=True,
                )
            )
        if load_pair:
            pair_offset = int(row["pair_offset"])
            pair_cells = int(row["pair_cells"])
            pair_shape = (target_length, binder_length, 128)
            output["target_binder_pair"] = torch.from_numpy(
                np.array(
                    self._array("target_binder_pair")[
                        pair_offset : pair_offset + pair_cells
                    ],
                    copy=True,
                ).reshape(pair_shape)
            )
            output["binder_target_pair"] = torch.from_numpy(
                np.array(
                    self._array("binder_target_pair")[
                        pair_offset : pair_offset + pair_cells
                    ],
                    copy=True,
                ).reshape(pair_shape)
            )
        return output


class AF3GeometryStore:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        metadata = json.loads(
            (self.cache_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if metadata.get("status") != "complete":
            raise ValueError(f"AF3 geometry cache is incomplete: {self.cache_dir}")
        self.metadata = metadata
        self.index: dict[str, dict[str, int]] = {}
        with (self.cache_dir / "index.tsv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                sample_id = row["sample_id"]
                if sample_id in self.index:
                    raise ValueError(f"Duplicate geometry cache sample: {sample_id}")
                self.index[sample_id] = {
                    "target_length": int(row["target_length"]),
                    "binder_length": int(row["binder_length"]),
                    "distance_offset": int(row["distance_offset"]),
                    "pair_cells": int(row["pair_cells"]),
                }
        self._distances: np.ndarray | None = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_distances"] = None
        return state

    def get(self, sample_id: str) -> torch.Tensor:
        row = self.index[sample_id]
        if self._distances is None:
            self._distances = np.load(
                Path(self.metadata["distance_array"]["path"]), mmap_mode="r"
            )
        offset = int(row["distance_offset"])
        cells = int(row["pair_cells"])
        shape = (int(row["target_length"]), int(row["binder_length"]))
        return torch.from_numpy(
            np.array(self._distances[offset : offset + cells], copy=True).reshape(shape)
        )


class AF3GeometryV2Store:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        metadata = json.loads(
            (self.cache_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if metadata.get("status") != "complete":
            raise ValueError(f"AF3 geometry v2 cache is incomplete: {self.cache_dir}")
        if metadata.get("schema_version") != "af3_residue_geometry_canonical_v2":
            raise ValueError(f"Unexpected AF3 geometry v2 schema: {self.cache_dir}")
        summary_specification = metadata["arrays"]["canonical_summary"]
        if summary_specification["shape"][1] != 52:
            raise ValueError(
                f"Unexpected canonical geometry width: {summary_specification['shape']}"
            )
        self.metadata = metadata
        self.index: dict[str, int] = {}
        with (self.cache_dir / "index.tsv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                sample_id = row["sample_id"]
                if sample_id in self.index:
                    raise ValueError(
                        f"Duplicate geometry v2 cache sample: {sample_id}"
                    )
                self.index[sample_id] = int(row["summary_index"])
        self._summaries: np.ndarray | None = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_summaries"] = None
        return state

    def get(self, sample_id: str) -> torch.Tensor:
        if self._summaries is None:
            self._summaries = np.load(
                Path(self.metadata["arrays"]["canonical_summary"]["path"]),
                mmap_mode="r",
            )
        return torch.from_numpy(
            np.array(self._summaries[self.index[sample_id]], copy=True)
        )


class AF3GeometryCellStore:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        metadata = json.loads(
            (self.cache_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if metadata.get("status") != "complete":
            raise ValueError(f"AF3 geometry cell cache is incomplete: {self.cache_dir}")
        if metadata.get("schema_version") != "af3_canonical_geometry_pair_cells_v1":
            raise ValueError(f"Unexpected AF3 geometry cell schema: {self.cache_dir}")
        if metadata.get("feature_dimension") != 26:
            raise ValueError(
                f"Unexpected geometry cell width: {metadata.get('feature_dimension')}"
            )
        self.metadata = metadata
        self.index: dict[str, dict[str, int]] = {}
        with (self.cache_dir / "index.tsv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                sample_id = row["sample_id"]
                if sample_id in self.index:
                    raise ValueError(
                        f"Duplicate geometry cell cache sample: {sample_id}"
                    )
                self.index[sample_id] = {
                    "target_length": int(row["target_length"]),
                    "binder_length": int(row["binder_length"]),
                    "geometry_offset": int(row["geometry_offset"]),
                    "pair_cells": int(row["pair_cells"]),
                }
        self._geometry: np.ndarray | None = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_geometry"] = None
        return state

    def get(self, sample_id: str) -> torch.Tensor:
        if self._geometry is None:
            self._geometry = np.load(
                Path(self.metadata["geometry_array"]["path"]), mmap_mode="r"
            )
        row = self.index[sample_id]
        offset = int(row["geometry_offset"])
        pair_cells = int(row["pair_cells"])
        shape = (int(row["target_length"]), int(row["binder_length"]), 26)
        return torch.from_numpy(
            np.array(
                self._geometry[offset : offset + pair_cells], copy=True
            ).reshape(shape)
        )


class InterfaceDescriptorStore:
    def __init__(self, cache_dir: Path, variant: str):
        self.cache_dir = Path(cache_dir)
        metadata = json.loads(
            (self.cache_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if metadata.get("status") != "complete":
            raise ValueError(f"Interface descriptor cache is incomplete: {self.cache_dir}")
        if metadata.get("schema_version") not in {
            "interface_geometry_descriptor_v1",
            "dmasif_interface_descriptor_v1",
        }:
            raise ValueError(f"Unexpected interface descriptor schema: {self.cache_dir}")
        if metadata.get("sealed_test_accessed") is not False:
            raise ValueError("Interface descriptor cache lacks sealed-test isolation")
        definitions = []
        with (self.cache_dir / "feature_names.tsv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for expected, row in enumerate(csv.DictReader(handle, delimiter="\t")):
                if int(row["feature_index"]) != expected:
                    raise ValueError("Non-contiguous interface descriptor feature indices")
                definitions.append(
                    FeatureDefinition(row["name"], row["group"], row["subgroup"])
                )
        self.selected_columns = feature_indices(definitions, variant)
        self.selected_feature_names = [
            definitions[index].name for index in self.selected_columns
        ]
        self.index: dict[str, int] = {}
        with (self.cache_dir / "index.tsv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                sample_id = row["sample_id"]
                if sample_id in self.index:
                    raise ValueError(f"Duplicate interface descriptor sample: {sample_id}")
                self.index[sample_id] = int(row["cache_index"])
        self._features: np.ndarray | None = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_features"] = None
        return state

    @property
    def width(self) -> int:
        return len(self.selected_columns)

    def get(self, sample_id: str) -> torch.Tensor:
        if self._features is None:
            self._features = np.load(
                self.cache_dir / "features.float32.npy", mmap_mode="r"
            )
        return torch.from_numpy(
            np.array(
                self._features[self.index[sample_id], self.selected_columns],
                dtype=np.float32,
                copy=True,
            )
        )


class MultimodalPairDataset(Dataset):
    def __init__(
        self,
        *,
        split_path: Path,
        esmc_cache_dir: Path,
        af3_cache_dir: Path,
        confidence_path: Path,
        variant: str,
        geometry_cache_dir: Path | None = None,
        geometry_v2_cache_dir: Path | None = None,
        geometry_cell_cache_dir: Path | None = None,
        interface_descriptor_cache_dir: Path | None = None,
        interface_descriptor_variant: str = "g1_aa_without_size",
        random_mask_reference_cutoff: float = 10.0,
        pair_mask_seed: int = 0,
    ):
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"Unsupported model variant: {variant}")
        self.rows = read_tsv(Path(split_path))
        self.flags = MODEL_VARIANTS[variant]
        self.esmc_store = (
            SequenceEmbeddingStore(esmc_cache_dir) if self.flags["use_esmc"] else None
        )
        uses_representation = (
            self.flags["use_af3_single"] or self.flags["use_af3_pair"]
        )
        self.af3_store = AF3CrossStore(af3_cache_dir) if uses_representation else None
        self.pair_selection_mode = self.flags.get("pair_selection_mode")
        self.use_pair_selection_mask = self.flags.get(
            "use_pair_selection_mask", False
        )
        if self.use_pair_selection_mask and self.pair_selection_mode not in {
            "random_count",
            "degree_rewired",
        }:
            raise ValueError(f"Unsupported pair selection mode: {self.pair_selection_mode}")
        if self.use_pair_selection_mask and random_mask_reference_cutoff <= 0:
            raise ValueError("Pair mask reference cutoff must be positive")
        self.random_mask_reference_cutoff = random_mask_reference_cutoff
        self.pair_mask_seed = pair_mask_seed
        self._pair_selection_masks: dict[str, torch.Tensor] = {}
        requires_distance = (
            self.flags.get("use_geometry", False)
            or self.flags.get("use_pair_distance_mask", False)
            or self.use_pair_selection_mask
        )
        self.geometry_store = (
            AF3GeometryStore(geometry_cache_dir)
            if requires_distance and geometry_cache_dir is not None
            else None
        )
        if requires_distance and self.geometry_store is None:
            raise ValueError(f"Variant {variant} requires an AF3 distance cache")
        self.geometry_v2_store = (
            AF3GeometryV2Store(geometry_v2_cache_dir)
            if self.flags.get("use_geometry_summary")
            and geometry_v2_cache_dir is not None
            else None
        )
        if self.flags.get("use_geometry_summary") and self.geometry_v2_store is None:
            raise ValueError(f"Variant {variant} requires an AF3 geometry v2 cache")
        self.geometry_cell_store = (
            AF3GeometryCellStore(geometry_cell_cache_dir)
            if self.flags.get("use_geometry_cells")
            and geometry_cell_cache_dir is not None
            else None
        )
        self.interface_descriptor_store = (
            InterfaceDescriptorStore(
                interface_descriptor_cache_dir, interface_descriptor_variant
            )
            if self.flags.get("use_interface_descriptor")
            and interface_descriptor_cache_dir is not None
            else None
        )
        if (
            self.flags.get("use_interface_descriptor")
            and self.interface_descriptor_store is None
        ):
            raise ValueError(f"Variant {variant} requires an interface descriptor cache")
        if self.flags.get("use_geometry_cells") and self.geometry_cell_store is None:
            raise ValueError(f"Variant {variant} requires an AF3 geometry cell cache")
        confidence_rows = read_tsv(confidence_path)
        self.confidence = {}
        for row in confidence_rows:
            sample_id = row["sample_ids"]
            if sample_id in self.confidence:
                raise ValueError(f"Duplicate confidence sample: {sample_id}")
            self.confidence[sample_id] = row
        missing = [row["sample_id"] for row in self.rows if row["sample_id"] not in self.confidence]
        if missing:
            raise ValueError(f"Split samples missing from confidence table: {len(missing)}")
        if self.af3_store is not None:
            missing_af3 = [
                row["sample_id"]
                for row in self.rows
                if row["sample_id"] not in self.af3_store.index
            ]
            if missing_af3:
                raise ValueError(
                    f"Split samples missing from representation cache: {len(missing_af3)}"
                )
        if self.geometry_store is not None:
            missing_geometry = [
                row["sample_id"]
                for row in self.rows
                if row["sample_id"] not in self.geometry_store.index
            ]
            if missing_geometry:
                raise ValueError(
                    f"Split samples missing from geometry cache: {len(missing_geometry)}"
                )
        if self.geometry_v2_store is not None:
            missing_geometry_v2 = [
                row["sample_id"]
                for row in self.rows
                if row["sample_id"] not in self.geometry_v2_store.index
            ]
            if missing_geometry_v2:
                raise ValueError(
                    "Split samples missing from geometry v2 cache: "
                    f"{len(missing_geometry_v2)}"
                )
        if self.geometry_cell_store is not None:
            missing_geometry_cells = [
                row["sample_id"]
                for row in self.rows
                if row["sample_id"] not in self.geometry_cell_store.index
            ]
            if missing_geometry_cells:
                raise ValueError(
                    "Split samples missing from geometry cell cache: "
                    f"{len(missing_geometry_cells)}"
                )
        if self.interface_descriptor_store is not None:
            missing_descriptors = [
                row["sample_id"]
                for row in self.rows
                if row["sample_id"] not in self.interface_descriptor_store.index
            ]
            if missing_descriptors:
                raise ValueError(
                    "Split samples missing from interface descriptor cache: "
                    f"{len(missing_descriptors)}"
                )
        self.lengths = [
            (
                self.af3_store.lengths(row["sample_id"])
                if self.af3_store is not None
                else (
                    sum(len(sequence) for sequence in json.loads(row["target_chain_sequences"])),
                    len(row["binder_sequence"]),
                )
            )
            for row in self.rows
        ]
        self.geometry_source_by_sample: dict[str, str] | None = None
        self.geometry_shuffle_statistics: dict[str, int] | None = None
        self.interface_descriptor_source_by_sample: dict[str, str] | None = None
        self.interface_descriptor_shuffle_statistics: dict[str, int] | None = None
        if self.flags.get("shuffle_geometry_within_target"):
            (
                self.geometry_source_by_sample,
                self.geometry_shuffle_statistics,
            ) = make_within_target_geometry_shuffle(self.rows)
        if self.flags.get("shuffle_geometry_axes"):
            self.geometry_shuffle_statistics = {
                "sample_count": len(self.rows),
                "target_axis_shifted_sample_count": sum(
                    target_length > 1 for target_length, _ in self.lengths
                ),
                "binder_axis_shifted_sample_count": sum(
                    binder_length > 1 for _, binder_length in self.lengths
                ),
            }
        if self.flags.get("shuffle_interface_descriptor_within_target"):
            (
                self.interface_descriptor_source_by_sample,
                self.interface_descriptor_shuffle_statistics,
            ) = make_within_target_sequence_descriptor_shuffle(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        sample_id = row["sample_id"]
        target_length, binder_length = self.lengths[index]
        output: dict[str, object] = {
            "sample_id": sample_id,
            "target_name": row["target_name"],
            "target_sequence_id": row["target_sequence_id"],
            "label": 1.0 if row["label"] == "positive" else 0.0,
            "sampling_weight": float(self.confidence[sample_id]["sampling_weight"]),
            "confidence": torch.tensor(
                [float(self.confidence[sample_id][field]) for field in CONFIDENCE_FIELDS],
                dtype=torch.float32,
            ),
            "target_length": target_length,
            "binder_length": binder_length,
        }
        if self.flags["use_esmc"]:
            target_esmc = self.esmc_store.get_target(
                row["target_sequence"], row["target_chain_sequences"]
            )
            binder_esmc = self.esmc_store.get(row["binder_sequence"])
            if target_esmc.shape[0] != target_length or binder_esmc.shape[0] != binder_length:
                raise ValueError(f"ESMC/AF3 residue length mismatch: {sample_id}")
            output["target_esmc"] = target_esmc
            output["binder_esmc"] = binder_esmc
        if self.af3_store is not None:
            output.update(
                self.af3_store.get(
                    sample_id,
                    load_single=self.flags["use_af3_single"],
                    load_pair=self.flags["use_af3_pair"],
                )
            )
        if self.geometry_store is not None:
            distances = self.geometry_store.get(sample_id)
            output["target_binder_distance"] = distances
            if self.use_pair_selection_mask:
                if sample_id not in self._pair_selection_masks:
                    reference_mask = distances <= self.random_mask_reference_cutoff
                    if self.pair_selection_mode == "random_count":
                        selection_mask = deterministic_random_pair_mask(
                            sample_id,
                            (target_length, binder_length),
                            int(reference_mask.sum()),
                        )
                    else:
                        selection_mask = deterministic_degree_preserving_pair_mask(
                            sample_id,
                            reference_mask,
                            self.pair_mask_seed,
                        )
                    self._pair_selection_masks[sample_id] = selection_mask
                output["target_binder_selection_mask"] = self._pair_selection_masks[
                    sample_id
                ]
        if self.geometry_v2_store is not None:
            geometry_sample_id = (
                self.geometry_source_by_sample[sample_id]
                if self.geometry_source_by_sample is not None
                else sample_id
            )
            output["target_binder_geometry_summary"] = self.geometry_v2_store.get(
                geometry_sample_id
            )
        if self.geometry_cell_store is not None:
            geometry = self.geometry_cell_store.get(sample_id)
            if self.flags.get("shuffle_geometry_axes"):
                shifts = geometry_axis_shifts(
                    sample_id, target_length, binder_length
                )
                geometry = torch.roll(geometry, shifts=shifts, dims=(0, 1))
            output["target_binder_geometry"] = geometry
        if self.interface_descriptor_store is not None:
            descriptor_sample_id = (
                self.interface_descriptor_source_by_sample[sample_id]
                if self.interface_descriptor_source_by_sample is not None
                else sample_id
            )
            output["interface_descriptor"] = self.interface_descriptor_store.get(
                descriptor_sample_id
            )
        return output


class PairCellBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: list[tuple[int, int]],
        *,
        max_batch_size: int,
        max_padded_pair_cells: int,
        shuffle: bool,
        seed: int,
    ):
        if max_batch_size < 1 or max_padded_pair_cells < 1:
            raise ValueError("Batch limits must be positive")
        self.lengths = lengths
        self.max_batch_size = max_batch_size
        self.max_padded_pair_cells = max_padded_pair_cells
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @staticmethod
    def _bucket_key(lengths: tuple[int, int]) -> tuple[int, int]:
        target_length, binder_length = lengths
        return math.ceil(math.log2(target_length)), math.ceil(math.log2(binder_length))

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, lengths in enumerate(self.lengths):
            buckets[self._bucket_key(lengths)].append(index)
        batches = []
        bucket_items = list(buckets.items())
        if self.shuffle:
            rng.shuffle(bucket_items)
        for _, indices in bucket_items:
            if self.shuffle:
                rng.shuffle(indices)
            current: list[int] = []
            max_target = 0
            max_binder = 0
            for index in indices:
                target_length, binder_length = self.lengths[index]
                proposed_target = max(max_target, target_length)
                proposed_binder = max(max_binder, binder_length)
                proposed_size = len(current) + 1
                proposed_cells = proposed_size * proposed_target * proposed_binder
                if current and (
                    proposed_size > self.max_batch_size
                    or proposed_cells > self.max_padded_pair_cells
                ):
                    batches.append(current)
                    current = []
                    max_target = 0
                    max_binder = 0
                current.append(index)
                max_target = max(max_target, target_length)
                max_binder = max(max_binder, binder_length)
            if current:
                batches.append(current)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())


def _pad_pair(
    tensors: list[torch.Tensor], max_target: int, max_binder: int
) -> torch.Tensor:
    output = torch.zeros(
        len(tensors),
        max_target,
        max_binder,
        tensors[0].shape[-1],
        dtype=tensors[0].dtype,
    )
    for index, tensor in enumerate(tensors):
        output[index, : tensor.shape[0], : tensor.shape[1]] = tensor
    return output


def _pad_scalar_pair(
    tensors: list[torch.Tensor], max_target: int, max_binder: int
) -> torch.Tensor:
    output = torch.zeros(
        len(tensors), max_target, max_binder, dtype=tensors[0].dtype
    )
    for index, tensor in enumerate(tensors):
        output[index, : tensor.shape[0], : tensor.shape[1]] = tensor
    return output


def collate_multimodal_pairs(batch: list[dict[str, object]]) -> dict[str, object]:
    target_lengths = torch.tensor([item["target_length"] for item in batch])
    binder_lengths = torch.tensor([item["binder_length"] for item in batch])
    max_target = int(target_lengths.max())
    max_binder = int(binder_lengths.max())
    output: dict[str, object] = {
        "sample_id": [item["sample_id"] for item in batch],
        "target_name": [item["target_name"] for item in batch],
        "target_sequence_id": [item["target_sequence_id"] for item in batch],
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "sampling_weight": torch.tensor(
            [item["sampling_weight"] for item in batch], dtype=torch.float32
        ),
        "confidence": torch.stack([item["confidence"] for item in batch]),
        "target_mask": torch.arange(max_target).unsqueeze(0) < target_lengths.unsqueeze(1),
        "binder_mask": torch.arange(max_binder).unsqueeze(0) < binder_lengths.unsqueeze(1),
    }
    for key in ("target_esmc", "binder_esmc", "target_single", "binder_single"):
        if key in batch[0]:
            output[key] = pad_sequence(
                [item[key] for item in batch], batch_first=True
            )
    for key in (
        "target_binder_pair",
        "binder_target_pair",
        "target_binder_geometry",
    ):
        if key in batch[0]:
            output[key] = _pad_pair(
                [item[key] for item in batch], max_target, max_binder
            )
    if "target_binder_distance" in batch[0]:
        output["target_binder_distance"] = _pad_scalar_pair(
            [item["target_binder_distance"] for item in batch],
            max_target,
            max_binder,
        )
    if "target_binder_selection_mask" in batch[0]:
        output["target_binder_selection_mask"] = _pad_scalar_pair(
            [item["target_binder_selection_mask"] for item in batch],
            max_target,
            max_binder,
        )
    if "target_binder_geometry_summary" in batch[0]:
        output["target_binder_geometry_summary"] = torch.stack(
            [item["target_binder_geometry_summary"] for item in batch]
        )
    if "interface_descriptor" in batch[0]:
        output["interface_descriptor"] = torch.stack(
            [item["interface_descriptor"] for item in batch]
        )
    return output
