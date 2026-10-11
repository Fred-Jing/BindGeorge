from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def sequence_key(sequence: str) -> str:
    return f"SEQ_{hashlib.sha256(sequence.encode()).hexdigest()[:24]}"


def read_split(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class SequenceEmbeddingStore:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.embedding_path = self.cache_dir / "embeddings.f16.npy"
        self.index_path = self.cache_dir / "sequence_manifest.tsv"
        self.metadata_path = self.cache_dir / "metadata.json"
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("complete"):
            raise ValueError(f"Embedding cache is incomplete: {self.cache_dir}")
        self.hidden_dim = int(metadata["hidden_dim"])
        self.index: dict[str, tuple[int, int]] = {}
        with self.index_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                self.index[row["sequence_key"]] = (
                    int(row["offset"]),
                    int(row["length"]),
                )
        self._embeddings: np.ndarray | None = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_embeddings"] = None
        return state

    @property
    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            self._embeddings = np.load(self.embedding_path, mmap_mode="r")
            if self._embeddings.ndim != 2 or self._embeddings.shape[1] != self.hidden_dim:
                raise ValueError(f"Unexpected embedding shape: {self._embeddings.shape}")
        return self._embeddings

    def get(self, sequence: str) -> torch.Tensor:
        key = sequence_key(sequence)
        if key not in self.index:
            raise KeyError(f"Sequence is absent from embedding cache: {key}")
        offset, length = self.index[key]
        array = np.array(self.embeddings[offset : offset + length], copy=True)
        return torch.from_numpy(array)

    def get_target(self, sequence: str, chain_sequences_json: str) -> torch.Tensor:
        if sequence_key(sequence) in self.index:
            return self.get(sequence)
        chain_sequences = json.loads(chain_sequences_json)
        if not isinstance(chain_sequences, list) or not chain_sequences:
            raise ValueError("Target chain sequence manifest must be a nonempty list")
        if not all(isinstance(chain, str) and chain for chain in chain_sequences):
            raise ValueError("Target chain sequence manifest contains an invalid chain")
        if "".join(chain_sequences) != sequence:
            raise ValueError("Target chain sequences do not reconstruct target_sequence")
        return torch.cat([self.get(chain) for chain in chain_sequences], dim=0)


class PairDataset(Dataset):
    def __init__(self, split_path: Path, cache_dir: Path):
        self.rows = read_split(Path(split_path))
        self.store = SequenceEmbeddingStore(Path(cache_dir))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        return {
            "sample_id": row["sample_id"],
            "target_name": row["target_name"],
            "target": self.store.get_target(
                row["target_sequence"], row["target_chain_sequences"]
            ),
            "binder": self.store.get(row["binder_sequence"]),
            "label": 1.0 if row["label"] == "positive" else 0.0,
        }


def collate_pairs(batch: list[dict[str, object]]) -> dict[str, object]:
    targets = [item["target"] for item in batch]
    binders = [item["binder"] for item in batch]
    target_lengths = torch.tensor([tensor.shape[0] for tensor in targets])
    binder_lengths = torch.tensor([tensor.shape[0] for tensor in binders])
    target_padded = pad_sequence(targets, batch_first=True)
    binder_padded = pad_sequence(binders, batch_first=True)
    target_mask = (
        torch.arange(target_padded.shape[1]).unsqueeze(0) < target_lengths.unsqueeze(1)
    )
    binder_mask = (
        torch.arange(binder_padded.shape[1]).unsqueeze(0) < binder_lengths.unsqueeze(1)
    )
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "target_name": [item["target_name"] for item in batch],
        "target": target_padded,
        "target_mask": target_mask,
        "binder": binder_padded,
        "binder_mask": binder_mask,
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
    }
