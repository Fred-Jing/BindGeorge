"""Synthetic cache, OOF aggregation and aggregate-table regression checks."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.prepare_representation_benchmark_completion_v1 import SEEDS
from scripts.show_results import read_results
from scripts.summarize_representation_benchmark_completion_v1 import collapse
from training.data import SequenceEmbeddingStore, collate_pairs, sequence_key


class ReleaseDataTests(unittest.TestCase):
    def test_synthetic_cache_and_padding(self):
        with tempfile.TemporaryDirectory(prefix="binder-synthetic-") as directory:
            cache = Path(directory)
            sequences = ("AC", "DEF")
            values = np.arange(20, dtype=np.float16).reshape(5, 4)
            np.save(cache / "embeddings.f16.npy", values)
            (cache / "metadata.json").write_text(json.dumps({"complete": True, "hidden_dim": 4}))
            with (cache / "sequence_manifest.tsv").open("w", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(("sequence_key", "offset", "length"))
                writer.writerow((sequence_key(sequences[0]), 0, 2))
                writer.writerow((sequence_key(sequences[1]), 2, 3))
            store = SequenceEmbeddingStore(cache)
            target = store.get_target("ACDEF", json.dumps(sequences))
            torch.testing.assert_close(target, torch.from_numpy(values))
            with self.assertRaises(KeyError):
                store.get("W")
            batch = collate_pairs([
                {"sample_id": "synthetic-a", "target_name": "synthetic", "target": target,
                 "binder": store.get("AC"), "label": 1.0},
                {"sample_id": "synthetic-b", "target_name": "synthetic", "target": store.get("DEF"),
                 "binder": store.get("DEF"), "label": 0.0},
            ])
            self.assertEqual(batch["target"].shape, (2, 5, 4))
            self.assertEqual(batch["binder"].shape, (2, 3, 4))
            self.assertEqual(batch["target_mask"].sum(dim=1).tolist(), [5, 3])
            self.assertEqual(batch["binder_mask"].sum(dim=1).tolist(), [2, 3])
            self.assertEqual(batch["label"].tolist(), [1.0, 0.0])

    def fixture(self):
        original = {sid: {"label": label, "target_name": "synthetic",
                          "target_sequence_id": "synthetic-target", "target_cluster_id": "synthetic-cluster"}
                    for sid, label in (("a", "positive"), ("b", "negative"))}
        folds = {"a": 0, "b": 1}
        members = [(fold, seed, [{"sample_id": sid, "label": int(sid == "a"),
                                 "target_sequence_id": "synthetic-target", "probability": 0.2 + index * 0.4}])
                   for fold, sid in enumerate(("a", "b")) for index, seed in enumerate(SEEDS)]
        return original, folds, members

    def test_oof_mean(self):
        original, folds, members = self.fixture()
        rows = collapse(list(reversed(members)), original, folds)
        self.assertEqual([row["sample_id"] for row in rows], ["a", "b"])
        for row in rows:
            self.assertAlmostEqual(row["probability"], 0.4)

    def test_missing_and_duplicate_seed_rejected(self):
        original, folds, members = self.fixture()
        for invalid in (members[:-1], members + [members[0]]):
            with self.assertRaises(ValueError):
                collapse(invalid, original, folds)

    def test_wrong_label_rejected(self):
        original, folds, members = self.fixture()
        members[0][2][0]["label"] = 0
        with self.assertRaises(ValueError):
            collapse(members, original, folds)

    def test_aggregate_table(self):
        rows = {row["model"]: row for row in read_results()}
        self.assertEqual(len(rows), 18)
        self.assertAlmostEqual(float(rows["AF3 mature pair"]["auprc"]), 0.613507)
        self.assertAlmostEqual(float(rows["confidence3"]["auprc"]), 0.5341855839539142)


if __name__ == "__main__":
    unittest.main()
