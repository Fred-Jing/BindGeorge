#!/usr/bin/env python3
"""Run one serial fold pack, with exact-coverage completion receipts."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_representation_benchmark_completion_v1 import (
    CACHES, RUN, SPLITS, read_tsv, sha256, unique, verify,
)

PYTHON = Path(sys.executable)
E3NN_PYTHON = Path(os.environ.get("BINDER_GRAPH_PYTHON", sys.executable))


def command(row, smoke=False):
    output = RUN / "models" / row["run_name"]
    fold, seed, adapter = int(row["fold"]), int(row["seed"]), row["adapter"]
    if adapter == "esmc":
        if smoke:
            raise ValueError("ESMC uses the existing tested OOF training entry point")
        return [str(PYTHON), "scripts/train_multimodal_v2.py", "--variant", "esmc",
                "--esmc-cache", str(CACHES["esmc"]), "--confidence-table", str(RUN / "auxiliary.tsv.gz"),
                "--train-file", str(SPLITS / f"fold{fold}/train.tsv.gz"),
                "--validation-file", str(SPLITS / f"fold{fold}/validation.tsv.gz"),
                "--development-holdout-file", str(SPLITS / f"fold{fold}/outer_holdout.tsv.gz"),
                "--development-holdout-name", "outer_holdout", "--output-dir", str(output),
                "--max-batch-size", "16", "--max-padded-pair-cells", "300000", "--num-workers", "2",
                "--epochs", "50", "--patience", "8", "--learning-rate", "3e-4", "--weight-decay", "1e-3",
                "--gradient-clip", "1.0", "--seed", str(seed)]
    cmd = [str(E3NN_PYTHON if adapter == "atom_gnn" else PYTHON),
           "scripts/train_representation_benchmark_completion_v1.py",
           "--adapter", adapter, "--fold", str(fold), "--seed", str(seed),
           "--device", row["device"], "--num-workers", "2", "--output-dir", str(output)]
    return cmd + (["--smoke"] if smoke else [])


def validate_predictions(row):
    output = RUN / "models" / row["run_name"]
    config = json.loads((output / "config.json").read_text())
    if int(config["seed"]) != int(row["seed"]) or config.get("sealed_test_accessed") is not False:
        raise ValueError(f"Unsafe run config: {output}")
    if config.get("project_validation_accessed") is not False:
        raise ValueError(f"Unexpected project validation access: {output}")
    if row["adapter"] == "esmc":
        if config.get("variant") != "esmc" or config.get("evaluate_test") is not False:
            raise ValueError(f"Wrong ESMC variant or test access: {output}")
        for key, split in (("train_file", "train"), ("validation_file", "validation"),
                           ("development_holdout_file", "outer_holdout")):
            if config.get(key) != str(SPLITS / f"fold{row['fold']}" / f"{split}.tsv.gz"):
                raise ValueError(f"Wrong ESMC split: {output}/{key}")
    elif config.get("adapter") != row["adapter"] or config.get("outer_fold") != int(row["fold"]):
        raise ValueError(f"Wrong adapter/fold: {output}")
    if any(path.name.startswith("test_predictions") for path in output.iterdir()):
        raise ValueError(f"Unexpected test predictions: {output}")
    hashes = {}
    for split in ("validation", "outer_holdout"):
        path = output / f"{split}_predictions.tsv.gz"
        expected = unique(read_tsv(SPLITS / f"fold{row['fold']}" / f"{split}.tsv.gz"))
        predictions = unique(read_tsv(path))
        if set(predictions) != set(expected):
            raise ValueError(f"Wrong prediction coverage: {path}")
        for sid, record in predictions.items():
            if int(float(record["label"])) != int(expected[sid]["label"] == "positive"):
                raise ValueError(f"Wrong label: {path}/{sid}")
            if record["target_sequence_id"] != expected[sid]["target_sequence_id"]:
                raise ValueError(f"Wrong target: {path}/{sid}")
            p = float(record["probability"])
            if not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError(f"Invalid probability: {path}/{sid}")
        hashes[path.name] = sha256(path)
    for filename in ("config.json", "metrics.json", "best_model.joblib" if row["adapter"] == "rosetta39" else "best_model.pt"):
        hashes[filename] = sha256(output / filename)
    return hashes


def valid_receipt(row):
    output = RUN / "models" / row["run_name"]
    receipt = json.loads((output / "complete.json").read_text())
    if receipt["task"] != row or receipt["protocol_sha256"] != sha256(RUN / "protocol.json"):
        raise ValueError(f"Receipt identity mismatch: {output}")
    if receipt["output_sha256"] != validate_predictions(row):
        raise ValueError(f"Receipt artifact mismatch: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    verify()
    tasks = [row for row in read_tsv(RUN / "manifest.tsv") if row["device"] == args.device
             and (args.fold is None or int(row["fold"]) == args.fold)]
    if args.device == "cuda" and args.fold is None:
        raise ValueError("A GPU pack must name its fold")
    if args.smoke_only:
        seen = set()
        for row in tasks:
            if row["adapter"] == "esmc" or row["adapter"] in seen:
                continue
            subprocess.run(command(row, smoke=True), cwd=ROOT, check=True)
            seen.add(row["adapter"])
        return
    failures = []
    for row in tasks:
        output = RUN / "models" / row["run_name"]
        try:
            if (output / "complete.json").is_file():
                valid_receipt(row)
                print(f"skip_complete={row['run_name']}", flush=True)
                continue
            if output.exists():
                raise FileExistsError(f"Partial run needs review: {output}")
            cmd = command(row)
            print(f"start={row['run_name']} time={time.time()}", flush=True)
            with (RUN / "logs" / f"{row['run_name']}.log").open("x") as handle:
                result = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Trainer exited {result.returncode}")
            hashes = validate_predictions(row)
            receipt = {"task": row, "protocol_sha256": sha256(RUN / "protocol.json"),
                       "output_sha256": hashes, "command": cmd, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                       "completed_unix_time": time.time()}
            (output / "complete.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            print(f"complete={row['run_name']} time={time.time()}", flush=True)
        except Exception as error:
            failures.append({"run_name": row["run_name"], "error": str(error)})
            print(f"failed={row['run_name']} error={error}", flush=True)
    if failures:
        path = RUN / "logs" / f"failures_{args.device}_fold{args.fold}_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
        path.write_text(json.dumps(failures, indent=2) + "\n")
        raise SystemExit(1)
    print("worker_status=complete", flush=True)


if __name__ == "__main__":
    main()
