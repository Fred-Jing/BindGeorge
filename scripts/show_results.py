"""Validate and display the supplied aggregate benchmark without real samples."""

import csv
import math
from pathlib import Path


TABLE = Path(__file__).resolve().parents[1] / "results/main_results.tsv"


def read_results(path=TABLE):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 18 or len({row["model"] for row in rows}) != 18:
        raise ValueError("Expected 18 distinct primary benchmark entries")
    for row in rows:
        if (int(row["count"]), int(row["positive"])) != (2881, 756):
            raise ValueError("Primary cohort mismatch")
        for key in ("auprc", "auroc"):
            value = float(row[key])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {key}")
    return rows


def main():
    print(f"{'Representation':30} {'AUPRC':>8} {'AUROC':>8}")
    for row in read_results():
        print(f"{row['model']:30} {float(row['auprc']):8.4f} {float(row['auroc']):8.4f}")
    print("Aggregate point estimates only; no new training or uncertainty calculation.")


if __name__ == "__main__":
    main()
