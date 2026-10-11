#!/usr/bin/env python3
"""CPU synthetic forward/backward checks for all multimodal v2 variants."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.multimodal_model import (  # noqa: E402
    MODEL_VARIANTS,
    PairConditionedBinderClassifier,
)


def make_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(20260727)
    batch_size = 3
    target_length = 9
    binder_length = 7
    target_mask = torch.arange(target_length).unsqueeze(0) < torch.tensor(
        [[9], [6], [8]]
    )
    binder_mask = torch.arange(binder_length).unsqueeze(0) < torch.tensor(
        [[7], [5], [6]]
    )
    return {
        "target_mask": target_mask,
        "binder_mask": binder_mask,
        "target_esmc": torch.randn(batch_size, target_length, 2560),
        "binder_esmc": torch.randn(batch_size, binder_length, 2560),
        "target_single": torch.randn(batch_size, target_length, 384),
        "binder_single": torch.randn(batch_size, binder_length, 384),
        "target_binder_pair": torch.randn(
            batch_size, target_length, binder_length, 128
        ),
        "binder_target_pair": torch.randn(
            batch_size, target_length, binder_length, 128
        ),
        "target_binder_distance": torch.rand(
            batch_size, target_length, binder_length
        )
        * 7.0
        + 2.0,
        "target_binder_selection_mask": torch.ones(
            batch_size, target_length, binder_length, dtype=torch.bool
        ),
        "target_binder_geometry_summary": torch.randn(batch_size, 32),
        "target_binder_geometry": torch.randn(
            batch_size, target_length, binder_length, 16
        ),
        "interface_descriptor": torch.randn(batch_size, 118),
        "confidence": torch.randn(batch_size, 3),
    }


def inputs_for_variant(
    batch: dict[str, torch.Tensor], variant: str
) -> dict[str, torch.Tensor]:
    flags = MODEL_VARIANTS[variant]
    keys = ["target_mask", "binder_mask"]
    if flags["use_esmc"]:
        keys.extend(["target_esmc", "binder_esmc"])
    if flags["use_af3_single"]:
        keys.extend(["target_single", "binder_single"])
    if flags["use_af3_pair"]:
        keys.extend(["target_binder_pair", "binder_target_pair"])
    if flags.get("use_pair_distance_mask") or flags.get("use_geometry"):
        keys.append("target_binder_distance")
    if flags.get("use_pair_selection_mask"):
        keys.append("target_binder_selection_mask")
    if flags.get("use_geometry_summary"):
        keys.append("target_binder_geometry_summary")
    if flags.get("use_geometry_cells"):
        keys.append("target_binder_geometry")
    if flags.get("use_interface_descriptor"):
        keys.append("interface_descriptor")
    if flags["use_confidence"]:
        keys.append("confidence")
    return {key: batch[key] for key in keys}


def main() -> None:
    batch = make_batch()
    labels = torch.tensor([1.0, 0.0, 1.0])
    for variant in MODEL_VARIANTS:
        flags = MODEL_VARIANTS[variant]
        model = PairConditionedBinderClassifier(
            variant=variant,
            residue_dimension=64,
            projected_pair_dimension=16,
            canonical_geometry_dimension=32,
            geometry_cell_dimension=16,
            geometry_rbf_bins=8,
            num_heads=4,
            pair_pooling=(
                "hierarchical_geometry_target_first"
                if flags.get("geometry_mode") == "attention_prior"
                else "attention"
            ),
            pair_readout_mode=(
                "layernorm" if flags.get("use_pair_readout_adapter") else "none"
            ),
            single_readout=(
                "interaction" if flags.get("use_grouped_single_readout") else "legacy_all"
            ),
            projection_dropout=0.0,
            attention_dropout=0.0,
            classifier_dropout=0.0,
        )
        inputs = inputs_for_variant(batch, variant)
        logits = model(**inputs)
        if logits.shape != (3,) or not torch.isfinite(logits).all():
            raise AssertionError(f"Invalid logits for {variant}: {logits}")
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise AssertionError(f"Invalid gradients for {variant}")
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"variant={variant} parameters={parameters} "
            f"loss={float(loss.detach()):.6f} status=ok"
        )
    print("synthetic_multimodal_v2=PASS")


if __name__ == "__main__":
    main()
