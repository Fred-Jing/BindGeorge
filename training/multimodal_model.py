from __future__ import annotations

import math

import torch
from torch import nn


MODEL_VARIANTS = {
    "confidence": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": False,
        "use_residue_cross_attention": False,
        "use_confidence": True,
    },
    "esmc": {
        "use_esmc": True,
        "use_af3_single": False,
        "use_af3_pair": False,
        "use_residue_cross_attention": True,
        "use_confidence": False,
    },
    "af3_single": {
        "use_esmc": False,
        "use_af3_single": True,
        "use_af3_pair": False,
        "use_residue_cross_attention": True,
        "use_confidence": False,
    },
    "af3_single_readout": {
        "use_esmc": False,
        "use_af3_single": True,
        "use_af3_pair": False,
        "use_residue_cross_attention": True,
        "use_confidence": False,
        "use_grouped_single_readout": True,
    },
    "af3_single_pair_bias_readout": {
        "use_esmc": False,
        "use_af3_single": True,
        "use_af3_pair": True,
        "use_residue_cross_attention": True,
        "use_confidence": False,
        "use_grouped_single_readout": True,
        "use_direct_pair_pool": False,
        "use_pair_bias": True,
    },
    "af3_single_pair_concat_readout": {
        "use_esmc": False,
        "use_af3_single": True,
        "use_af3_pair": True,
        "use_residue_cross_attention": True,
        "use_confidence": False,
        "use_grouped_single_readout": True,
        "use_direct_pair_pool": True,
        "use_pair_bias": False,
    },
    "af3_single_pair_bias_concat_readout": {
        "use_esmc": False,
        "use_af3_single": True,
        "use_af3_pair": True,
        "use_residue_cross_attention": True,
        "use_confidence": False,
        "use_grouped_single_readout": True,
        "use_direct_pair_pool": True,
        "use_pair_bias": True,
    },
    "af3_pair": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": False,
    },
    "af3_pair_readout_adapter": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": False,
        "use_pair_readout_adapter": True,
    },
    "af3_pair_interface_mask": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": False,
        "use_pair_distance_mask": True,
    },
    "af3_pair_random_count_mask": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": False,
        "use_pair_selection_mask": True,
        "pair_selection_mode": "random_count",
    },
    "af3_pair_degree_rewired_mask": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": False,
        "use_pair_selection_mask": True,
        "pair_selection_mode": "degree_rewired",
    },
    "af3_pair_geometry": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": False,
        "use_geometry": True,
        "geometry_mode": "attention_prior",
    },
    "geometry_rbf": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": False,
        "use_residue_cross_attention": False,
        "use_confidence": False,
        "use_geometry": True,
        "geometry_mode": "rbf_only",
        "use_direct_pair_pool": False,
    },
    "af3_pair_geometry_cell_residual": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": False,
        "use_geometry": True,
        "geometry_mode": "cell_residual",
    },
    "af3_pair_cell_capacity_control": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": False,
        "geometry_mode": "cell_capacity_control",
    },
    "af3_pair_geometry_dual": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": False,
        "use_geometry": True,
        "geometry_mode": "dual_branch",
    },
    "af3_pair_dual_capacity_control": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": False,
        "geometry_mode": "dual_capacity_control",
    },
    "af3_pair_geometry_canonical_dual": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": False,
        "use_geometry_summary": True,
        "geometry_mode": "canonical_dual",
    },
    "af3_pair_canonical_capacity_control": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": False,
        "geometry_mode": "canonical_capacity_control",
    },
    "af3_pair_geometry_canonical_shuffled": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": False,
        "use_geometry_summary": True,
        "shuffle_geometry_within_target": True,
        "geometry_mode": "canonical_shuffled",
    },
    "af3_pair_descriptor_dual": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": False,
        "use_interface_descriptor": True,
        "descriptor_mode": "descriptor_dual",
    },
    "af3_pair_descriptor_capacity_control": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": False,
        "descriptor_mode": "descriptor_capacity_control",
    },
    "af3_pair_descriptor_shuffled": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": False,
        "use_interface_descriptor": True,
        "shuffle_interface_descriptor_within_target": True,
        "descriptor_mode": "descriptor_shuffled",
    },
    "af3_pair_geometry_cell_hierarchical": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": False,
        "use_geometry_cells": True,
        "geometry_mode": "geometry_cell_hierarchical",
    },
    "af3_pair_geometry_cell_hierarchical_capacity_control": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": False,
        "geometry_mode": "geometry_cell_hierarchical_capacity_control",
    },
    "af3_pair_geometry_cell_hierarchical_shuffled": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": False,
        "use_geometry_cells": True,
        "shuffle_geometry_axes": True,
        "geometry_mode": "geometry_cell_hierarchical_shuffled",
    },
    "af3_pair_confidence": {
        "use_esmc": False,
        "use_af3_single": False,
        "use_af3_pair": True,
        "use_residue_cross_attention": False,
        "use_confidence": True,
    },
    "af3_single_pair": {
        "use_esmc": False,
        "use_af3_single": True,
        "use_af3_pair": True,
        "use_residue_cross_attention": True,
        "use_confidence": False,
    },
    "full": {
        "use_esmc": True,
        "use_af3_single": True,
        "use_af3_pair": True,
        "use_residue_cross_attention": True,
        "use_confidence": False,
    },
    "full_confidence": {
        "use_esmc": True,
        "use_af3_single": True,
        "use_af3_pair": True,
        "use_residue_cross_attention": True,
        "use_confidence": True,
    },
}

PAIR_FUSIONS = ("projected_sum", "concat", "separate_concat", "sum_absdiff")
PAIR_READOUT_MODES = ("none", "layernorm", "linear_residual", "mlp_residual")
SINGLE_READOUTS = (
    "legacy_all",
    "interaction",
    "interaction_raw",
    "interaction_matching",
    "interaction_raw_matching",
    "matching",
)
PAIR_POOLINGS = (
    "attention",
    "mean",
    "mean_max",
    "multi_query",
    "hierarchical_target_first",
    "hierarchical_gated_target_first",
    "hierarchical_geometry_target_first",
    "hierarchical_binder_first",
)
GEOMETRY_POOLINGS = ("mean", "mean_max")


class AttentionPool(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.score = nn.Linear(dimension, 1, bias=False)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(features).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(features.dtype)
        return torch.sum(features * weights.unsqueeze(-1), dim=1)


class PairAttentionPool(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.output_dimension = dimension
        self.score = nn.Linear(dimension, 1, bias=False)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, target_length, binder_length, dimension = features.shape
        flattened = features.reshape(batch_size, target_length * binder_length, dimension)
        flattened_mask = mask.reshape(batch_size, target_length * binder_length)
        scores = self.score(flattened).squeeze(-1)
        scores = scores.masked_fill(~flattened_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(features.dtype)
        return torch.sum(flattened * weights.unsqueeze(-1), dim=1)


class PairMeanPool(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.output_dimension = dimension

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(features.dtype)
        denominator = weights.sum(dim=(1, 2)).clamp_min(1.0)
        return (features * weights).sum(dim=(1, 2)) / denominator


class PairMeanMaxPool(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.output_dimension = 2 * dimension

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(features.dtype)
        denominator = weights.sum(dim=(1, 2)).clamp_min(1.0)
        mean_features = (features * weights).sum(dim=(1, 2)) / denominator
        masked_features = features.masked_fill(
            ~mask.unsqueeze(-1), torch.finfo(features.dtype).min
        )
        max_features = masked_features.amax(dim=(1, 2))
        return torch.cat([mean_features, max_features], dim=-1)


class DistanceRBF(nn.Module):
    """Encode scalar distances with fixed Gaussian radial basis functions."""

    def __init__(self, bins: int, minimum: float, maximum: float):
        super().__init__()
        if bins < 2:
            raise ValueError("distance RBF requires at least two bins")
        if maximum <= minimum:
            raise ValueError("distance RBF maximum must exceed its minimum")
        centers = torch.linspace(minimum, maximum, bins)
        self.register_buffer("centers", centers)
        self.width = float(centers[1] - centers[0])

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        scaled = (distances.float().unsqueeze(-1) - self.centers) / self.width
        return torch.exp(-0.5 * scaled.square())


class PairMultiQueryAttentionPool(nn.Module):
    def __init__(self, dimension: int, queries: int):
        super().__init__()
        if queries < 1:
            raise ValueError("pair pooling queries must be positive")
        self.queries = queries
        self.output_dimension = queries * dimension
        self.score = nn.Linear(dimension, queries, bias=False)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, target_length, binder_length, dimension = features.shape
        flattened = features.reshape(batch_size, target_length * binder_length, dimension)
        flattened_mask = mask.reshape(batch_size, target_length * binder_length)
        scores = self.score(flattened).permute(0, 2, 1)
        scores = scores.masked_fill(
            ~flattened_mask.unsqueeze(1), torch.finfo(scores.dtype).min
        )
        weights = torch.softmax(scores.float(), dim=-1).to(features.dtype)
        pooled = torch.matmul(weights, flattened)
        return pooled.reshape(batch_size, self.output_dimension)


class PairHierarchicalAttentionPool(nn.Module):
    def __init__(self, dimension: int, *, target_first: bool):
        super().__init__()
        self.output_dimension = dimension
        self.target_first = target_first
        self.first_score = nn.Linear(dimension, 1, bias=False)
        self.second_score = nn.Linear(dimension, 1, bias=False)

    @staticmethod
    def _masked_attention_pool(
        features: torch.Tensor,
        mask: torch.Tensor,
        score: nn.Linear,
        dimension: int,
    ) -> torch.Tensor:
        scores = score(features).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=dimension).to(features.dtype)
        return torch.sum(features * weights.unsqueeze(-1), dim=dimension)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.target_first:
            intermediate = self._masked_attention_pool(
                features, mask, self.first_score, dimension=1
            )
            intermediate_mask = mask.any(dim=1)
        else:
            intermediate = self._masked_attention_pool(
                features, mask, self.first_score, dimension=2
            )
            intermediate_mask = mask.any(dim=2)
        return self._masked_attention_pool(
            intermediate, intermediate_mask, self.second_score, dimension=1
        )


class PairHierarchicalGatedPool(nn.Module):
    """Pool pair cells while preserving absolute evidence at both levels."""

    def __init__(self, dimension: int):
        super().__init__()
        hidden_dimension = max(8, dimension // 2)
        self.output_dimension = dimension + 2
        self.first_score = nn.Sequential(
            nn.Linear(dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )
        self.second_score = nn.Sequential(
            nn.Linear(dimension + 1, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )

    @staticmethod
    def _masked_gated_pool(
        features: torch.Tensor,
        mask: torch.Tensor,
        score: nn.Module,
        dimension: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = score(features).squeeze(-1)
        gates = torch.sigmoid(scores.float()).to(features.dtype)
        gates = gates * mask.to(features.dtype)
        gate_sum = gates.sum(dim=dimension)
        pooled = torch.sum(
            features * gates.unsqueeze(-1), dim=dimension
        ) / gate_sum.clamp_min(1e-6).unsqueeze(-1)
        valid_count = mask.sum(dim=dimension).clamp_min(1).to(gate_sum.dtype)
        coverage = gate_sum / valid_count
        return pooled, coverage

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        intermediate, first_coverage = self._masked_gated_pool(
            features, mask, self.first_score, dimension=1
        )
        intermediate_mask = mask.any(dim=1)
        intermediate_with_evidence = torch.cat(
            [intermediate, first_coverage.unsqueeze(-1)], dim=-1
        )
        pooled, second_coverage = self._masked_gated_pool(
            intermediate_with_evidence,
            intermediate_mask,
            self.second_score,
            dimension=1,
        )
        return torch.cat([pooled, second_coverage.unsqueeze(-1)], dim=-1)


class PairHierarchicalGeometryAttentionPool(nn.Module):
    """Apply a fixed soft C-beta distance prior to hierarchical attention."""

    def __init__(
        self,
        dimension: int,
        *,
        distance_cutoff: float = 8.0,
        transition_width: float = 1.0,
    ):
        super().__init__()
        if transition_width <= 0:
            raise ValueError("geometry transition width must be positive")
        self.output_dimension = dimension
        self.distance_cutoff = distance_cutoff
        self.transition_width = transition_width
        self.first_score = nn.Linear(dimension, 1, bias=False)
        self.second_score = nn.Linear(dimension, 1, bias=False)

    @staticmethod
    def _masked_prior_attention_pool(
        features: torch.Tensor,
        mask: torch.Tensor,
        prior: torch.Tensor,
        score: nn.Linear,
        dimension: int,
    ) -> torch.Tensor:
        scores = score(features).squeeze(-1).float()
        scores = scores + torch.log(prior.float().clamp_min(1e-6))
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=dimension).to(features.dtype)
        return torch.sum(features * weights.unsqueeze(-1), dim=dimension)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        if distances.shape != mask.shape:
            raise ValueError(
                f"Pair distance/mask mismatch: {distances.shape} != {mask.shape}"
            )
        prior = torch.sigmoid(
            (self.distance_cutoff - distances.float()) / self.transition_width
        )
        prior = prior * mask.to(prior.dtype)
        intermediate = self._masked_prior_attention_pool(
            features, mask, prior, self.first_score, dimension=1
        )
        intermediate_mask = mask.any(dim=1)
        binder_prior = prior.amax(dim=1)
        return self._masked_prior_attention_pool(
            intermediate,
            intermediate_mask,
            binder_prior,
            self.second_score,
            dimension=1,
        )


def make_pair_pool(
    pooling: str, dimension: int, queries: int
) -> nn.Module:
    if pooling == "attention":
        return PairAttentionPool(dimension)
    if pooling == "mean":
        return PairMeanPool(dimension)
    if pooling == "mean_max":
        return PairMeanMaxPool(dimension)
    if pooling == "multi_query":
        return PairMultiQueryAttentionPool(dimension, queries)
    if pooling == "hierarchical_target_first":
        return PairHierarchicalAttentionPool(dimension, target_first=True)
    if pooling == "hierarchical_gated_target_first":
        return PairHierarchicalGatedPool(dimension)
    if pooling == "hierarchical_geometry_target_first":
        return PairHierarchicalGeometryAttentionPool(dimension)
    if pooling == "hierarchical_binder_first":
        return PairHierarchicalAttentionPool(dimension, target_first=False)
    raise ValueError(f"Unsupported pair pooling: {pooling}")


class PairBiasedCrossAttention(nn.Module):
    def __init__(
        self,
        dimension: int,
        num_heads: int,
        pair_dimension: int | None,
        dropout: float,
    ):
        super().__init__()
        if dimension % num_heads != 0:
            raise ValueError("dimension must be divisible by num_heads")
        self.dimension = dimension
        self.num_heads = num_heads
        self.head_dimension = dimension // num_heads
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)
        self.pair_bias = (
            nn.Linear(pair_dimension, num_heads, bias=False)
            if pair_dimension is not None
            else None
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, 2 * dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dimension, dimension),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dimension)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = tensor.shape
        return tensor.reshape(
            batch_size, length, self.num_heads, self.head_dimension
        ).permute(0, 2, 1, 3)

    def forward(
        self,
        *,
        binder: torch.Tensor,
        target: torch.Tensor,
        binder_mask: torch.Tensor,
        target_mask: torch.Tensor,
        pair_features: torch.Tensor | None,
    ) -> torch.Tensor:
        query = self._split_heads(self.query(binder))
        key = self._split_heads(self.key(target))
        value = self._split_heads(self.value(target))
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self.head_dimension
        )
        if pair_features is not None:
            if self.pair_bias is None:
                raise ValueError("Pair features were provided to attention without pair bias")
            bias = self.pair_bias(pair_features).permute(0, 3, 2, 1)
            logits = logits + bias
        logits = logits.masked_fill(
            ~target_mask[:, None, None, :], torch.finfo(logits.dtype).min
        )
        attention = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        attention = self.attention_dropout(attention)
        context = torch.matmul(attention, value)
        context = context.permute(0, 2, 1, 3).reshape(
            binder.shape[0], binder.shape[1], self.dimension
        )
        context = self.output_dropout(self.output(context))
        hidden = self.norm1(binder + context)
        hidden = self.norm2(hidden + self.ffn(hidden))
        return hidden * binder_mask.unsqueeze(-1).to(hidden.dtype)


class PairConditionedBinderClassifier(nn.Module):
    def __init__(
        self,
        *,
        variant: str = "full",
        esmc_dimension: int = 2560,
        af3_single_dimension: int = 384,
        af3_pair_dimension: int = 128,
        residue_dimension: int = 128,
        projected_pair_dimension: int = 32,
        num_heads: int = 4,
        projection_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        classifier_dropout: float = 0.2,
        confidence_dimension: int = 3,
        pair_fusion: str = "projected_sum",
        pair_pooling: str = "attention",
        pair_pooling_queries: int = 1,
        single_readout: str = "legacy_all",
        raw_branch_dimension: int = 32,
        matching_branch_dimension: int = 32,
        single_fusion_dimension: int = 64,
        geometry_rbf_bins: int = 16,
        geometry_rbf_minimum: float = 2.0,
        geometry_rbf_maximum: float = 32.0,
        geometry_pooling: str = "mean_max",
        canonical_geometry_dimension: int = 52,
        geometry_cell_dimension: int = 26,
        pair_distance_cutoff: float = 10.0,
        interface_descriptor_dimension: int = 118,
        descriptor_branch_dimension: int = 8,
        pair_readout_mode: str = "none",
        pair_readout_hidden_dimension: int = 32,
        pair_readout_depth: int = 1,
    ):
        super().__init__()
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"Unsupported model variant: {variant}")
        if pair_fusion not in PAIR_FUSIONS:
            raise ValueError(f"Unsupported pair fusion: {pair_fusion}")
        if pair_pooling not in PAIR_POOLINGS:
            raise ValueError(f"Unsupported pair pooling: {pair_pooling}")
        if single_readout not in SINGLE_READOUTS:
            raise ValueError(f"Unsupported single readout: {single_readout}")
        if geometry_pooling not in GEOMETRY_POOLINGS:
            raise ValueError(f"Unsupported geometry pooling: {geometry_pooling}")
        if pair_readout_mode not in PAIR_READOUT_MODES:
            raise ValueError(f"Unsupported pair readout mode: {pair_readout_mode}")
        self.variant = variant
        flags = MODEL_VARIANTS[variant]
        self.use_esmc = flags["use_esmc"]
        self.use_af3_single = flags["use_af3_single"]
        self.use_af3_pair = flags["use_af3_pair"]
        self.use_residue_cross_attention = flags["use_residue_cross_attention"]
        self.use_confidence = flags["use_confidence"]
        self.use_geometry = flags.get("use_geometry", False)
        self.use_geometry_summary = flags.get("use_geometry_summary", False)
        self.use_geometry_cells = flags.get("use_geometry_cells", False)
        self.use_interface_descriptor = flags.get(
            "use_interface_descriptor", False
        )
        self.use_pair_readout_adapter = flags.get(
            "use_pair_readout_adapter", False
        )
        if self.use_pair_readout_adapter != (pair_readout_mode != "none"):
            raise ValueError(f"Inconsistent pair readout adapter for {variant}")
        if pair_readout_hidden_dimension < 1 or pair_readout_depth < 1:
            raise ValueError("Pair readout dimensions must be positive")
        if pair_readout_mode != "mlp_residual" and pair_readout_depth != 1:
            raise ValueError("Pair readout depth only applies to mlp_residual")
        self.pair_readout_mode = pair_readout_mode
        self.descriptor_mode = flags.get("descriptor_mode")
        self.use_pair_distance_mask = flags.get("use_pair_distance_mask", False)
        self.use_pair_selection_mask = flags.get("use_pair_selection_mask", False)
        self.geometry_mode = flags.get("geometry_mode")
        if self.use_pair_distance_mask and (
            not self.use_af3_pair or pair_distance_cutoff <= 0
        ):
            raise ValueError(
                "Pair distance masking requires AF3 pair features and a positive cutoff"
            )
        if self.use_pair_selection_mask and not self.use_af3_pair:
            raise ValueError("Pair selection masking requires AF3 pair features")
        if self.use_pair_distance_mask and self.use_pair_selection_mask:
            raise ValueError("Pair distance and selection masks are mutually exclusive")
        self.pair_distance_cutoff = pair_distance_cutoff
        geometry_input_modes = {
            "attention_prior",
            "rbf_only",
            "cell_residual",
            "dual_branch",
        }
        if self.use_geometry != (self.geometry_mode in geometry_input_modes):
            raise ValueError(f"Inconsistent geometry configuration for {variant}")
        canonical_modes = {
            "canonical_dual",
            "canonical_capacity_control",
            "canonical_shuffled",
        }
        if self.use_geometry_summary != (
            self.geometry_mode in {"canonical_dual", "canonical_shuffled"}
        ):
            raise ValueError(f"Inconsistent canonical geometry input for {variant}")
        if not (
            projected_pair_dimension
            <= canonical_geometry_dimension
            <= 2 * projected_pair_dimension
        ):
            raise ValueError(
                "Canonical geometry dimension must be between one and two pair widths"
            )
        geometry_cell_modes = {
            "geometry_cell_hierarchical",
            "geometry_cell_hierarchical_capacity_control",
            "geometry_cell_hierarchical_shuffled",
        }
        if self.use_geometry_cells != (
            self.geometry_mode
            in {"geometry_cell_hierarchical", "geometry_cell_hierarchical_shuffled"}
        ):
            raise ValueError(f"Inconsistent geometry cell input for {variant}")
        descriptor_modes = {
            "descriptor_dual",
            "descriptor_capacity_control",
            "descriptor_shuffled",
        }
        if self.descriptor_mode is not None and self.descriptor_mode not in descriptor_modes:
            raise ValueError(f"Unsupported descriptor mode: {self.descriptor_mode}")
        if self.use_interface_descriptor != (
            self.descriptor_mode in {"descriptor_dual", "descriptor_shuffled"}
        ):
            raise ValueError(f"Inconsistent interface descriptor input for {variant}")
        if min(interface_descriptor_dimension, descriptor_branch_dimension) < 1:
            raise ValueError("Interface descriptor dimensions must be positive")
        if not 1 <= geometry_cell_dimension <= projected_pair_dimension:
            raise ValueError("Geometry cell dimension must not exceed pair dimension")
        self.use_grouped_single_readout = flags.get(
            "use_grouped_single_readout", False
        )
        self.use_direct_pair_pool = flags.get(
            "use_direct_pair_pool", self.use_af3_pair
        )
        self.use_pair_bias = flags.get(
            "use_pair_bias",
            self.use_af3_pair and self.use_residue_cross_attention,
        )
        if self.use_pair_bias and not (
            self.use_af3_pair and self.use_residue_cross_attention
        ):
            raise ValueError("Pair bias requires AF3 pair and cross-attention branches")
        if self.use_grouped_single_readout == (single_readout == "legacy_all"):
            expected = "a grouped readout" if self.use_grouped_single_readout else "legacy_all"
            raise ValueError(f"Variant {variant} requires {expected}")
        self.single_readout = single_readout
        self.use_interaction_readout = self.use_grouped_single_readout and single_readout in {
            "interaction",
            "interaction_raw",
            "interaction_matching",
            "interaction_raw_matching",
        }
        self.use_raw_readout = self.use_grouped_single_readout and single_readout in {
            "interaction_raw",
            "interaction_raw_matching",
        }
        self.use_matching_readout = self.use_grouped_single_readout and single_readout in {
            "interaction_matching",
            "interaction_raw_matching",
            "matching",
        }
        if self.use_grouped_single_readout:
            self.use_residue_cross_attention = self.use_interaction_readout
            if min(
                raw_branch_dimension,
                matching_branch_dimension,
                single_fusion_dimension,
            ) < 1:
                raise ValueError("Grouped single readout dimensions must be positive")
            if self.use_pair_bias and not self.use_interaction_readout:
                raise ValueError("Pair-bias readout requires an interaction branch")
        self.residue_dimension = residue_dimension
        self.projected_pair_dimension = projected_pair_dimension
        self.pair_fusion = pair_fusion
        self.pair_pooling = pair_pooling
        self.geometry_pooling = geometry_pooling
        self.distance_rbf_bins = geometry_rbf_bins

        if self.geometry_mode == "attention_prior" and pair_pooling != (
            "hierarchical_geometry_target_first"
        ):
            raise ValueError(
                "Geometry attention prior requires hierarchical_geometry_target_first"
            )
        if self.geometry_mode in {
            "cell_residual",
            "cell_capacity_control",
            "dual_branch",
            "dual_capacity_control",
        } and (
            pair_pooling == "hierarchical_geometry_target_first"
        ):
            raise ValueError("Learned geometry fusion cannot use the fixed geometry prior")

        if self.use_esmc:
            self.esmc_projection = nn.Sequential(
                nn.LayerNorm(esmc_dimension),
                nn.Linear(esmc_dimension, residue_dimension),
                nn.GELU(),
                nn.Dropout(projection_dropout),
            )
        if self.use_af3_single:
            self.single_projection = nn.Sequential(
                nn.LayerNorm(af3_single_dimension),
                nn.Linear(af3_single_dimension, residue_dimension),
                nn.GELU(),
                nn.Dropout(projection_dropout),
            )
        if self.use_esmc and self.use_af3_single:
            self.modality_fusion = nn.Sequential(
                nn.Linear(2 * residue_dimension, residue_dimension),
                nn.GELU(),
                nn.Dropout(projection_dropout),
            )
        if self.use_esmc or self.use_af3_single:
            self.role_embedding = nn.Parameter(torch.zeros(2, residue_dimension))
            nn.init.normal_(self.role_embedding, std=0.02)
            self.target_fusion_norm = nn.LayerNorm(residue_dimension)
            self.binder_fusion_norm = nn.LayerNorm(residue_dimension)
            if not self.use_grouped_single_readout or (
                self.use_raw_readout or self.use_matching_readout
            ):
                self.target_pool = AttentionPool(residue_dimension)
                self.binder_pool = AttentionPool(residue_dimension)
            if self.use_residue_cross_attention:
                self.interaction_pool = AttentionPool(residue_dimension)

        if self.use_af3_pair:
            self.target_binder_pair_norm = nn.LayerNorm(af3_pair_dimension)
            self.binder_target_pair_norm = nn.LayerNorm(af3_pair_dimension)
            if pair_fusion == "projected_sum":
                self.target_binder_pair_projection = nn.Linear(
                    af3_pair_dimension, projected_pair_dimension, bias=False
                )
                self.binder_target_pair_projection = nn.Linear(
                    af3_pair_dimension, projected_pair_dimension, bias=False
                )
                self.pair_projection_bias = nn.Parameter(
                    torch.zeros(projected_pair_dimension)
                )
            elif pair_fusion == "separate_concat":
                if projected_pair_dimension % 2 != 0:
                    raise ValueError(
                        "separate_concat requires an even pair dimension"
                    )
                direction_dimension = projected_pair_dimension // 2
                self.target_binder_pair_projection = nn.Linear(
                    af3_pair_dimension, direction_dimension, bias=False
                )
                self.binder_target_pair_projection = nn.Linear(
                    af3_pair_dimension, direction_dimension, bias=False
                )
                self.pair_projection_bias = nn.Parameter(
                    torch.zeros(projected_pair_dimension)
                )
            else:
                self.pair_direction_fusion_projection = nn.Linear(
                    2 * af3_pair_dimension, projected_pair_dimension
                )
            self.pair_activation = nn.GELU()
            self.pair_dropout = nn.Dropout(projection_dropout)
            if self.use_direct_pair_pool:
                self.pair_pool = make_pair_pool(
                    pair_pooling, projected_pair_dimension, pair_pooling_queries
                )

        if self.use_residue_cross_attention:
            self.cross_attention = PairBiasedCrossAttention(
                residue_dimension,
                num_heads,
                projected_pair_dimension if self.use_pair_bias else None,
                attention_dropout,
            )

        if self.use_confidence:
            self.confidence_projection = nn.Sequential(
                nn.Linear(confidence_dimension, 16),
                nn.GELU(),
            )

        if self.geometry_mode in {"rbf_only", "cell_residual", "dual_branch"}:
            self.distance_rbf = DistanceRBF(
                geometry_rbf_bins,
                geometry_rbf_minimum,
                geometry_rbf_maximum,
            )
        if self.geometry_mode in {
            "rbf_only",
            "dual_branch",
            "dual_capacity_control",
        }:
            if self.geometry_mode != "dual_capacity_control":
                self.geometry_pool = make_pair_pool(
                    geometry_pooling, geometry_rbf_bins, pair_pooling_queries
                )
                geometry_summary_dimension = self.geometry_pool.output_dimension
            else:
                geometry_summary_dimension = 2 * geometry_rbf_bins
                if geometry_summary_dimension != projected_pair_dimension:
                    raise ValueError(
                        "Dual capacity control requires pair dimension to equal "
                        "twice the geometry RBF bins"
                    )
            self.geometry_summary_projection = nn.Sequential(
                nn.Linear(geometry_summary_dimension, projected_pair_dimension),
                nn.GELU(),
                nn.Dropout(projection_dropout),
            )
        if self.geometry_mode in {"cell_residual", "cell_capacity_control"}:
            if geometry_rbf_bins > projected_pair_dimension:
                raise ValueError(
                    "Cell capacity control requires RBF bins not to exceed pair dimension"
                )
            self.geometry_cell_residual = nn.Sequential(
                nn.Linear(
                    projected_pair_dimension + geometry_rbf_bins,
                    projected_pair_dimension,
                ),
                nn.GELU(),
                nn.Dropout(projection_dropout),
                nn.Linear(projected_pair_dimension, projected_pair_dimension),
            )
            nn.init.normal_(self.geometry_cell_residual[-1].weight, std=1e-3)
            nn.init.zeros_(self.geometry_cell_residual[-1].bias)
        if self.geometry_mode in {"dual_branch", "dual_capacity_control"}:
            if self.pair_pool.output_dimension != projected_pair_dimension:
                raise ValueError(
                    "Dual geometry branch requires pair pooling to preserve pair dimension"
                )
            self.geometry_dual_residual = nn.Sequential(
                nn.Linear(2 * projected_pair_dimension, projected_pair_dimension),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(projected_pair_dimension, projected_pair_dimension),
            )
            nn.init.normal_(self.geometry_dual_residual[-1].weight, std=1e-3)
            nn.init.zeros_(self.geometry_dual_residual[-1].bias)
        if self.geometry_mode in canonical_modes:
            if self.pair_pool.output_dimension != projected_pair_dimension:
                raise ValueError(
                    "Canonical geometry fusion requires pair pooling to preserve pair dimension"
                )
            self.canonical_summary_projection = nn.Sequential(
                nn.Linear(canonical_geometry_dimension, projected_pair_dimension),
                nn.GELU(),
                nn.Dropout(projection_dropout),
            )
            self.canonical_dual_residual = nn.Sequential(
                nn.Linear(2 * projected_pair_dimension, projected_pair_dimension),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(projected_pair_dimension, projected_pair_dimension),
            )
            nn.init.normal_(self.canonical_dual_residual[-1].weight, std=1e-3)
            nn.init.zeros_(self.canonical_dual_residual[-1].bias)
            self.canonical_geometry_dimension = canonical_geometry_dimension
        if self.geometry_mode in geometry_cell_modes:
            self.geometry_cell_projection = nn.Sequential(
                nn.Linear(geometry_cell_dimension, projected_pair_dimension),
                nn.GELU(),
                nn.Dropout(projection_dropout),
            )
            self.geometry_cell_pool = make_pair_pool(
                pair_pooling, projected_pair_dimension, pair_pooling_queries
            )
            if self.geometry_cell_pool.output_dimension != projected_pair_dimension:
                raise ValueError(
                    "Geometry cell fusion requires pooling to preserve pair dimension"
                )
            self.geometry_cell_dual_residual = nn.Sequential(
                nn.Linear(2 * projected_pair_dimension, projected_pair_dimension),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(projected_pair_dimension, projected_pair_dimension),
            )
            nn.init.normal_(self.geometry_cell_dual_residual[-1].weight, std=1e-3)
            nn.init.zeros_(self.geometry_cell_dual_residual[-1].bias)
            self.geometry_cell_dimension = geometry_cell_dimension
        if self.use_pair_readout_adapter:
            if self.pair_pool.output_dimension != projected_pair_dimension:
                raise ValueError(
                    "Pair readout adapter requires pooling to preserve pair dimension"
                )
            rng_state = torch.random.get_rng_state()
            try:
                self.pair_readout_norm = nn.LayerNorm(projected_pair_dimension)
                if pair_readout_mode == "layernorm":
                    self.pair_readout_residual = None
                elif pair_readout_mode == "linear_residual":
                    self.pair_readout_residual = nn.Linear(
                        projected_pair_dimension, projected_pair_dimension
                    )
                else:
                    layers: list[nn.Module] = []
                    input_dimension = projected_pair_dimension
                    for _ in range(pair_readout_depth):
                        layers.extend(
                            [
                                nn.Linear(
                                    input_dimension, pair_readout_hidden_dimension
                                ),
                                nn.GELU(),
                                nn.Dropout(classifier_dropout),
                            ]
                        )
                        input_dimension = pair_readout_hidden_dimension
                    layers.append(
                        nn.Linear(input_dimension, projected_pair_dimension)
                    )
                    self.pair_readout_residual = nn.Sequential(*layers)
                if self.pair_readout_residual is not None:
                    final_layer = (
                        self.pair_readout_residual
                        if isinstance(self.pair_readout_residual, nn.Linear)
                        else self.pair_readout_residual[-1]
                    )
                    nn.init.normal_(final_layer.weight, std=1e-3)
                    nn.init.zeros_(final_layer.bias)
            finally:
                # Adapter initialization must not alter shared classifier weights.
                torch.random.set_rng_state(rng_state)
        if self.descriptor_mode in descriptor_modes:
            if self.pair_pool.output_dimension != projected_pair_dimension:
                raise ValueError(
                    "Interface descriptor fusion requires pair pooling to preserve pair dimension"
                )
            self.interface_descriptor_adapter = nn.Sequential(
                nn.LayerNorm(interface_descriptor_dimension),
                nn.Linear(interface_descriptor_dimension, descriptor_branch_dimension),
                nn.GELU(),
                nn.Dropout(projection_dropout),
            )
            self.interface_descriptor_residual = nn.Sequential(
                nn.Linear(
                    projected_pair_dimension + descriptor_branch_dimension,
                    projected_pair_dimension,
                ),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(projected_pair_dimension, projected_pair_dimension),
            )
            nn.init.normal_(self.interface_descriptor_residual[-1].weight, std=1e-3)
            nn.init.zeros_(self.interface_descriptor_residual[-1].bias)
            self.interface_descriptor_dimension = interface_descriptor_dimension

        fusion_dimension = 0
        if self.use_grouped_single_readout:
            grouped_dimension = 0
            if self.use_interaction_readout:
                grouped_dimension += residue_dimension
            if self.use_raw_readout:
                self.raw_adapter = nn.Sequential(
                    nn.Linear(2 * residue_dimension, raw_branch_dimension),
                    nn.GELU(),
                    nn.Dropout(projection_dropout),
                )
                grouped_dimension += raw_branch_dimension
            if self.use_matching_readout:
                self.matching_adapter = nn.Sequential(
                    nn.Linear(2 * residue_dimension, matching_branch_dimension),
                    nn.GELU(),
                    nn.Dropout(projection_dropout),
                )
                grouped_dimension += matching_branch_dimension
            self.single_fusion = nn.Sequential(
                nn.Linear(grouped_dimension, single_fusion_dimension),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
            )
            fusion_dimension += single_fusion_dimension
        elif self.use_residue_cross_attention:
            fusion_dimension += 5 * residue_dimension
        if self.geometry_mode == "rbf_only":
            fusion_dimension += projected_pair_dimension
        elif self.use_direct_pair_pool:
            fusion_dimension += self.pair_pool.output_dimension
        if self.use_confidence:
            fusion_dimension += 16
        if fusion_dimension == 0:
            raise ValueError("Model variant has no active inputs")
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dimension, residue_dimension),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(residue_dimension, 32),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(32, 1),
        )

    def _pair_readout(self, pair_vector: torch.Tensor) -> torch.Tensor:
        normalized = self.pair_readout_norm(pair_vector)
        if self.pair_readout_mode == "layernorm":
            return normalized
        return pair_vector + self.pair_readout_residual(normalized)

    def _residue_features(
        self,
        *,
        esmc: torch.Tensor | None,
        single: torch.Tensor | None,
        role_index: int,
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        esmc_features = self.esmc_projection(esmc.float()) if self.use_esmc else None
        single_features = (
            self.single_projection(single.float()) if self.use_af3_single else None
        )
        if esmc_features is not None and single_features is not None:
            fused = esmc_features + self.modality_fusion(
                torch.cat([esmc_features, single_features], dim=-1)
            )
        elif esmc_features is not None:
            fused = esmc_features
        elif single_features is not None:
            fused = single_features
        else:
            raise RuntimeError("Residue feature branch has no inputs")
        return norm(fused + self.role_embedding[role_index])

    def _pair_features(
        self,
        target_binder_pair: torch.Tensor,
        binder_target_pair: torch.Tensor,
    ) -> torch.Tensor:
        target_binder = self.target_binder_pair_norm(target_binder_pair.float())
        binder_target = self.binder_target_pair_norm(binder_target_pair.float())
        if self.pair_fusion == "projected_sum":
            fused = (
                self.target_binder_pair_projection(target_binder)
                + self.binder_target_pair_projection(binder_target)
                + self.pair_projection_bias
            )
        elif self.pair_fusion == "separate_concat":
            fused = torch.cat(
                [
                    self.target_binder_pair_projection(target_binder),
                    self.binder_target_pair_projection(binder_target),
                ],
                dim=-1,
            ) + self.pair_projection_bias
        elif self.pair_fusion == "concat":
            fused = self.pair_direction_fusion_projection(
                torch.cat([target_binder, binder_target], dim=-1)
            )
        elif self.pair_fusion == "sum_absdiff":
            fused = self.pair_direction_fusion_projection(
                torch.cat(
                    [
                        target_binder + binder_target,
                        torch.abs(target_binder - binder_target),
                    ],
                    dim=-1,
                )
            )
        else:
            raise RuntimeError(f"Unsupported pair fusion: {self.pair_fusion}")
        return self.pair_dropout(
            self.pair_activation(fused)
        )

    def forward(
        self,
        *,
        target_mask: torch.Tensor,
        binder_mask: torch.Tensor,
        target_esmc: torch.Tensor | None = None,
        binder_esmc: torch.Tensor | None = None,
        target_single: torch.Tensor | None = None,
        binder_single: torch.Tensor | None = None,
        target_binder_pair: torch.Tensor | None = None,
        binder_target_pair: torch.Tensor | None = None,
        target_binder_distance: torch.Tensor | None = None,
        target_binder_selection_mask: torch.Tensor | None = None,
        target_binder_geometry_summary: torch.Tensor | None = None,
        target_binder_geometry: torch.Tensor | None = None,
        interface_descriptor: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pair_mask = target_mask.unsqueeze(2) & binder_mask.unsqueeze(1)
        if self.use_pair_distance_mask:
            if target_binder_distance is None:
                raise ValueError("Pair distance masking requires pair distances")
            pair_mask = pair_mask & (
                target_binder_distance <= self.pair_distance_cutoff
            )
            if not pair_mask.flatten(1).any(dim=1).all():
                raise ValueError("Pair distance cutoff removed every pair cell")
        if self.use_pair_selection_mask:
            if target_binder_selection_mask is None:
                raise ValueError("Pair selection masking requires a selection mask")
            if target_binder_selection_mask.dtype != torch.bool:
                raise ValueError("Pair selection mask must be boolean")
            pair_mask = pair_mask & target_binder_selection_mask
            if not pair_mask.flatten(1).any(dim=1).all():
                raise ValueError("Pair selection mask removed every pair cell")
        pair_features = None
        geometry_features = None
        fusion = []
        if self.use_geometry:
            if target_binder_distance is None:
                raise ValueError("Pair distances are required by this model variant")
            if self.geometry_mode != "attention_prior":
                geometry_features = self.distance_rbf(target_binder_distance)
        if self.use_af3_pair:
            if target_binder_pair is None or binder_target_pair is None:
                raise ValueError("AF3 pair tensors are required by this model variant")
            pair_features = self._pair_features(target_binder_pair, binder_target_pair)
            if self.geometry_mode == "cell_residual":
                pair_features = pair_features + self.geometry_cell_residual(
                    torch.cat([pair_features, geometry_features], dim=-1)
                )
            elif self.geometry_mode == "cell_capacity_control":
                pair_features = pair_features + self.geometry_cell_residual(
                    torch.cat(
                        [
                            pair_features,
                            pair_features[..., : self.distance_rbf_bins],
                        ],
                        dim=-1,
                    )
                )

        if self.use_residue_cross_attention or self.use_grouped_single_readout:
            target = self._residue_features(
                esmc=target_esmc,
                single=target_single,
                role_index=0,
                norm=self.target_fusion_norm,
            )
            binder = self._residue_features(
                esmc=binder_esmc,
                single=binder_single,
                role_index=1,
                norm=self.binder_fusion_norm,
            )
            target = target * target_mask.unsqueeze(-1).to(target.dtype)
            binder = binder * binder_mask.unsqueeze(-1).to(binder.dtype)
            target_vector = None
            binder_vector = None
            if not self.use_grouped_single_readout or (
                self.use_raw_readout or self.use_matching_readout
            ):
                target_vector = self.target_pool(target, target_mask)
                binder_vector = self.binder_pool(binder, binder_mask)
            interaction_vector = None
            if self.use_residue_cross_attention:
                interaction = self.cross_attention(
                    binder=binder,
                    target=target,
                    binder_mask=binder_mask,
                    target_mask=target_mask,
                    pair_features=pair_features if self.use_pair_bias else None,
                )
                interaction_vector = self.interaction_pool(interaction, binder_mask)
            if self.use_grouped_single_readout:
                grouped = []
                if self.use_interaction_readout:
                    grouped.append(interaction_vector)
                if self.use_raw_readout:
                    grouped.append(
                        self.raw_adapter(torch.cat([target_vector, binder_vector], dim=-1))
                    )
                if self.use_matching_readout:
                    grouped.append(
                        self.matching_adapter(
                            torch.cat(
                                [
                                    torch.abs(target_vector - binder_vector),
                                    target_vector * binder_vector,
                                ],
                                dim=-1,
                            )
                        )
                    )
                fusion.append(self.single_fusion(torch.cat(grouped, dim=-1)))
            else:
                fusion.extend(
                    [
                        target_vector,
                        binder_vector,
                        torch.abs(target_vector - binder_vector),
                        target_vector * binder_vector,
                        interaction_vector,
                    ]
                )
        if self.use_direct_pair_pool:
            if self.descriptor_mode in {
                "descriptor_dual",
                "descriptor_capacity_control",
                "descriptor_shuffled",
            }:
                pair_vector = self.pair_pool(pair_features, pair_mask)
                if self.descriptor_mode == "descriptor_capacity_control":
                    repeats = math.ceil(
                        self.interface_descriptor_dimension / pair_vector.shape[-1]
                    )
                    descriptor_input = pair_vector.repeat(1, repeats)[
                        ..., : self.interface_descriptor_dimension
                    ]
                else:
                    if interface_descriptor is None:
                        raise ValueError(
                            "Interface descriptor is required by this model variant"
                        )
                    if interface_descriptor.shape[-1] != self.interface_descriptor_dimension:
                        raise ValueError(
                            "Unexpected interface descriptor width: "
                            f"{interface_descriptor.shape[-1]}"
                        )
                    descriptor_input = interface_descriptor.float()
                descriptor_vector = self.interface_descriptor_adapter(descriptor_input)
                fusion.append(
                    pair_vector
                    + self.interface_descriptor_residual(
                        torch.cat([pair_vector, descriptor_vector], dim=-1)
                    )
                )
            elif self.geometry_mode == "attention_prior":
                fusion.append(
                    self.pair_pool(
                        pair_features, pair_mask, target_binder_distance
                    )
                )
            elif self.geometry_mode in {"dual_branch", "dual_capacity_control"}:
                pair_vector = self.pair_pool(pair_features, pair_mask)
                if self.geometry_mode == "dual_branch":
                    auxiliary_input = self.geometry_pool(geometry_features, pair_mask)
                else:
                    auxiliary_input = pair_vector
                geometry_vector = self.geometry_summary_projection(auxiliary_input)
                fusion.append(
                    pair_vector
                    + self.geometry_dual_residual(
                        torch.cat([pair_vector, geometry_vector], dim=-1)
                    )
                )
            elif self.geometry_mode in {
                "canonical_dual",
                "canonical_capacity_control",
                "canonical_shuffled",
            }:
                pair_vector = self.pair_pool(pair_features, pair_mask)
                if self.geometry_mode == "canonical_capacity_control":
                    auxiliary_input = torch.cat(
                        [
                            pair_vector,
                            pair_vector[
                                ..., : self.canonical_geometry_dimension - pair_vector.shape[-1]
                            ],
                        ],
                        dim=-1,
                    )
                else:
                    if target_binder_geometry_summary is None:
                        raise ValueError(
                            "Canonical geometry summary is required by this model variant"
                        )
                    auxiliary_input = target_binder_geometry_summary.float()
                geometry_vector = self.canonical_summary_projection(auxiliary_input)
                fusion.append(
                    pair_vector
                    + self.canonical_dual_residual(
                        torch.cat([pair_vector, geometry_vector], dim=-1)
                    )
                )
            elif self.geometry_mode in {
                "geometry_cell_hierarchical",
                "geometry_cell_hierarchical_capacity_control",
                "geometry_cell_hierarchical_shuffled",
            }:
                pair_vector = self.pair_pool(pair_features, pair_mask)
                if self.geometry_mode == (
                    "geometry_cell_hierarchical_capacity_control"
                ):
                    geometry_input = pair_features.detach()[
                        ..., : self.geometry_cell_dimension
                    ]
                else:
                    if target_binder_geometry is None:
                        raise ValueError(
                            "Pair-cell geometry is required by this model variant"
                        )
                    geometry_input = target_binder_geometry.float()
                geometry_features = self.geometry_cell_projection(geometry_input)
                geometry_vector = self.geometry_cell_pool(
                    geometry_features, pair_mask
                )
                fusion.append(
                    pair_vector
                    + self.geometry_cell_dual_residual(
                        torch.cat([pair_vector, geometry_vector], dim=-1)
                    )
                )
            else:
                pair_vector = self.pair_pool(pair_features, pair_mask)
                if self.use_pair_readout_adapter:
                    pair_vector = self._pair_readout(pair_vector)
                fusion.append(pair_vector)
        if self.geometry_mode == "rbf_only":
            fusion.append(
                self.geometry_summary_projection(
                    self.geometry_pool(geometry_features, pair_mask)
                )
            )
        if self.use_confidence:
            if confidence is None:
                raise ValueError("Confidence features are required by this model variant")
            fusion.append(self.confidence_projection(confidence.float()))
        return self.classifier(torch.cat(fusion, dim=-1)).squeeze(-1)
