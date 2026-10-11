from __future__ import annotations

import torch
from torch import nn


class AttentionPool(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.score = nn.Linear(dimension, 1, bias=False)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(features).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(features * weights.unsqueeze(-1), dim=1)


class BinderClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 2560,
        projection_dim: int = 128,
        num_heads: int = 4,
        model_type: str = "cross_attention",
        projection_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        classifier_dropout: float = 0.2,
    ):
        super().__init__()
        if model_type not in {"pooling", "cross_attention"}:
            raise ValueError(f"Unsupported model type: {model_type}")
        if projection_dim % num_heads != 0:
            raise ValueError("projection_dim must be divisible by num_heads")
        self.model_type = model_type
        self.target_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(projection_dropout),
        )
        self.binder_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(projection_dropout),
        )
        self.target_pool = AttentionPool(projection_dim)
        self.binder_pool = AttentionPool(projection_dim)
        if model_type == "cross_attention":
            self.cross_attention = nn.MultiheadAttention(
                projection_dim,
                num_heads,
                dropout=attention_dropout,
                batch_first=True,
            )
            self.cross_dropout = nn.Dropout(attention_dropout)
            self.cross_norm = nn.LayerNorm(projection_dim)
            self.interaction_pool = AttentionPool(projection_dim)
            fusion_dim = 5 * projection_dim
        else:
            fusion_dim = 4 * projection_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(projection_dim, 1),
        )

    def forward(
        self,
        target: torch.Tensor,
        target_mask: torch.Tensor,
        binder: torch.Tensor,
        binder_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_features = self.target_projection(target.float())
        binder_features = self.binder_projection(binder.float())
        target_vector = self.target_pool(target_features, target_mask)
        binder_vector = self.binder_pool(binder_features, binder_mask)
        fusion = [
            target_vector,
            binder_vector,
            torch.abs(target_vector - binder_vector),
            target_vector * binder_vector,
        ]
        if self.model_type == "cross_attention":
            cross_features, _ = self.cross_attention(
                query=binder_features,
                key=target_features,
                value=target_features,
                key_padding_mask=~target_mask,
                need_weights=False,
            )
            interaction_features = self.cross_norm(
                binder_features + self.cross_dropout(cross_features)
            )
            fusion.append(self.interaction_pool(interaction_features, binder_mask))
        return self.classifier(torch.cat(fusion, dim=-1)).squeeze(-1)
