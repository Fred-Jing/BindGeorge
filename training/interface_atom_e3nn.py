from __future__ import annotations

import math

import torch
from e3nn import o3
from e3nn.nn import Gate
from torch import nn


ELEMENT_VOCAB_SIZE = 8
RESIDUE_VOCAB_SIZE = 22
ATOM_NAME_VOCAB_SIZE = 39
BACKBONE_VOCAB_SIZE = 3
ROLE_VOCAB_SIZE = 2
INTERACTION_MODES = ("full", "no_cross_message", "partner_marginal")
PAIR_GATE_MODES = ("none", "unknown", "real", "shuffled")


def segment_sum(
    values: torch.Tensor, index: torch.Tensor, count: int
) -> torch.Tensor:
    output = values.new_zeros((count, values.shape[-1]))
    output.index_add_(0, index, values)
    return output


def distance_rbf(distance: torch.Tensor, bins: int, cutoff: float) -> torch.Tensor:
    centers = torch.linspace(0.0, cutoff, bins, device=distance.device, dtype=distance.dtype)
    width = cutoff / max(bins - 1, 1)
    features = torch.exp(-0.5 * ((distance[:, None] - centers) / width) ** 2)
    envelope = 0.5 * (torch.cos(torch.pi * distance / cutoff) + 1.0)
    return features * envelope.clamp_min(0.0)[:, None]


class EquivariantMessageLayer(nn.Module):
    def __init__(
        self,
        irreps_input: o3.Irreps,
        *,
        hidden_scalars: int = 16,
        hidden_vectors: int = 4,
        rbf_bins: int = 16,
        relation_dimension: int = 4,
        radial_hidden_dimension: int = 64,
        cutoff: float = 6.0,
    ):
        super().__init__()
        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_hidden = o3.Irreps(
            f"{hidden_scalars}x0e+{hidden_vectors}x1o"
        )
        self.sh_irreps = o3.Irreps.spherical_harmonics(1)
        self.gate = Gate(
            f"{hidden_scalars}x0e",
            [torch.nn.functional.silu],
            f"{hidden_vectors}x0e",
            [torch.sigmoid],
            f"{hidden_vectors}x1o",
        )
        self.tensor_product = o3.FullyConnectedTensorProduct(
            self.irreps_input,
            self.sh_irreps,
            self.gate.irreps_in,
            internal_weights=False,
            shared_weights=False,
        )
        self.relation_embedding = nn.Embedding(3, relation_dimension)
        self.radial_network = nn.Sequential(
            nn.Linear(rbf_bins + relation_dimension, radial_hidden_dimension),
            nn.SiLU(),
            nn.Linear(radial_hidden_dimension, self.tensor_product.weight_numel),
        )
        self.skip = o3.Linear(self.irreps_input, self.irreps_hidden)
        self.rbf_bins = rbf_bins
        self.cutoff = cutoff

    def forward(
        self,
        node_features: torch.Tensor,
        position: torch.Tensor,
        source: torch.Tensor,
        destination: torch.Tensor,
        relation: torch.Tensor,
        edge_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_vector = position[source] - position[destination]
        distance = torch.linalg.vector_norm(edge_vector, dim=-1).clamp_min(1e-8)
        spherical_harmonics = o3.spherical_harmonics(
            self.sh_irreps,
            edge_vector,
            normalize=True,
            normalization="component",
        )
        radial_input = torch.cat(
            [
                distance_rbf(distance, self.rbf_bins, self.cutoff),
                self.relation_embedding(relation),
            ],
            dim=-1,
        )
        weights = self.radial_network(radial_input)
        messages = self.tensor_product(
            node_features[source], spherical_harmonics, weights
        )
        if edge_gate is not None:
            messages = messages * edge_gate
        aggregated = node_features.new_zeros(
            (len(node_features), self.gate.irreps_in.dim)
        )
        aggregated.index_add_(0, destination, messages)
        degree = node_features.new_zeros((len(node_features), 1))
        degree.index_add_(
            0, destination, torch.ones_like(destination, dtype=node_features.dtype)[:, None]
        )
        aggregated = aggregated / degree.clamp_min(1.0).sqrt()
        return (self.gate(aggregated) + self.skip(node_features)) / math.sqrt(2.0)


class ChemistryPairGate(nn.Module):
    def __init__(
        self,
        chemistry_dimension: int,
        *,
        rbf_bins: int = 16,
        relation_dimension: int = 4,
        hidden_dimension: int = 32,
        cutoff: float = 6.0,
        alpha: float = 1.0,
    ):
        super().__init__()
        if not 0.0 < alpha <= 1.0:
            raise ValueError("Pair-gate alpha must be in (0, 1]")
        self.relation_embedding = nn.Embedding(3, relation_dimension)
        pair_dimension = chemistry_dimension * 4 + rbf_bins + relation_dimension
        self.network = nn.Sequential(
            nn.Linear(pair_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.rbf_bins = rbf_bins
        self.cutoff = cutoff
        self.alpha = alpha

    def forward(
        self,
        chemistry: torch.Tensor,
        position: torch.Tensor,
        source: torch.Tensor,
        destination: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        gate = chemistry.new_ones((len(source), 1))
        cross_index = torch.nonzero(relation == 2, as_tuple=False).flatten()
        if len(cross_index) == 0:
            return gate
        cross_source = source[cross_index]
        cross_destination = destination[cross_index]
        source_chemistry = chemistry[cross_source]
        destination_chemistry = chemistry[cross_destination]
        distance = torch.linalg.vector_norm(
            position[cross_source] - position[cross_destination], dim=-1
        )
        pair_features = torch.cat(
            [
                source_chemistry,
                destination_chemistry,
                source_chemistry * destination_chemistry,
                torch.abs(source_chemistry - destination_chemistry),
                distance_rbf(distance, self.rbf_bins, self.cutoff),
                self.relation_embedding(relation[cross_index]),
            ],
            dim=-1,
        )
        gate[cross_index] = 1.0 + self.alpha * torch.tanh(
            self.network(pair_features) / 2.0
        )
        return gate


class InterfaceAtomE3NNClassifier(nn.Module):
    def __init__(
        self,
        *,
        hidden_scalars: int = 16,
        hidden_vectors: int = 4,
        message_layers: int = 2,
        rbf_bins: int = 16,
        message_cutoff: float = 6.0,
        classifier_dropout: float = 0.2,
        interaction_mode: str = "full",
        pair_gate_mode: str = "none",
        pair_gate_alpha: float = 1.0,
    ):
        super().__init__()
        if message_layers < 2:
            raise ValueError("At least two message layers are required")
        if interaction_mode not in INTERACTION_MODES:
            raise ValueError(f"Unsupported interaction mode: {interaction_mode}")
        if pair_gate_mode not in PAIR_GATE_MODES:
            raise ValueError(f"Unsupported pair gate mode: {pair_gate_mode}")
        if not 0.0 < pair_gate_alpha <= 1.0:
            raise ValueError("Pair-gate alpha must be in (0, 1]")
        self.element_embedding = nn.Embedding(ELEMENT_VOCAB_SIZE, 3)
        self.residue_embedding = nn.Embedding(RESIDUE_VOCAB_SIZE, 4)
        self.atom_name_embedding = nn.Embedding(ATOM_NAME_VOCAB_SIZE, 4)
        self.backbone_embedding = nn.Embedding(BACKBONE_VOCAB_SIZE, 2)
        self.role_embedding = nn.Embedding(ROLE_VOCAB_SIZE, 3)
        input_irreps = o3.Irreps("16x0e")
        hidden_irreps = o3.Irreps(f"{hidden_scalars}x0e+{hidden_vectors}x1o")
        layers = []
        for layer_index in range(message_layers):
            layers.append(
                EquivariantMessageLayer(
                    input_irreps if layer_index == 0 else hidden_irreps,
                    hidden_scalars=hidden_scalars,
                    hidden_vectors=hidden_vectors,
                    rbf_bins=rbf_bins,
                    cutoff=message_cutoff,
                )
            )
        self.message_layers = nn.ModuleList(layers)
        self.scalar_readout = o3.Linear(hidden_irreps, f"{hidden_scalars}x0e")
        self.core_edge_network = nn.Sequential(
            nn.Linear(hidden_scalars * 2 + rbf_bins, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.core_gate = nn.Linear(64, 1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(32, 1),
        )
        self.rbf_bins = rbf_bins
        self.core_cutoff = 5.0
        self.interaction_mode = interaction_mode
        self.pair_gate_mode = pair_gate_mode
        self.pair_gate_alpha = pair_gate_alpha

        self.gate_element_embedding = None
        self.gate_atom_name_embedding = None
        self.gate_backbone_embedding = None
        self.pair_gates = nn.ModuleList()
        if pair_gate_mode != "none":
            # Appending these modules preserves the seeded initialization of
            # the validated backbone used in the preceding experiments.
            self.gate_element_embedding = nn.Embedding(ELEMENT_VOCAB_SIZE, 3)
            self.gate_atom_name_embedding = nn.Embedding(ATOM_NAME_VOCAB_SIZE, 4)
            self.gate_backbone_embedding = nn.Embedding(BACKBONE_VOCAB_SIZE, 2)
            self.pair_gates = nn.ModuleList(
                [
                    ChemistryPairGate(
                        9,
                        rbf_bins=rbf_bins,
                        cutoff=message_cutoff,
                        alpha=pair_gate_alpha,
                    )
                    for _ in range(message_layers)
                ]
            )

    def forward(
        self,
        *,
        node_position: torch.Tensor,
        node_element: torch.Tensor,
        node_residue_type: torch.Tensor,
        node_atom_name: torch.Tensor,
        node_backbone: torch.Tensor,
        node_role: torch.Tensor,
        gate_node_element: torch.Tensor,
        gate_node_atom_name: torch.Tensor,
        gate_node_backbone: torch.Tensor,
        node_graph_index: torch.Tensor,
        message_source: torch.Tensor,
        message_destination: torch.Tensor,
        message_relation: torch.Tensor,
        core_target_node: torch.Tensor,
        core_binder_node: torch.Tensor,
        core_graph_index: torch.Tensor,
        graph_count: int,
    ) -> torch.Tensor:
        node_features = torch.cat(
            [
                self.element_embedding(node_element),
                self.residue_embedding(node_residue_type),
                self.atom_name_embedding(node_atom_name),
                self.backbone_embedding(node_backbone),
                self.role_embedding(node_role),
            ],
            dim=-1,
        )
        gate_chemistry = None
        if self.pair_gate_mode != "none":
            gate_chemistry = torch.cat(
                [
                    self.gate_element_embedding(gate_node_element),
                    self.gate_atom_name_embedding(gate_node_atom_name),
                    self.gate_backbone_embedding(gate_node_backbone),
                ],
                dim=-1,
            )
        if self.interaction_mode != "full":
            intra_chain = message_relation != 2
            message_source = message_source[intra_chain]
            message_destination = message_destination[intra_chain]
            message_relation = message_relation[intra_chain]
        for layer_index, layer in enumerate(self.message_layers):
            edge_gate = None
            if self.pair_gate_mode != "none":
                edge_gate = self.pair_gates[layer_index](
                    gate_chemistry,
                    node_position,
                    message_source,
                    message_destination,
                    message_relation,
                )
            node_features = layer(
                node_features,
                node_position,
                message_source,
                message_destination,
                message_relation,
                edge_gate,
            )
        scalar = self.scalar_readout(node_features)
        if self.interaction_mode == "partner_marginal":
            role_graph_index = node_graph_index * ROLE_VOCAB_SIZE + node_role
            role_count = graph_count * ROLE_VOCAB_SIZE
            role_sum = segment_sum(scalar, role_graph_index, role_count)
            role_normalizer = segment_sum(
                scalar.new_ones((len(scalar), 1)), role_graph_index, role_count
            ).clamp_min(1.0)
            role_mean = (role_sum / role_normalizer).reshape(graph_count, -1)
            marginal_input = torch.cat(
                [role_mean, role_mean.new_zeros((graph_count, self.rbf_bins))], dim=-1
            )
            return self.classifier(self.core_edge_network(marginal_input)).squeeze(-1)
        core_vector = (
            node_position[core_binder_node] - node_position[core_target_node]
        )
        core_distance = torch.linalg.vector_norm(core_vector, dim=-1)
        core_features = self.core_edge_network(
            torch.cat(
                [
                    scalar[core_target_node],
                    scalar[core_binder_node],
                    distance_rbf(core_distance, self.rbf_bins, self.core_cutoff),
                ],
                dim=-1,
            )
        )
        gate = torch.sigmoid(self.core_gate(core_features))
        pooled = segment_sum(gate * core_features, core_graph_index, graph_count)
        normalizer = segment_sum(gate, core_graph_index, graph_count).clamp_min(1e-6)
        return self.classifier(pooled / normalizer).squeeze(-1)
