"""Deterministic fixed-width descriptors for predicted protein interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree, distance


AA_ORDER = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {residue: index for index, residue in enumerate(AA_ORDER)}
RESIDUE_HEAVY_ELEMENTS = {
    "A": (3, 1, 1, 0), "R": (6, 4, 1, 0), "N": (4, 2, 2, 0),
    "D": (4, 1, 3, 0), "C": (3, 1, 1, 1), "E": (5, 1, 3, 0),
    "Q": (5, 2, 2, 0), "G": (2, 1, 1, 0), "H": (6, 3, 1, 0),
    "I": (6, 1, 1, 0), "L": (6, 1, 1, 0), "K": (6, 2, 1, 0),
    "M": (5, 1, 1, 1), "F": (9, 1, 1, 0), "P": (5, 1, 1, 0),
    "S": (3, 1, 2, 0), "T": (4, 1, 2, 0), "W": (11, 2, 1, 0),
    "Y": (9, 1, 2, 0), "V": (5, 1, 1, 0),
}
ATOM_DISTANCE_BINS = np.asarray([0.0, 2.0, 2.5, 3.5, 5.0, 8.0, 12.0])
CA_DISTANCE_BINS = np.asarray([0.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0])
CONTACT_THRESHOLDS = (2.5, 3.5, 5.0, 8.0, 10.0, 12.0)
ROTATION_ANGLE_BINS = np.linspace(0.0, np.pi, 7)
POSE_CONTACT_CUTOFF = 12.0
INTERFACE_CUTOFF = 8.0


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    group: str
    subgroup: str


class FeatureCollector:
    def __init__(self) -> None:
        self.values: list[float] = []
        self.definitions: list[FeatureDefinition] = []

    def add(self, name: str, value: float, group: str, subgroup: str = "") -> None:
        self.values.append(float(value))
        self.definitions.append(FeatureDefinition(name, group, subgroup))

    def add_vector(
        self,
        prefix: str,
        values: np.ndarray | list[float],
        labels: list[str] | tuple[str, ...],
        group: str,
        subgroup: str = "",
    ) -> None:
        if len(values) != len(labels):
            raise ValueError(f"Feature label mismatch for {prefix}")
        for label, value in zip(labels, values, strict=True):
            self.add(f"{prefix}_{label}", float(value), group, subgroup)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def distribution_statistics(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(8, dtype=np.float64)
    return np.asarray(
        [
            values.min(), values.mean(), values.std(),
            *np.quantile(values, [0.25, 0.5, 0.75, 0.9]), values.max(),
        ],
        dtype=np.float64,
    )


def degree_statistics(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(6, dtype=np.float64)
    return np.asarray(
        [values.mean(), values.std(), *np.quantile(values, [0.25, 0.5, 0.75]), values.max()]
    )


def local_frames(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    e1 = c - ca
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True).clip(min=1e-8)
    n_vector = n - ca
    e2 = n_vector - np.sum(n_vector * e1, axis=1, keepdims=True) * e1
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True).clip(min=1e-8)
    e3 = np.cross(e1, e2)
    e3 /= np.linalg.norm(e3, axis=1, keepdims=True).clip(min=1e-8)
    return np.stack([e1, e2, e3], axis=2)


def atom_cloud(
    ca: np.ndarray,
    local: np.ndarray,
    mask: np.ndarray,
    sidechain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = mask.astype(bool)
    coordinates = ca[:, None, :] + local.astype(np.float32)
    residue = np.broadcast_to(np.arange(len(ca), dtype=np.int32)[:, None], valid.shape)
    return coordinates[valid], residue[valid], sidechain[valid].astype(bool)


def cross_atom_contacts(
    target: tuple[np.ndarray, np.ndarray, np.ndarray],
    binder: tuple[np.ndarray, np.ndarray, np.ndarray],
    cutoff: float = POSE_CONTACT_CUTOFF,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = cKDTree(target[0]).sparse_distance_matrix(
        cKDTree(binder[0]), cutoff, output_type="coo_matrix"
    )
    return matrix.row.astype(np.int32), matrix.col.astype(np.int32), matrix.data


def residue_contacts(
    target_atom_residue: np.ndarray,
    binder_atom_residue: np.ndarray,
    atom_target: np.ndarray,
    atom_binder: np.ndarray,
    atom_distance: np.ndarray,
    binder_length: int,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = atom_distance <= cutoff
    flat = (
        target_atom_residue[atom_target[keep]].astype(np.int64) * binder_length
        + binder_atom_residue[atom_binder[keep]]
    )
    if flat.size == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )
    unique, inverse = np.unique(flat, return_inverse=True)
    minimum = np.full(len(unique), np.inf, dtype=np.float64)
    np.minimum.at(minimum, inverse, atom_distance[keep])
    return (
        (unique // binder_length).astype(np.int32),
        (unique % binder_length).astype(np.int32),
        minimum,
    )


def connected_components(
    target_nodes: np.ndarray, binder_nodes: np.ndarray, target_length: int, binder_length: int
) -> tuple[int, float]:
    present_target = np.unique(target_nodes)
    present_binder = np.unique(binder_nodes)
    if len(target_nodes) == 0:
        return 0, 0.0
    parent = np.arange(target_length + binder_length, dtype=np.int32)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for target_node, binder_node in zip(target_nodes, binder_nodes, strict=True):
        left = find(int(target_node))
        right = find(target_length + int(binder_node))
        if left != right:
            parent[right] = left
    present = np.concatenate([present_target, target_length + present_binder])
    roots, counts = np.unique([find(int(node)) for node in present], return_counts=True)
    del roots
    return len(counts), float(counts.max() / len(present))


def shape_features(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(11, dtype=np.float64)
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(1, len(points))
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1].clip(min=0.0)
    total = eigenvalues.sum()
    fractions = eigenvalues / total if total > 1e-12 else np.zeros(3)
    radius = float(np.sqrt(total))
    l1, l2, l3 = eigenvalues
    anisotropy = safe_divide(l1 - l3, l1)
    linearity = safe_divide(l1 - l2, l1)
    planarity = safe_divide(l2 - l3, l1)
    sphericity = safe_divide(l3, l1)
    maximum_span = float(distance.pdist(points).max()) if len(points) > 1 else 0.0
    normal_thickness = float(np.sqrt(l3))
    tangential_radius = float(np.sqrt(l1 + l2))
    return np.asarray(
        [*fractions, radius, anisotropy, linearity, planarity, sphericity,
         maximum_span, normal_thickness, tangential_radius]
    )


def principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        return np.eye(3), np.zeros(3)
    centered = points - points.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered / len(points))
    order = np.argsort(values)[::-1]
    return vectors[:, order], values[order].clip(min=0.0)


def composition(sequence: str, indices: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(AA_ORDER), dtype=np.float64)
    for index in indices:
        residue = sequence[int(index)]
        if residue in AA_INDEX:
            counts[AA_INDEX[residue]] += 1.0
    return counts / counts.sum() if counts.sum() else counts


def element_fractions(sequence: str, indices: np.ndarray) -> np.ndarray:
    counts = np.zeros(4, dtype=np.float64)
    for index in indices:
        formula = RESIDUE_HEAVY_ELEMENTS.get(sequence[int(index)])
        if formula is not None:
            counts += formula
    return counts / counts.sum() if counts.sum() else counts


def _add_size_and_topology(
    collector: FeatureCollector,
    target_length: int,
    binder_length: int,
    target_interface: np.ndarray,
    binder_interface: np.ndarray,
    target_nodes: np.ndarray,
    binder_nodes: np.ndarray,
    atom_contact_count: int,
) -> None:
    size_values = (
        target_length, binder_length, target_length + binder_length,
        np.log1p(target_length), np.log1p(binder_length),
        safe_divide(binder_length, target_length),
        len(target_interface), len(binder_interface),
        len(target_interface) + len(binder_interface),
        np.log1p(len(target_nodes)), np.log1p(atom_contact_count),
        safe_divide(len(binder_interface), len(target_interface)),
    )
    size_names = (
        "target_length", "binder_length", "total_length", "log_target_length",
        "log_binder_length", "binder_target_length_ratio", "target_interface_residues",
        "binder_interface_residues", "total_interface_residues", "log_residue_contacts",
        "log_atom_contacts", "binder_target_interface_ratio",
    )
    collector.add_vector("size", size_values, size_names, "g0_topology", "size")
    collector.add("topology_target_interface_fraction", safe_divide(len(target_interface), target_length), "g0_topology")
    collector.add("topology_binder_interface_fraction", safe_divide(len(binder_interface), binder_length), "g0_topology")
    collector.add("topology_residue_contact_density", safe_divide(len(target_nodes), len(target_interface) * len(binder_interface)), "g0_topology")
    target_degree = np.bincount(target_nodes, minlength=target_length)[target_interface]
    binder_degree = np.bincount(binder_nodes, minlength=binder_length)[binder_interface]
    labels = ("mean", "std", "q25", "q50", "q75", "max")
    collector.add_vector("topology_target_degree", degree_statistics(target_degree), labels, "g0_topology")
    collector.add_vector("topology_binder_degree", degree_statistics(binder_degree), labels, "g0_topology")
    components, largest = connected_components(target_nodes, binder_nodes, target_length, binder_length)
    collector.add("topology_component_count", components, "g0_topology")
    collector.add("topology_largest_component_fraction", largest, "g0_topology")
    collector.add("topology_edge_per_interface_residue", safe_divide(len(target_nodes), len(target_interface) + len(binder_interface)), "g0_topology")


def _add_distances(
    collector: FeatureCollector,
    atom_distances: np.ndarray,
    ca_distances: np.ndarray,
    residue_minimum: np.ndarray,
    residue_target: np.ndarray,
    residue_binder: np.ndarray,
    target_interface: np.ndarray,
    binder_interface: np.ndarray,
) -> None:
    for prefix, values, denominator in (
        ("atom", atom_distances, max(1, len(atom_distances))),
        ("ca", ca_distances, max(1, len(ca_distances))),
    ):
        counts = np.asarray([np.count_nonzero(values <= threshold) for threshold in CONTACT_THRESHOLDS])
        if prefix == "atom":
            collector.add_vector(f"distance_{prefix}_log_count_le", np.log1p(counts), [f"{x:g}a" for x in CONTACT_THRESHOLDS], "g1_distance")
        collector.add_vector(f"distance_{prefix}_fraction_le", counts / denominator, [f"{x:g}a" for x in CONTACT_THRESHOLDS], "g1_distance")
    stats = ("min", "mean", "std", "q25", "q50", "q75", "q90", "max")
    collector.add_vector("distance_atom", distribution_statistics(atom_distances), stats, "g1_distance")
    collector.add_vector("distance_ca", distribution_statistics(ca_distances), stats, "g1_distance")
    for prefix, values, bins in (("atom", atom_distances, ATOM_DISTANCE_BINS), ("ca", ca_distances, CA_DISTANCE_BINS)):
        histogram = np.histogram(values, bins=bins)[0].astype(np.float64)
        histogram /= histogram.sum() if histogram.sum() else 1.0
        labels = [f"{bins[i]:g}_{bins[i + 1]:g}a" for i in range(len(bins) - 1)]
        collector.add_vector(f"distance_{prefix}_hist", histogram, labels, "g1_distance")
    min_labels = ("min", "mean", "std", "q25", "q50", "q75", "max")
    for role, nodes, interface in (("target", residue_target, target_interface), ("binder", residue_binder, binder_interface)):
        values = np.zeros(len(interface), dtype=np.float64)
        for position, node in enumerate(interface):
            selected = residue_minimum[nodes == node]
            values[position] = selected.min() if selected.size else 0.0
        stats7 = distribution_statistics(values)[[0, 1, 2, 3, 4, 5, 7]]
        collector.add_vector(f"distance_{role}_minimum", stats7, min_labels, "g1_distance")


def _add_pose(
    collector: FeatureCollector,
    target_n: np.ndarray,
    target_ca: np.ndarray,
    target_c: np.ndarray,
    binder_n: np.ndarray,
    binder_ca: np.ndarray,
    binder_c: np.ndarray,
    target_nodes: np.ndarray,
    binder_nodes: np.ndarray,
) -> None:
    target_frame = local_frames(target_n, target_ca, target_c)
    binder_frame = local_frames(binder_n, binder_ca, binder_c)
    displacement = binder_ca[binder_nodes] - target_ca[target_nodes]
    norm = np.linalg.norm(displacement, axis=1).clip(min=1e-8)
    target_direction = np.einsum("nij,nj->ni", target_frame[target_nodes].transpose(0, 2, 1), displacement) / norm[:, None]
    binder_direction = np.einsum("nij,nj->ni", binder_frame[binder_nodes].transpose(0, 2, 1), -displacement) / norm[:, None]
    for role, vectors in (("target", target_direction), ("binder", binder_direction)):
        collector.add_vector(f"pose_{role}_direction_mean", vectors.mean(axis=0), ("x", "y", "z"), "g2_pose")
        collector.add_vector(f"pose_{role}_direction_std", vectors.std(axis=0), ("x", "y", "z"), "g2_pose")
        octant = ((vectors[:, 0] >= 0).astype(int) * 4 + (vectors[:, 1] >= 0).astype(int) * 2 + (vectors[:, 2] >= 0).astype(int))
        octant_fraction = np.bincount(octant, minlength=8) / len(vectors)
        collector.add_vector(f"pose_{role}_octant", octant_fraction, tuple(str(i) for i in range(8)), "g2_pose")
        eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(vectors, rowvar=False, bias=True)))[::-1].clip(min=0.0) if len(vectors) > 1 else np.zeros(3)
        eigenvalues /= eigenvalues.sum() if eigenvalues.sum() else 1.0
        collector.add_vector(f"pose_{role}_direction_covariance", eigenvalues, ("eig1", "eig2", "eig3"), "g2_pose")
        collector.add(f"pose_{role}_direction_concentration", np.linalg.norm(vectors.mean(axis=0)), "g2_pose")
    rotations = np.einsum("nji,njk->nik", target_frame[target_nodes], binder_frame[binder_nodes])
    labels = tuple(f"r{row + 1}{column + 1}" for row in range(3) for column in range(3))
    collector.add_vector("pose_rotation_mean", rotations.mean(axis=0).ravel(), labels, "g2_pose")
    collector.add_vector("pose_rotation_std", rotations.std(axis=0).ravel(), labels, "g2_pose")
    trace = np.trace(rotations, axis1=1, axis2=2)
    angle_values = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    histogram = np.histogram(angle_values, bins=ROTATION_ANGLE_BINS)[0] / len(angle_values)
    collector.add_vector("pose_rotation_angle_hist", histogram, tuple(f"bin{i}" for i in range(6)), "g2_pose")
    collector.add_vector("pose_rotation_angle", distribution_statistics(angle_values)[[1, 2, 3, 4, 5]], ("mean", "std", "q25", "q50", "q75"), "g2_pose")
    for axis in range(3):
        cosine = rotations[:, axis, axis]
        fractions = (np.mean(cosine > 0.5), np.mean(cosine < -0.5), np.mean(np.abs(cosine) <= 0.5))
        collector.add_vector(f"pose_axis{axis + 1}", fractions, ("parallel", "antiparallel", "perpendicular"), "g2_pose")


def _add_shape(
    collector: FeatureCollector,
    target_points: np.ndarray,
    binder_points: np.ndarray,
) -> None:
    labels = ("eig1_fraction", "eig2_fraction", "eig3_fraction", "radius_gyration", "anisotropy", "linearity", "planarity", "sphericity", "maximum_span", "normal_thickness", "tangential_radius")
    for role, points in (("target", target_points), ("binder", binder_points)):
        collector.add_vector(f"shape_{role}", shape_features(points), labels, "g3_shape")
    target_axes, target_values = principal_axes(target_points)
    binder_axes, binder_values = principal_axes(binder_points)
    alignment = np.abs(target_axes.T @ binder_axes)
    collector.add_vector("shape_axis_alignment", alignment.ravel(), tuple(f"t{i + 1}_b{j + 1}" for i in range(3) for j in range(3)), "g3_shape")
    target_radius = np.sqrt(target_values.sum())
    binder_radius = np.sqrt(binder_values.sum())
    centroid_distance = np.linalg.norm(target_points.mean(axis=0) - binder_points.mean(axis=0))
    collector.add("shape_binder_target_scale_ratio", safe_divide(binder_radius, target_radius), "g3_shape")
    collector.add("shape_centroid_distance", centroid_distance, "g3_shape")


def _add_identity(
    collector: FeatureCollector,
    target_sequence: str,
    binder_sequence: str,
    target_interface: np.ndarray,
    binder_interface: np.ndarray,
    target_sidechain: np.ndarray,
    binder_sidechain: np.ndarray,
    atom_target: np.ndarray,
    atom_binder: np.ndarray,
    atom_distance: np.ndarray,
) -> None:
    collector.add_vector("identity_target_aa", composition(target_sequence, target_interface), list(AA_ORDER), "g4_identity")
    collector.add_vector("identity_binder_aa", composition(binder_sequence, binder_interface), list(AA_ORDER), "g4_identity")
    elements = ("C", "N", "O", "S")
    collector.add_vector("identity_target_element", element_fractions(target_sequence, target_interface), elements, "g4_identity")
    collector.add_vector("identity_binder_element", element_fractions(binder_sequence, binder_interface), elements, "g4_identity")
    keep = atom_distance <= INTERFACE_CUTOFF
    target_sc = target_sidechain[atom_target[keep]]
    binder_sc = binder_sidechain[atom_binder[keep]]
    types = np.asarray([
        np.mean(~target_sc & ~binder_sc), np.mean(~target_sc & binder_sc),
        np.mean(target_sc & ~binder_sc), np.mean(target_sc & binder_sc),
    ]) if np.any(keep) else np.zeros(4)
    collector.add_vector("identity_atom_contact", types, ("bb_bb", "bb_sc", "sc_bb", "sc_sc"), "g4_identity")


def build_descriptor(
    *,
    target_sequence: str,
    binder_sequence: str,
    target: dict[str, np.ndarray],
    binder: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[FeatureDefinition]]:
    """Build one invariant descriptor; target and binder arrays are residue aligned."""
    if len(target_sequence) != len(target["ca"]) or len(binder_sequence) != len(binder["ca"]):
        raise ValueError("Sequence/coordinate length mismatch")
    target_cloud = atom_cloud(target["ca"], target["heavy_atom_local"], target["heavy_atom_mask"], target["heavy_atom_sidechain"])
    binder_cloud = atom_cloud(binder["ca"], binder["heavy_atom_local"], binder["heavy_atom_mask"], binder["heavy_atom_sidechain"])
    atom_target, atom_binder, atom_distance = cross_atom_contacts(target_cloud, binder_cloud)
    residue_target, residue_binder, residue_minimum = residue_contacts(
        target_cloud[1], binder_cloud[1], atom_target, atom_binder, atom_distance,
        len(binder_sequence), POSE_CONTACT_CUTOFF,
    )
    interface_keep = residue_minimum <= INTERFACE_CUTOFF
    target_interface = np.unique(residue_target[interface_keep])
    binder_interface = np.unique(residue_binder[interface_keep])
    if len(residue_target) == 0 or len(target_interface) == 0 or len(binder_interface) == 0:
        raise ValueError("Predicted structure has no cross-chain interface contacts")
    ca_matrix = cKDTree(target["ca"]).sparse_distance_matrix(cKDTree(binder["ca"]), 16.0, output_type="coo_matrix")
    collector = FeatureCollector()
    _add_size_and_topology(collector, len(target_sequence), len(binder_sequence), target_interface, binder_interface, residue_target[interface_keep], residue_binder[interface_keep], int(np.count_nonzero(atom_distance <= INTERFACE_CUTOFF)))
    _add_distances(collector, atom_distance, ca_matrix.data, residue_minimum, residue_target, residue_binder, target_interface, binder_interface)
    _add_pose(collector, target["n"], target["ca"], target["c"], binder["n"], binder["ca"], binder["c"], residue_target, residue_binder)
    _add_shape(collector, target["ca"][target_interface], binder["ca"][binder_interface])
    _add_identity(collector, target_sequence, binder_sequence, target_interface, binder_interface, target_cloud[2], binder_cloud[2], atom_target, atom_binder, atom_distance)
    output = np.asarray(collector.values, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Descriptor contains non-finite values")
    return output, collector.definitions


def feature_indices(definitions: list[FeatureDefinition], variant: str) -> np.ndarray:
    if variant == "dmasif_learned_surface":
        selected = np.asarray(
            [
                index
                for index, definition in enumerate(definitions)
                if definition.group in {"s2_matching", "s3_embedding", "s4_interaction"}
            ],
            dtype=np.int64,
        )
        if selected.size != 344:
            raise ValueError(
                "dMaSIF learned-surface descriptor has width "
                f"{selected.size}, expected 344"
            )
        return selected

    atomic_surface_variants = {
        "asp_pose_edges_zero": "p_uniform",
        "asp_pose_edges_uniform_clearance": "p_uniform",
        "asp_pose_edges_uniform_gap": "p_uniform",
        "asp_pose_edges_uniform_normal": "p_uniform",
        "asp_pose_edges_uniform_all": "p_uniform",
        "asp_pose_edges_uniform_shuffled": "p_uniform",
        "asp_pose_edges_vdw_clearance": "p_vdw",
        "asp_pose_edges_vdw_gap": "p_vdw",
        "asp_pose_edges_vdw_normal": "p_vdw",
        "asp_pose_edges_vdw_all": "p_vdw",
        "asp_pose_edges_vdw_shuffled": "p_vdw",
        "asp_pose_edges_capacity": "p_capacity",
        "asp_clearance_confirm_zero": "p_uniform",
        "asp_clearance_confirm_real": "p_uniform",
        "asp_clearance_confirm_shuffled": "p_uniform",
        "asp_clearance_confirm_capacity": "p_capacity",
    }
    if variant in atomic_surface_variants:
        groups = {"a0_pose_anchor", "a1_edge_density", atomic_surface_variants[variant]}
        selected = np.asarray(
            [
                index
                for index, definition in enumerate(definitions)
                if definition.group in groups
            ],
            dtype=np.int64,
        )
        if selected.size != 192:
            raise ValueError(
                f"Atomic-surface variant {variant} has width {selected.size}, expected 192"
            )
        return selected

    contact_axis_variants = {
        "cai_pose_zero": False,
        "cai_pose_separation": False,
        "cai_pose_footprint": False,
        "cai_pose_relief": False,
        "cai_pose_all": False,
        "cai_pose_all_shuffled": False,
        "cai_pose_capacity": False,
        "cai_pose_edges_zero": True,
        "cai_pose_edges_all": True,
        "cai_pose_edges_all_shuffled": True,
    }
    if variant in contact_axis_variants:
        groups = {"a0_pose_anchor"}
        if contact_axis_variants[variant]:
            groups.add("a1_edge_density")
        groups.add(
            "c3_capacity" if variant == "cai_pose_capacity" else "c0_separation"
        )
        if variant != "cai_pose_capacity":
            groups.update({"c1_footprint", "c2_relief"})
        selected = np.asarray(
            [
                index
                for index, definition in enumerate(definitions)
                if definition.group in groups
            ],
            dtype=np.int64,
        )
        expected = 210 if contact_axis_variants[variant] else 206
        if selected.size != expected:
            raise ValueError(
                f"Contact-axis variant {variant} has width {selected.size}, expected {expected}"
            )
        return selected

    occupancy_subsets = {
        "dcg_pose_occupancy_le_3p5a": ("cutoff", "le_3.5a", True),
        "dcg_pose_occupancy_le_5a": ("cutoff", "le_5a", True),
        "dcg_pose_occupancy_le_8a": ("cutoff", "le_8a", True),
        "dcg_pose_occupancy_le_12a": ("cutoff", "le_12a", True),
        "dcg_pose_occupancy_edges": ("edges", "", True),
        "dcg_pose_occupancy_active": ("active", "", True),
        "dcg_pose_occupancy_variable": ("variable", "", True),
        "dcg_pose_occupancy_variable_shuffled": ("variable", "", True),
        "dcg_pose_occupancy_shuffled": ("all", "", True),
        "dcg_occupancy_variable_only": ("variable", "", False),
    }
    occupancy_capacity_subsets = {
        "dcg_pose_occupancy_capacity_10": 10,
        "dcg_pose_occupancy_capacity_12": 12,
    }
    edge_confirmation_subsets = {
        "dcg_pose_edges_matched_real": "real",
        "dcg_pose_edges_matched_shuffled": "real",
        "dcg_pose_edges_matched_zero": "real",
        "dcg_pose_edges_matched_capacity": "capacity",
    }
    if variant in edge_confirmation_subsets:
        selected = [
            index
            for index, definition in enumerate(definitions)
            if definition.group in {"g0_topology", "g1_distance", "g2_pose"}
            and definition.subgroup != "size"
        ]
        if edge_confirmation_subsets[variant] == "capacity":
            additions = [
                index
                for index, definition in enumerate(definitions)
                if definition.group == "d3_capacity_noise"
            ][:4]
        else:
            additions = [
                index
                for index, definition in enumerate(definitions)
                if definition.group == "d2_spectral"
                and definition.name.endswith("edge_density")
            ]
        if len(selected) != 152 or len(additions) != 4:
            raise ValueError(f"Descriptor cache lacks matched edge features for {variant}")
        return np.asarray(selected + additions, dtype=np.int64)

    if variant in occupancy_subsets or variant in occupancy_capacity_subsets:
        selected = []
        include_pose_anchor = variant != "dcg_occupancy_variable_only"
        if include_pose_anchor:
            selected.extend(
                index
                for index, definition in enumerate(definitions)
                if definition.group in {"g0_topology", "g1_distance", "g2_pose"}
                and definition.subgroup != "size"
            )
        if variant in occupancy_capacity_subsets:
            noise_indices = [
                index
                for index, definition in enumerate(definitions)
                if definition.group == "d3_capacity_noise"
            ]
            selected.extend(noise_indices[:occupancy_capacity_subsets[variant]])
        else:
            mode, cutoff, _ = occupancy_subsets[variant]
            for index, definition in enumerate(definitions):
                if definition.group != "d2_spectral":
                    continue
                is_edge = definition.name.endswith("edge_density")
                is_active = definition.name.endswith(
                    ("active_target_fraction", "active_binder_fraction")
                )
                is_variable_active = is_active and definition.subgroup != "le_12a"
                include = (
                    mode == "cutoff"
                    and definition.subgroup == cutoff
                    and (is_edge or is_active)
                    or mode == "edges" and is_edge
                    or mode == "active" and is_variable_active
                    or mode == "variable" and (is_edge or is_variable_active)
                    or mode == "all" and (is_edge or is_active)
                )
                if include:
                    selected.append(index)
        if not selected:
            raise ValueError(f"Descriptor cache lacks occupancy features for {variant}")
        return np.asarray(selected, dtype=np.int64)

    spectral_subsets = {
        "dcg_spectral_only": "all",
        "dcg_pose_spectral_le_3p5a": "le_3.5a",
        "dcg_pose_spectral_le_5a": "le_5a",
        "dcg_pose_spectral_le_8a": "le_8a",
        "dcg_pose_spectral_le_12a": "le_12a",
        "dcg_pose_spectral_occupancy": "occupancy",
        "dcg_pose_spectral_degree": "degree",
        "dcg_pose_spectral_components": "components",
        "dcg_pose_spectral_spectrum": "spectrum",
        "dcg_pose_spectral_persistence": "persistence",
        "dcg_pose_spectral_normalized": "normalized",
        "dcg_pose_spectral_all_shuffled": "all",
        "dcg_pose_spectral_normalized_shuffled": "normalized",
    }
    capacity_subsets = {
        "dcg_pose_spectral_capacity_51": 51,
        "dcg_pose_spectral_capacity_39": 39,
    }
    if variant in spectral_subsets or variant in capacity_subsets:
        selected = []
        include_pose_anchor = variant != "dcg_spectral_only"
        if include_pose_anchor:
            selected.extend(
                index
                for index, definition in enumerate(definitions)
                if definition.group in {"g0_topology", "g1_distance", "g2_pose"}
                and definition.subgroup != "size"
            )
        if variant in capacity_subsets:
            noise_indices = [
                index
                for index, definition in enumerate(definitions)
                if definition.group == "d3_capacity_noise"
            ]
            width = capacity_subsets[variant]
            if len(noise_indices) < width:
                raise ValueError(f"Descriptor cache lacks capacity features for {variant}")
            selected.extend(noise_indices[:width])
        else:
            subset = spectral_subsets[variant]
            occupancy_suffixes = (
                "edge_density", "active_target_fraction", "active_binder_fraction",
            )
            degree_suffixes = ("target_degree_entropy", "binder_degree_entropy")
            component_suffixes = ("component_density", "largest_component_fraction")
            spectrum_tokens = (
                "singular_value_", "normalized_effective_rank", "spectral_entropy",
            )
            for index, definition in enumerate(definitions):
                if definition.group != "d2_spectral":
                    continue
                include = (
                    subset == "all"
                    or definition.subgroup == subset
                    or subset == "occupancy" and definition.name.endswith(occupancy_suffixes)
                    or subset == "degree" and definition.name.endswith(degree_suffixes)
                    or subset == "components" and definition.name.endswith(component_suffixes)
                    or subset == "spectrum" and any(
                        token in definition.name for token in spectrum_tokens
                    )
                    or subset == "persistence" and definition.subgroup == "persistence"
                    or subset == "normalized" and "singular_value_" not in definition.name
                )
                if include:
                    selected.append(index)
        if not selected:
            raise ValueError(f"Descriptor cache lacks spectral features for {variant}")
        return np.asarray(selected, dtype=np.int64)

    dcg_groups = {
        "dcg_distance_anchor": {"g0_topology", "g1_distance"},
        "dcg_pose_anchor": {"g0_topology", "g1_distance", "g2_pose"},
        "dcg_conditioned_from_distance": {
            "g0_topology", "g1_distance", "d1_conditioned_pose",
        },
        "dcg_pose_conditioned": {
            "g0_topology", "g1_distance", "g2_pose", "d1_conditioned_pose",
        },
        "dcg_pose_spectral": {
            "g0_topology", "g1_distance", "g2_pose", "d2_spectral",
        },
        "dcg_full": {
            "g0_topology", "g1_distance", "g2_pose", "d1_conditioned_pose",
            "d2_spectral",
        },
        "dcg_full_shuffled": {
            "g0_topology", "g1_distance", "g2_pose", "d1_conditioned_pose",
            "d2_spectral",
        },
        "dcg_capacity_noise": {
            "g0_topology", "g1_distance", "g2_pose", "d3_capacity_noise",
        },
    }
    if variant in dcg_groups:
        selected = np.asarray(
            [
                index for index, definition in enumerate(definitions)
                if definition.group in dcg_groups[variant] and definition.subgroup != "size"
            ],
            dtype=np.int64,
        )
        expected_new_group = dcg_groups[variant] - {"g0_topology", "g1_distance", "g2_pose"}
        observed_groups = {definitions[int(index)].group for index in selected}
        if not expected_new_group.issubset(observed_groups):
            raise ValueError(f"Descriptor cache lacks required DCG features for {variant}")
        return selected

    surface_groups = {
        "surface_distance": {"s0_distance"},
        "surface_distance_shape": {"s0_distance", "s1_local_shape"},
        "surface_distance_normals": {"s0_distance", "s2_normals"},
        "surface_distance_shape_normals": {
            "s0_distance", "s1_local_shape", "s2_normals",
        },
        "surface_full": {
            "s0_distance", "s1_local_shape", "s2_normals", "s3_complementarity",
        },
        "surface_full_normals_within_target_shuffled": {
            "s0_distance", "s1_local_shape", "s2_normals", "s3_complementarity",
        },
        "surface_full_within_target_shuffled": {
            "s0_distance", "s1_local_shape", "s2_normals", "s3_complementarity",
        },
    }
    if variant in surface_groups:
        selected = np.asarray(
            [
                index
                for index, definition in enumerate(definitions)
                if definition.group in surface_groups[variant]
            ],
            dtype=np.int64,
        )
        if selected.size == 0:
            raise ValueError(
                f"Descriptor cache does not contain surface-proxy features for {variant}"
            )
        return selected

    identity_prefixes = {
        "g1_target_aa_without_size": ("identity_target_aa_",),
        "g1_binder_aa_without_size": ("identity_binder_aa_",),
        "g1_aa_without_size": ("identity_target_aa_", "identity_binder_aa_"),
        "g1_elements_without_size": (
            "identity_target_element_", "identity_binder_element_",
        ),
        "g1_contact_classes_without_size": ("identity_atom_contact_",),
        "g1_binder_aa_contact_without_size": (
            "identity_binder_aa_", "identity_atom_contact_",
        ),
    }
    identity_subset = identity_prefixes.get(variant)
    if identity_subset is not None:
        selected = []
        for index, definition in enumerate(definitions):
            is_geometry = definition.group in {"g0_topology", "g1_distance"}
            is_selected_identity = any(
                definition.name.startswith(prefix) for prefix in identity_subset
            )
            if (is_geometry and definition.subgroup != "size") or is_selected_identity:
                selected.append(index)
        if not selected:
            raise ValueError(f"Descriptor cache does not contain features for {variant}")
        return np.asarray(selected, dtype=np.int64)

    groups = {
        "size_only": {"g0_topology"},
        "g0": {"g0_topology"},
        "g1": {"g0_topology", "g1_distance"},
        "g2": {"g0_topology", "g1_distance", "g2_pose"},
        "g3": {"g0_topology", "g1_distance", "g2_pose", "g3_shape"},
        "g4": {"g0_topology", "g1_distance", "g2_pose", "g3_shape", "g4_identity"},
        "g2_frame_shuffled": {"g0_topology", "g1_distance", "g2_pose"},
        "g4_identity_shuffled": {"g0_topology", "g1_distance", "g2_pose", "g3_shape", "g4_identity"},
        "geometry_without_size": {"g0_topology", "g1_distance", "g2_pose", "g3_shape"},
        "g1_without_size": {"g0_topology", "g1_distance"},
        "g2_without_size": {"g0_topology", "g1_distance", "g2_pose"},
        "g1_identity_without_size": {"g0_topology", "g1_distance", "g4_identity"},
        "g2_identity": {"g0_topology", "g1_distance", "g2_pose", "g4_identity"},
        "g2_identity_without_size": {"g0_topology", "g1_distance", "g2_pose", "g4_identity"},
        "g4_without_size": {"g0_topology", "g1_distance", "g2_pose", "g3_shape", "g4_identity"},
        "g2_identity_without_size_pose_shuffled": {"g0_topology", "g1_distance", "g2_pose", "g4_identity"},
        "g2_identity_without_size_identity_shuffled": {"g0_topology", "g1_distance", "g2_pose", "g4_identity"},
        "g1_identity_without_size_within_target_shuffled": {"g0_topology", "g1_distance", "g4_identity"},
    }
    if variant not in groups:
        raise ValueError(f"Unknown descriptor variant: {variant}")
    selected = []
    for index, definition in enumerate(definitions):
        if definition.group not in groups[variant]:
            continue
        if variant == "size_only" and definition.subgroup != "size":
            continue
        if (
            variant == "geometry_without_size" or "without_size" in variant
        ) and definition.subgroup == "size":
            continue
        selected.append(index)
    if not selected:
        raise ValueError(f"Descriptor cache does not contain features for {variant}")
    return np.asarray(selected, dtype=np.int64)


DESCRIPTOR_VARIANTS = (
    "dmasif_learned_surface",
    "size_only", "g0", "g1", "g2", "g3", "g4",
    "geometry_without_size", "g2_frame_shuffled", "g4_identity_shuffled",
    "g1_without_size", "g2_without_size", "g1_identity_without_size",
    "g2_identity", "g2_identity_without_size", "g4_without_size",
    "g2_identity_without_size_pose_shuffled",
    "g2_identity_without_size_identity_shuffled",
    "g1_target_aa_without_size", "g1_binder_aa_without_size",
    "g1_aa_without_size", "g1_elements_without_size",
    "g1_contact_classes_without_size", "g1_binder_aa_contact_without_size",
    "g1_identity_without_size_within_target_shuffled",
    "surface_distance", "surface_distance_shape", "surface_distance_normals",
    "surface_distance_shape_normals", "surface_full",
    "surface_full_normals_within_target_shuffled",
    "surface_full_within_target_shuffled",
    "dcg_distance_anchor", "dcg_pose_anchor", "dcg_conditioned_from_distance",
    "dcg_pose_conditioned", "dcg_pose_spectral", "dcg_full",
    "dcg_full_shuffled", "dcg_capacity_noise",
    "dcg_spectral_only", "dcg_pose_spectral_le_3p5a",
    "dcg_pose_spectral_le_5a", "dcg_pose_spectral_le_8a",
    "dcg_pose_spectral_le_12a", "dcg_pose_spectral_occupancy",
    "dcg_pose_spectral_degree", "dcg_pose_spectral_components",
    "dcg_pose_spectral_spectrum", "dcg_pose_spectral_persistence",
    "dcg_pose_spectral_normalized", "dcg_pose_spectral_all_shuffled",
    "dcg_pose_spectral_capacity_51",
    "dcg_pose_spectral_normalized_shuffled",
    "dcg_pose_spectral_capacity_39",
    "dcg_pose_occupancy_le_3p5a", "dcg_pose_occupancy_le_5a",
    "dcg_pose_occupancy_le_8a", "dcg_pose_occupancy_le_12a",
    "dcg_pose_occupancy_edges", "dcg_pose_occupancy_active",
    "dcg_pose_occupancy_variable", "dcg_pose_occupancy_variable_shuffled",
    "dcg_pose_occupancy_shuffled", "dcg_occupancy_variable_only",
    "dcg_pose_occupancy_capacity_10", "dcg_pose_occupancy_capacity_12",
    "dcg_pose_edges_matched_real", "dcg_pose_edges_matched_shuffled",
    "dcg_pose_edges_matched_zero", "dcg_pose_edges_matched_capacity",
    "cai_pose_zero", "cai_pose_separation", "cai_pose_footprint",
    "cai_pose_relief", "cai_pose_all", "cai_pose_all_shuffled",
    "cai_pose_capacity", "cai_pose_edges_zero", "cai_pose_edges_all",
    "cai_pose_edges_all_shuffled",
    "asp_pose_edges_zero", "asp_pose_edges_uniform_clearance",
    "asp_pose_edges_uniform_gap", "asp_pose_edges_uniform_normal",
    "asp_pose_edges_uniform_all", "asp_pose_edges_uniform_shuffled",
    "asp_pose_edges_vdw_clearance", "asp_pose_edges_vdw_gap",
    "asp_pose_edges_vdw_normal", "asp_pose_edges_vdw_all",
    "asp_pose_edges_vdw_shuffled", "asp_pose_edges_capacity",
    "asp_clearance_confirm_zero", "asp_clearance_confirm_real",
    "asp_clearance_confirm_shuffled", "asp_clearance_confirm_capacity",
)
