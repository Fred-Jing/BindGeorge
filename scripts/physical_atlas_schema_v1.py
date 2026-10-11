#!/usr/bin/env python3
"""Frozen feature schema for the Rosetta physical-interface atlas."""

from __future__ import annotations

from collections import OrderedDict


FEATURE_GROUPS = OrderedDict(
    [
        (
            "energy",
            (
                "total_score",
                "total_score_per_residue",
                "binder_score",
                "binder_score_per_residue",
                "target_score",
                "target_score_per_residue",
                "interface_dg",
                "interface_dg_dsasa_x100",
            ),
        ),
        (
            "shape_packing",
            (
                "binder_sasa",
                "target_sasa",
                "interface_sc",
                "interface_packstat",
                "interface_dsasa",
                "interface_fraction_of_binder_sasa",
                "interface_fraction_of_target_sasa",
                "interface_nres_binder",
                "interface_nres_target",
                "interface_dsasa_per_nres_total",
                "interface_dsasa_per_nres_binder",
            ),
        ),
        (
            "composition",
            (
                "binder_surface_hydrophobic_fraction",
                "interface_binder_hydrophobic_fraction",
                "interface_target_hydrophobic_fraction",
                "interface_binder_aromatic_fraction",
                "interface_target_aromatic_fraction",
                "interface_binder_charged_fraction",
                "interface_target_charged_fraction",
                "interface_binder_polar_fraction",
                "interface_target_polar_fraction",
            ),
        ),
        (
            "hbond_unsat",
            (
                "interface_hbonds",
                "interface_hbond_energy",
                "interface_hbonds_per_nres_binder",
                "interface_delta_unsat_hbonds",
                "interface_delta_unsat_per_nres_binder",
                "interface_delta_unsat_per_1000_dsasa",
            ),
        ),
        (
            "sap",
            (
                "sap_binder",
                "sap_target",
                "sap_complex",
                "sap_delta",
                "sap_delta_per_1000_dsasa",
            ),
        ),
    ]
)

FEATURE_NAMES = tuple(
    feature for features in FEATURE_GROUPS.values() for feature in features
)

LEGACY_FEATURES = (
    "total_score",
    "total_score_per_residue",
    "binder_score",
    "binder_sasa",
    "interface_sc",
    "interface_packstat",
    "interface_dg",
    "interface_dsasa",
    "interface_dg_dsasa_x100",
    "interface_hbonds",
    "interface_fraction_of_binder_sasa",
)

CONDITIONS = ("raw", "relax_1", "relax_2", "relax_3")
RESPONSE_BLOCKS = ("raw", "relax_mean", "delta_mean", "relax_std")


def validate_schema() -> None:
    if len(FEATURE_NAMES) != len(set(FEATURE_NAMES)):
        raise ValueError("Physical-atlas feature names are not unique")
    if not set(LEGACY_FEATURES).issubset(FEATURE_NAMES):
        raise ValueError("Legacy Rosetta fields are not contained in the atlas")


validate_schema()
