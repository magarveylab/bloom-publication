import json
from collections import Counter
from typing import Set

import networkx as nx

from BearLinker.jaccard.utils import (
    get_kmers,
    get_modular_bear_graph,
    get_signature_from_kmers,
    translate_kmers,
)


def get_unit_signature(G: nx.Graph, label_lookup: dict, node_ids: Set[int]):
    kmers = get_kmers(graph=G, nodes=node_ids, kmer_length=3)
    kmers = translate_kmers(kmers=kmers, label_lookup=label_lookup)
    return get_signature_from_kmers(kmers=kmers)


def get_active_units(
    base_value: float,
    shap_values: list,
    nodes_to_consider: list,
    min_activity: float = 0.9,
):
    filtered_shape_values = sorted(
        [n for n in shap_values if n["unit_id"] in nodes_to_consider],
        key=lambda x: x["shap"],
        reverse=True,
    )
    current_activity = base_value
    active_units = []
    for s in filtered_shape_values:
        current_activity += s["shap"]
        active_units.append(s["unit_id"])
        if current_activity >= min_activity:
            break
    return active_units, round(current_activity, 3)


def get_active_signature(
    bear_fp: str,
    base_value: float,
    shap_values: list,
    min_activity: float = 0.9,
):
    # define bear graph with only known units
    G = get_modular_bear_graph(bear_fp=bear_fp)
    active_units, current_activity = get_active_units(
        base_value=base_value,
        shap_values=shap_values,
        nodes_to_consider=set(G.nodes),
        min_activity=min_activity,
    )
    label_lookup = {n: G.nodes[n]["label"] for n in active_units}
    signature = get_unit_signature(
        G=G,
        label_lookup=label_lookup,
        node_ids=set(active_units),
    )
    return {
        "signature": signature,
        "activity": current_activity,
        "active_units": active_units,
        "total_units": len(G.nodes),
    }


def get_mol_signature(bear_fp: str):
    # define bear graph with only known units
    G = get_modular_bear_graph(bear_fp)
    node_ids = set(G.nodes)
    label_lookup = {n: G.nodes[n]["label"] for n in node_ids}
    return get_unit_signature(
        G=G,
        label_lookup=label_lookup,
        node_ids=node_ids,
    )
