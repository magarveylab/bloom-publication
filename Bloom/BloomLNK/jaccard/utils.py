import json
from collections import Counter
from typing import List, Set

import networkx as nx


def jaccard_similarity(s1: set, s2: set):
    return round(len(s1 & s2) / len(s1 | s2), 2)


def get_kmers(graph: nx.Graph, nodes: Set[int], kmer_length: int):
    kmers = []
    subgraph = graph.subgraph(nodes)
    for n in nodes:
        paths = nx.single_source_shortest_path(
            G=subgraph, source=n, cutoff=kmer_length - 1
        )
        kmers.extend(paths.values())
    return kmers


def translate_kmers(kmers: List[tuple], label_lookup: dict):
    kmers = [tuple([label_lookup.get(i) for i in k]) for k in kmers]
    return [k for k in kmers if None not in k]


def get_signature_from_kmers(kmers: List[tuple]):
    counts = Counter(kmers)
    return {(n, i) for n, c in counts.items() for i in range(c)}


def get_containment_score(s1: set, s2: set):
    if len(s1) == 0:
        return 0
    # s1 is the active signature
    # s2 is the signature of the gene cluster
    overlap = s1 & s2
    return len(overlap) / len(s1)


def get_modular_bear_graph(bear_fp: str):
    bear_data = json.load(open(bear_fp, "r"))
    G = nx.Graph()
    for n in bear_data["nodes"]:
        if "aa" in n["enums"] or "pk" in n["enums"]:
            G.add_node(
                n["node_id"],
                label=n["top_label"],
                smarts_hash_id=n["meta"]["hash_id"],
            )
    if len(G.nodes) > 0:
        edges_to_add = [
            (e["n1"], e["n2"])
            for e in bear_data["edges"]
            if e["n1"] in G.nodes and e["n2"] in G.nodes
        ]
        if len(edges_to_add) > 0:
            G.add_edges_from(edges_to_add)
    return G


def get_nonmodular_bear_graph(bear_fp: str):
    bear_data = json.load(open(bear_fp, "r"))
    G = nx.Graph()
    for n in bear_data["nodes"]:
        # ignore if it is a modular component.
        if set(n["enums"]).intersection(set(["aa", "pk"])):
            continue
        # ignore unknown units for this analysis.
        elif n["meta"].get("hash_id") is None:
            continue
        G.add_node(
            n["node_id"],
            label=n["top_label"],
            smarts_hash_id=n["meta"]["hash_id"],
        )
    if len(G.nodes) > 0:
        edges_to_add = [
            (e["n1"], e["n2"])
            for e in bear_data["edges"]
            if e["n1"] in G.nodes and e["n2"] in G.nodes
        ]
        if len(edges_to_add) > 0:
            G.add_edges_from(edges_to_add)
    return G
