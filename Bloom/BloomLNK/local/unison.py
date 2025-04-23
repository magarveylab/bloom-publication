from typing import List

import networkx as nx
import pandas as pd

from Bloom.BloomLNK.local.common import (
    create_graph_from_dags,
    get_edge_subgraph,
    get_node_subgraph,
)
from Bloom.BloomLNK.utils import curdir

########################################################################
# Graph Schema
########################################################################

dag_filter_criteria = {
    "module_tag": ("module_tag", "ModuleTag"),
    "pks_reaction_tag": ("pks_reaction_tag", "PKSReactionTag"),
    "substrate": ("substrate", "Substrate"),
    "substrate_family_tag": ("substrate_family_tag", "SubstrateFamilyTag"),
    "sugar_rule_ec4": ("ec4", "EC4"),
    "tailoring_rule_ec3": ("ec3", "EC3"),
    "source_reaction_ec4": ("ec4", "EC4"),
    "source_reaction_gene": ("protein_family_tag", "ProteinFamilyTag"),
    "modifying_reaction_ec4": ("ec4", "EC4"),
    "modifying_reaction_gene": ("protein_family_tag", "ProteinFamilyTag"),
    # dags from local rendering
    "reaction_ec4": ("ec4", "EC4"),
    "reaction_gene": ("protein_family_tag", "ProteinFamilyTag"),
}

col_name_to_node_type = {
    "unit_region_id": "MetaboliteMolecularRegion",
    "module_tag": "ModuleTag",
    "pks_reaction_tag": "PKSReactionTag",
    "substrate": "Substrate",
    "substrate_family_tag": "SubstrateFamilyTag",
    "sugar_reaction_tag": "SugarReactionTag",
    "tailoring_reaction_tag": "TailoringReactionTag",
    "reaction_id": "Reaction",
    "rule_id": "UnitRule",
    "ec4": "EC4",
    "ec3": "EC3",
    "protein_family_tag": "ProteinFamilyTag",
}
col_name_to_node_properties = {
    "unit_region_id": ["unit"],
    "reaction_id": ["reaction_id"],
}
col_name_to_edge_type = {
    # unit to module connections
    ("unit_region_id", "module_tag"): "unit_to_module_tag",
    # unit to pks connections
    ("unit_region_id", "pks_reaction_tag"): "unit_to_pks_reaction_tag",
    # unit to substrate connections
    ("unit_region_id", "substrate"): "unit_to_substrate",
    (
        "unit_region_id",
        "substrate_family_tag",
    ): "unit_to_substrate_family_tag",
    # unit to rule connections
    ("unit_region_id", "rule_id"): "unit_to_rule",
    # unit to sugar connections
    ("rule_id", "sugar_reaction_tag"): "rule_to_sugar_reaction_tag",
    ("sugar_reaction_tag", "ec4"): "sugar_reaction_tag_to_ec4",
    # unit to tailoring connections
    ("rule_id", "tailoring_reaction_tag"): "rule_to_tailoring_reaction_tag",
    (
        "tailoring_reaction_tag",
        "ec3",
    ): "tailoring_reaction_tag_to_ec3",
    # unit to reaction connections
    ("rule_id", "reaction_id"): "rule_to_reaction",
    ("reaction_id", "ec4"): "reaction_to_ec4",
    ("reaction_id", "protein_family_tag"): "reaction_to_protein_family_tag",
}
col_name_to_edge_properties = {}

########################################################################
# Load requirements
########################################################################


def load_rule_requirements(fp: str):
    df = pd.read_csv(fp)
    return dict(zip(df["rule_id"], df["rxn_n"]))


def load_rxn_requirements(fp: str):
    df = pd.read_csv(fp)
    return dict(zip(df["reaction_id"], df["ec_n"] + df["pf_n"]))


metab_unit_rule_req = load_rule_requirements(
    f"{curdir}/preprocess/metabolism_unit_rule_requirements.csv"
)
sugar_unit_rule_req = load_rule_requirements(
    f"{curdir}/preprocess/sugar_unit_rule_requirements.csv"
)
rxn_req = load_rxn_requirements(
    f"{curdir}/preprocess/reaction_requirements.csv"
)


def remove_disconnected_rules(G: nx.Graph) -> nx.Graph:
    unit_nodes = set(n for n in G.nodes if n[0] == "MetaboliteMolecularRegion")
    subG = get_node_subgraph(
        G=G,
        node_types=[
            "MetaboliteMolecularRegion",
            "UnitRule",
            "SugarReactionTag",
            "TailoringReactionTag",
            "Reaction",
        ],
    )
    groups = nx.connected_components(subG)
    nodes_to_delete = set()
    for g in groups:
        if len(set(g) & unit_nodes) == 0:
            nodes_to_delete.update(g)
    G.remove_nodes_from(nodes_to_delete)
    return G


def clean_module_connections(G: nx.Graph) -> nx.Graph:
    # find units connected to modules
    subG_module = get_edge_subgraph(
        G=G,
        edge_types=[
            "orf_to_module",
            "module_to_tag",
            "module_to_domain",
            "module_to_substrate",
            "substrate_to_family_tag",
            "module_to_pks_reaction_tag",
            "unit_to_module_tag",
            "unit_to_substrate",
            "unit_to_substrate_family_tag",
            "unit_to_pks_reaction_tag",
        ],
    )
    groups = nx.connected_components(subG_module)
    units_connected_to_modules = set()
    for g in groups:
        g_units = [n for n in g if n[0] == "MetaboliteMolecularRegion"]
        g_modules = [n for n in g if n[0] == "Module"]
        if len(g_units) > 0 and len(g_modules) > 0:
            units_connected_to_modules.update(g_units)
    # remove unit to unit rule
    edges_to_delete = set()
    for n1, n2, e in G.edges(data=True):
        if e["edge_type"] == "unit_to_rule":
            if n1[0] == "MetaboliteMolecularRegion":
                unit_node, rule_node = n1, n2
            else:
                unit_node, rule_node = n2, n1
            if unit_node in units_connected_to_modules:
                edges_to_delete.add((n1, n2))
    G.remove_edges_from(edges_to_delete)
    G = remove_disconnected_rules(G)
    return G


def clean_sugar_connections(
    G: nx.Graph, min_sugar_rxn_freq: float
) -> nx.Graph:
    # determine if GTr is present
    has_GTr = (
        len([n for n in G.nodes if n[0] == "DomainLabel" and n[1] == "GTr"])
        > 0
    )
    # determine rules connected to sugar
    unit_rules = set(n for n in G.nodes if n[0] == "UnitRule")
    sugar_tags = set(n for n in G.nodes if n[0] == "SugarReactionTag")
    sugar_unit_rules = set(
        n for n in unit_rules if len(set(G.neighbors(n)) & sugar_tags) > 0
    )
    # remove all sugars if GTr is not present
    if has_GTr == False:
        to_delete = sugar_unit_rules
    # remove rules that do not meet sugar freq
    else:
        to_delete = set()
        for rule in sugar_unit_rules:
            rule_id = rule[1]
            connected_tags = set(G.neighbors(rule)) & sugar_tags
            score = len(connected_tags) / sugar_unit_rule_req[rule_id]
            if score < min_sugar_rxn_freq:
                to_delete.add(rule)
    # clean graph
    G.remove_nodes_from(to_delete)
    G = remove_disconnected_rules(G)
    return G


def get_isolated_breakdown_G(G):
    # multiple bear solutions might exist - isolate each solution and their
    # connections to the genomic content
    breakdown_graphs = {}
    breakdown_nodes = set(n for n in G.nodes if n[0] == "MetaboliteBreakdown")
    all_region_nodes = set(
        n for n in G.nodes if n[0] == "MetaboliteMolecularRegion"
    )
    for n in breakdown_nodes:
        connected_regions = set(G.neighbors(n)) & all_region_nodes
        to_skip = set()
        to_skip.update(all_region_nodes - connected_regions)
        to_skip.update(breakdown_nodes - {n})
        breakdown_graphs[n] = G.subgraph(G.nodes - to_skip)
    return breakdown_graphs


def count_connected_orfs(G):
    subG = get_edge_subgraph(
        G=G,
        edge_types=[
            "orf_to_ec4",
            "ec4_to_ec3",
            "orf_to_gene",
            "gene_to_protein_family_tag",
            "unit_to_rule",
            "rule_to_sugar_reaction_tag",
            "sugar_reaction_tag_to_ec4",
            "rule_to_tailoring_reaction_tag",
            "tailoring_reaction_tag_to_ec3",
            "rule_to_reaction",
            "reaction_to_ec4",
            "reaction_to_protein_family_tag",
        ],
    )
    all_orfs = set(n for n in G.nodes if n[0] == "Orf")
    all_units = set(n for n in G.nodes if n[0] == "MetaboliteMolecularRegion")
    connected_orfs = set()
    for g in nx.connected_components(subG):
        if len(g & all_units) > 0:
            connected_orfs.update(g & all_orfs)
    return len(connected_orfs)


def get_orf_to_unit_connections(G):
    units = set(n for n in G.nodes if n[0] == "MetaboliteMolecularRegion")
    orfs = set(n for n in G.nodes if n[0] == "Orf")
    paths = set()
    for o in orfs:
        for u in units:
            if nx.has_path(G, o, u):
                paths.add((o, u))
    return paths


def profile_orfs(G):
    paths = {}
    # find orfs connected to units by reaction
    subG = get_edge_subgraph(
        G=G,
        edge_types=[
            "unit_to_rule",
            "rule_to_reaction",
            "reaction_to_ec4",
            "reaction_to_protein_family_tag",
            "gene_to_protein_family_tag",
            "orf_to_ec4",
            "orf_to_gene",
        ],
    )
    paths["rxn"] = get_orf_to_unit_connections(subG)
    # find orfs connected to units by sugar
    subG = get_edge_subgraph(
        G=G,
        edge_types=[
            "unit_to_rule",
            "rule_to_sugar_reaction_tag",
            "sugar_reaction_tag_to_ec4",
            "orf_to_ec4",
        ],
    )
    paths["sugar"] = get_orf_to_unit_connections(subG)
    # find orfs connected to units by tailoring
    subG = get_edge_subgraph(
        G=G,
        edge_types=[
            "unit_to_rule",
            "rule_to_tailoring_reaction_tag",
            "tailoring_reaction_tag_to_ec3",
            "ec4_to_ec3",
            "orf_to_ec4",
        ],
    )
    paths["tailoring"] = get_orf_to_unit_connections(subG)
    # connected orfs
    orf_lookup = {}
    connected_orfs = set()
    for path_type, plist in paths.items():
        for o, u in plist:
            if o not in orf_lookup:
                orf_lookup[o] = {}
            if u not in orf_lookup[o]:
                orf_lookup[o][u] = {
                    "orf_id": o[-1],
                    "unit_id": u[-1],
                    "connected_by_rxn": False,
                    "connected_by_sugar": False,
                    "connected_by_tailoring": False,
                }
            orf_lookup[o][u][f"connected_by_{path_type}"] = True
    return [y for x in orf_lookup.values() for y in x.values()]


def count_connected_modules(G):
    subG = get_edge_subgraph(
        G=G,
        edge_types=[
            "module_to_tag",
            "module_to_substrate",
            "module_to_substrate_family_tag",
            "module_to_pks_reaction_tag",
            "unit_to_module_tag",
            "unit_to_pks_reaction_tag",
            "unit_to_substrate",
            "unit_to_substrate_family_tag",
        ],
    )
    all_units = set(n for n in G.nodes if n[0] == "MetaboliteMolecularRegion")
    all_modules = set(n for n in G.nodes if n[0] == "Module")
    connected_modules = set()
    for g in nx.connected_components(subG):
        if len(g & all_units) > 0:
            connected_modules.update(g & all_modules)
    return len(connected_modules)


def count_connected_modules_via_tags(G):
    subG = get_edge_subgraph(
        G=G,
        edge_types=[
            "module_to_tag",
            "unit_to_module_tag",
        ],
    )
    all_units = set(n for n in G.nodes if n[0] == "MetaboliteMolecularRegion")
    all_modules = set(n for n in G.nodes if n[0] == "Module")
    connected_modules = set()
    for g in nx.connected_components(subG):
        if len(g & all_units) > 0:
            connected_modules.update(g & all_modules)
    return len(connected_modules)


def clean_reaction_connections(
    G: nx.Graph, min_rxn_anno_freq: float, min_rule_rxn_freq: float
) -> nx.Graph:
    # remove reactions
    to_delete = set()
    all_anno_nodes = set(
        n for n in G.nodes if n[0] in ["EC4", "ProteinFamilyTag"]
    )
    for n in G.nodes:
        if n[0] == "Reaction":
            rxn_id = n[1]
            neighbors = set(G.neighbors(n))
            connected_anno = neighbors & all_anno_nodes
            score = len(connected_anno) / rxn_req[rxn_id]
            G.nodes[n]["properties"]["score"] = score
            if score < min_rxn_anno_freq:
                to_delete.add(n)
    G.remove_nodes_from(to_delete)
    # remove rules
    all_rxn_nodes = set(n for n in G.nodes if n[0] == "Reaction")
    to_delete = set()
    for n in G.nodes:
        if n[0] == "UnitRule":
            unit_id = n[1]
            if unit_id not in metab_unit_rule_req:
                continue
            neighbors = set(G.neighbors(n))
            connected_rxn = neighbors & all_rxn_nodes
            score = len(connected_rxn) / metab_unit_rule_req[unit_id]
            G.nodes[n]["properties"]["score"] = score
            if score < min_rule_rxn_freq:
                to_delete.add(n)
    G.remove_nodes_from(to_delete)
    G = remove_disconnected_rules(G)
    return G


def keep_best_breakdown(G: nx.Graph):
    # find breakdowns connected to metabolites
    breakdown_nodes = [n for n in G.nodes if n[0] == "MetaboliteBreakdown"]
    if len(breakdown_nodes) == 0:
        return G
    else:
        profiles = []
        breakdown_graphs = get_isolated_breakdown_G(G)
        for breakdown_node, subG in breakdown_graphs.items():
            connected_orfs = count_connected_orfs(subG)
            connected_modules = count_connected_modules(subG)
            connected_modules_via_tags = count_connected_modules_via_tags(subG)
            profiles.append(
                {
                    "breakdown_node": breakdown_node,
                    "breakdown_regions": set(
                        n
                        for n in subG.nodes
                        if n[0] == "MetaboliteMolecularRegion"
                    ),
                    "connected_orfs": connected_orfs,
                    "connected_modules": connected_modules,
                    "connected_modules_via_tags": connected_modules_via_tags,
                }
            )
        # find best breakdown
        profiles = sorted(
            profiles,
            key=lambda x: (
                x["connected_modules"],
                x["connected_modules_via_tags"],
                x["connected_orfs"],
                x["breakdown_node"],
            ),
            reverse=True,
        )
        # clean graph
        to_remove = set()
        for n in profiles[1:]:
            to_remove.add(n["breakdown_node"])
            to_remove.update(n["breakdown_regions"])
        G.remove_nodes_from(to_remove)
        G = remove_disconnected_rules(G)
        return G


def prepare_graph_for_training(G: nx.Graph):
    bgc_nodes = set(n for n in G.nodes if n[0] == "BGC")
    metabolite_nodes = set(n for n in G.nodes if n[0] == "Metabolite")
    breakdown_nodes = set(n for n in G.nodes if n[0] == "MetaboliteBreakdown")
    pair_id = "{}-{}".format(
        list(bgc_nodes)[0][1], list(metabolite_nodes)[0][1]
    )
    to_remove = set()
    for g in nx.connected_components(G):
        if len(g & bgc_nodes) == 0 and len(g & metabolite_nodes) == 0:
            to_remove.update(g)
    G.remove_nodes_from(to_remove)
    G.remove_nodes_from(bgc_nodes | metabolite_nodes | breakdown_nodes)
    node_type = "Pair"
    node_id = (node_type, pair_id)
    G.add_node(node_id, node_type=node_type, properties={})
    return G


def unite_graphs(
    mol_G: nx.Graph,
    bgc_G: nx.Graph,
    mol_dags: List[dict],
    min_sugar_rxn_freq: float = 0,
    min_rxn_anno_freq: float = 0,
    min_rule_rxn_freq: float = 0,
) -> nx.Graph:
    # trace nodes
    bgc_node_lookup = {
        "ModuleTag": set(),
        "PKSReactionTag": set(),
        "Substrate": set(),
        "SubstrateFamilyTag": set(),
        "EC4": set(),
        "EC3": set(),
        "ProteinFamilyTag": set(),
    }
    for node in bgc_G.nodes:
        node_type, node_id = node
        if node_type not in bgc_node_lookup:
            continue
        bgc_node_lookup[node_type].add(node_id)
    # filter dags
    filtered_mol_dags = []
    for dag in mol_dags:
        dag_type = dag["dag"]
        dag_key, node_type = dag_filter_criteria[dag_type]
        if dag[dag_key] in bgc_node_lookup[node_type]:
            filtered_mol_dags.append(dag)
    # create dags
    G_connect = create_graph_from_dags(
        dags=filtered_mol_dags,
        col_name_to_node_type=col_name_to_node_type,
        col_name_to_node_properties=col_name_to_node_properties,
        col_name_to_edge_type=col_name_to_edge_type,
        col_name_to_edge_properties=col_name_to_edge_properties,
    )
    # unify graphs
    unified_G = nx.compose(bgc_G, G_connect)
    unified_G = nx.compose(unified_G, mol_G)
    # clean graph
    unified_G = clean_module_connections(unified_G)
    unified_G = clean_sugar_connections(
        unified_G, min_sugar_rxn_freq=min_sugar_rxn_freq
    )
    unified_G = clean_reaction_connections(
        unified_G,
        min_rxn_anno_freq=min_rxn_anno_freq,
        min_rule_rxn_freq=min_rule_rxn_freq,
    )
    unified_G = keep_best_breakdown(unified_G)
    unified_G = prepare_graph_for_training(unified_G)
    return unified_G
