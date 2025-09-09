import networkx as nx

from Bloom.BloomLNK.jaccard.jaccard_signature import (
    build_module_graph_from_module_lookup,
    parse_ibis_dir_modular,
)
from Bloom.BloomLNK.jaccard.library import (
    module_to_pks_reaction,
    smarts_to_unit,
    unit_to_module_tag,
    unit_to_pks_reaction_tag,
)
from Bloom.BloomLNK.jaccard.utils import (
    get_kmers,
    get_modular_bear_graph,
    get_signature_from_kmers,
    jaccard_similarity,
    translate_kmers,
)


def get_bfs_kmers(G, target_node):
    all_kmers = []
    for depth in range(3):
        bfs = nx.bfs_tree(G, source=target_node, depth_limit=depth)
        kmers = get_kmers(graph=G, nodes=set(bfs.nodes()), kmer_length=3)
        all_kmers.extend(kmers)
    return all_kmers


def get_module_fingerprint_from_bgcs(ibis_dir: str, cluster_id: str) -> dict:
    # parse ibis results extracting all module annotations
    bgc_data = parse_ibis_dir_modular(
        ibis_dir=ibis_dir, cluster_ids=[cluster_id]
    )
    # for each module, pull out kmers at different graph traversal breadths
    module_fingerprint_lookup = {}
    for bgc in bgc_data:
        module_paths = bgc["module_paths"]
        for path in module_paths:
            module_lookup = {m["module_idx"]: m["tag"] for m in path}
            G = build_module_graph_from_module_lookup(module_lookup)
            for m in path:
                module_idx = m["module_idx"]
                module_id = m["module_id"]
                module_tag = m["tag"]
                kmers = get_bfs_kmers(G, module_idx)
                module_kmers = translate_kmers(
                    kmers=kmers, label_lookup=module_lookup
                )
                pks_rxn_kmers = translate_kmers(
                    kmers=module_kmers, label_lookup=module_to_pks_reaction
                )
                signature = get_signature_from_kmers(
                    module_kmers + pks_rxn_kmers
                )
                module_fingerprint_lookup[module_id] = {
                    "module_tag": module_tag,
                    "signature": signature,
                }
    return module_fingerprint_lookup


def get_module_fingerprint_from_molecules(bear_fp: str):
    # parse molecular breakdown to build graph
    G = get_modular_bear_graph(bear_fp=bear_fp)
    if len(G.nodes) == 0:
        return {}
    # filter nodes to only those with smarts that map to a module unit
    keep_nodes = []
    module_lookup = {}
    pks_rxn_lookup = {}
    for n in G.nodes:
        smarts_hash_id = G.nodes[n]["smarts_hash_id"]
        if smarts_hash_id not in smarts_to_unit:
            continue
        unit_id = smarts_to_unit[smarts_hash_id]
        if unit_id not in unit_to_module_tag:
            continue
        module_tag = unit_to_module_tag[unit_id]
        module_lookup[n] = module_tag
        if unit_id in unit_to_pks_reaction_tag:
            pks_rxn_lookup[n] = unit_to_pks_reaction_tag[unit_id]
        keep_nodes.append(n)
    G = G.subgraph(keep_nodes)
    # create module fingerprint lookup
    module_fingerprint_lookup = {}
    for n in keep_nodes:
        kmers = get_bfs_kmers(G, n)
        module_kmers = translate_kmers(kmers=kmers, label_lookup=module_lookup)
        pks_rxn_kmers = translate_kmers(
            kmers=module_kmers, label_lookup=pks_rxn_lookup
        )
        signature = get_signature_from_kmers(module_kmers + pks_rxn_kmers)
        module_fingerprint_lookup[n] = {
            "module_tag": module_lookup[n],
            "signature": signature,
        }
    return module_fingerprint_lookup


def bgc_molecule_module_alignment(
    bear_fp: str, ibis_dir: str, cluster_id: str
):
    bgc_module_fingerprints = get_module_fingerprint_from_bgcs(
        ibis_dir=ibis_dir, cluster_id=cluster_id
    )
    mol_module_fingerprints = get_module_fingerprint_from_molecules(
        bear_fp=bear_fp
    )
    # sort units by tag
    tag_to_units = {}
    for unit_id, profile in mol_module_fingerprints.items():
        tag = profile["module_tag"]
        if tag not in tag_to_units:
            tag_to_units[tag] = set()
        tag_to_units[tag].add(unit_id)
    # sort units by modules
    tag_to_modules = {}
    for module_id, profile in bgc_module_fingerprints.items():
        tag = profile["module_tag"]
        if tag not in tag_to_modules:
            tag_to_modules[tag] = set()
        tag_to_modules[tag].add(module_id)
    # align modules to units
    overlapping_tags = set(tag_to_units) & set(tag_to_modules)
    hits = []
    for tag in overlapping_tags:
        for unit_id in tag_to_units[tag]:
            for module_id in tag_to_modules[tag]:
                mol_signature = mol_module_fingerprints[unit_id]["signature"]
                bgc_signature = bgc_module_fingerprints[module_id]["signature"]
                score = jaccard_similarity(mol_signature, bgc_signature)
                hits.append((unit_id, module_id, tag, score))
    # sort hits by score
    hits = sorted(hits, key=lambda x: x[-1], reverse=True)
    # do not allow multiple hits to the same unit or module
    used_units = set()
    used_modules = set()
    final_hits = []
    for unit_id, module_id, tag, score in hits:
        if unit_id in used_units or module_id in used_modules:
            continue
        used_units.add(unit_id)
        used_modules.add(module_id)
        final_hits.append(
            {
                "unit_id": unit_id,
                "module_id": module_id,
                "module_tag": tag,
                "score": score,
            }
        )
    return final_hits
