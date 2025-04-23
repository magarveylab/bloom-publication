import json
import pickle
from typing import List, Literal, Set

import networkx as nx

from Bloom.BloomLNK.jaccard.library import (
    ec3_to_tailoring_unit,
    ec4_to_reaction,
    ec4_to_sugar_tag,
    gene_to_protein_family_tag,
    loose_pks_unit_ids,
    loose_sugar_unit_ids,
    module_to_pks_reaction,
    module_to_substrate,
    module_to_substrate_family,
    reaction_to_ec4,
    reaction_to_protein_family_tag,
    rule_to_reaction,
    rule_to_unit,
    smarts_to_unit,
    unit_id_to_unit,
    unit_to_module_tag,
    unit_to_pks_reaction_tag,
    unit_to_substrate,
    unit_to_substrate_family,
    unit_to_sugar_tag,
)
from Bloom.BloomLNK.jaccard.utils import (
    get_kmers,
    get_modular_bear_graph,
    get_nonmodular_bear_graph,
    get_signature_from_kmers,
    translate_kmers,
)


def get_final_signature_from_molecule(
    bear_fp: str,
    modular_keys_to_use: List[
        Literal[
            "module_signature",
            "substrate_signature",
            "substrate_family_signature",
            "pks_rxn_signature",
        ]
    ] = ["module_signature", "pks_rxn_signature"],
    nodes_to_consider: Set[int] = None,
):
    module_signatures = get_module_signatures_from_molecule(
        bear_fp=bear_fp, nodes=nodes_to_consider
    )
    nonmod_signatures = get_nonmodular_signature_from_molecule(
        bear_fp=bear_fp, nodes=nodes_to_consider
    )
    final_signature = set()
    for k in modular_keys_to_use:
        final_signature.update(module_signatures[k])
    final_signature.update(nonmod_signatures["nonmodular_signature"])
    return final_signature


def get_final_signatures_from_genome(
    ibis_dir: str,
    cluster_ids: List[str] = None,
    modular_keys_to_use: List[
        Literal[
            "module_signature",
            "substrate_signature",
            "substrate_family_signature",
            "pks_rxn_signature",
        ]
    ] = ["module_signature", "pks_rxn_signature"],
    nonmod_clust_min_size: int = 3,
):
    # load module proteins
    module_proteins = set(
        i["protein_id"]
        for i in json.load(open(f"{ibis_dir}/module_predictions.json"))
        if len(i["modules"]) > 0
    )
    # load orf to protein ids
    orf_to_protein_id = {}
    for p in json.load(open(f"{ibis_dir}/prodigal.json")):
        protein_id = p["protein_id"]
        contig_id = p["contig_id"]
        contig_start = p["contig_start"]
        contig_stop = p["contig_stop"]
        orf_id = f"{contig_id}_{contig_start}_{contig_stop}"
        orf_to_protein_id[orf_id] = protein_id
    # load bgcs to analyze
    modular_clusters = set()
    non_modular_clusters = set()
    bgc_lkp = {}
    for b in json.load(open(f"{ibis_dir}/bgc_predictions.json")):
        contig_id = b["contig_id"]
        contig_start = b["contig_start"]
        contig_stop = b["contig_stop"]
        chemotypes = set(b["internal_chemotypes"])
        cluster_id = f"{contig_id}_{contig_start}_{contig_stop}"
        if cluster_ids is not None and cluster_id not in cluster_ids:
            continue
        protein_ids = set(
            orf_to_protein_id[orf_id]
            for orf_id in b["orfs"]
            if orf_id in orf_to_protein_id
        )
        if len(protein_ids & module_proteins) > 0:
            modular_clusters.add(cluster_id)
        elif len(chemotypes) == 0:
            continue
        elif {"RiPP", "Bacteriocin"} & chemotypes:
            continue
        elif (
            len(b["orfs"]) >= nonmod_clust_min_size
            and len(
                set(b["internal_chemotypes"]).intersection(
                    {"NonRibosomalPeptide", "TypeIPolyketide", "Hybrid"}
                )
            )
            == 0
        ):
            non_modular_clusters.add(cluster_id)
        bgc_lkp[cluster_id] = b
    out = {}
    # calculate module signatures
    if len(modular_clusters) > 0:
        mod_out = get_module_signatures_from_bgcs(
            ibis_dir=ibis_dir, cluster_ids=list(modular_clusters)
        )
    else:
        mod_out = {}
    if len(non_modular_clusters | modular_clusters) > 0:
        nonmod_out = get_nonmodular_signatures_from_bgcs(
            ibis_dir=ibis_dir,
            cluster_ids=list(non_modular_clusters | modular_clusters),
        )
    else:
        nonmod_out = {}
    for clust_id in modular_clusters.union(non_modular_clusters):
        module_signatures = mod_out.get(clust_id)
        nonmod_signatures = nonmod_out[clust_id]
        final_signature = set()
        if module_signatures is not None:
            for mod_k in modular_keys_to_use:
                final_signature.update(module_signatures[mod_k])
        final_signature.update(nonmod_signatures["nonmodular_signature"])
        out[clust_id] = final_signature
    return out


def parse_ibis_dir_modular(
    ibis_dir: str,
    cluster_ids: List[str] = None,
):
    # load orf to proteins
    orf_to_proteins = {}
    for p in json.load(open(f"{ibis_dir}/prodigal.json")):
        contig_id = p["contig_id"]
        contig_start = p["contig_start"]
        contig_stop = p["contig_stop"]
        protein_id = p["protein_id"]
        orf_id = f"{contig_id}_{contig_start}_{contig_stop}"
        orf_to_proteins[orf_id] = protein_id
    # load protein to module tags (sequential order)
    protein_to_module_tags = {}
    for p in json.load(open(f"{ibis_dir}/module_predictions.json")):
        protein_id = p["protein_id"]
        if len(p["modules"]) == 0:
            continue
        module_tags = [m["tags"][0]["tag"] for m in p["modules"]]
        protein_to_module_tags[protein_id] = module_tags
    bgc_dat = json.load(open(f"{ibis_dir}/bgc_predictions.json"))
    if cluster_ids is None:
        cluster_ids = [
            f"{x['contig_id']}_{x['contig_start']}_{x['contig_stop']}"
            for x in bgc_dat
        ]
    # reorganize data to cluster level
    reorganized = {
        cluster_id: {
            "cluster_id": cluster_id,
            "module_paths": [],
        }
        for cluster_id in cluster_ids
    }
    for c in bgc_dat:
        contig_id = c["contig_id"]
        contig_start = c["contig_start"]
        contig_stop = c["contig_stop"]
        cluster_id = f"{contig_id}_{contig_start}_{contig_stop}"
        orfs = c["orfs"]
        if cluster_id not in reorganized:
            continue
        for orf in orfs:
            if orf not in orf_to_proteins:
                continue
            protein_id = orf_to_proteins[orf]
            if protein_id in protein_to_module_tags:
                reorganized[cluster_id]["module_paths"].append(
                    protein_to_module_tags[protein_id]
                )
    return list(reorganized.values())


def parse_ibis_dir_nonmodular(
    ibis_dir: str,
    cluster_ids: List[str] = None,
    ec_homol_score: float = 0.6,
    gene_similarity_score: float = 0.09,
):
    # load orf to proteins
    orf_to_proteins = {}
    for p in json.load(open(f"{ibis_dir}/prodigal.json")):
        contig_id = p["contig_id"]
        contig_start = p["contig_start"]
        contig_stop = p["contig_stop"]
        protein_id = p["protein_id"]
        orf_id = f"{contig_id}_{contig_start}_{contig_stop}"
        orf_to_proteins[orf_id] = protein_id
    prot_to_enz = {}
    for p in pickle.load(open(f"{ibis_dir}/protein_embedding.pkl", "rb")):
        if p["ec1"] != "EC:-":
            prot_to_enz[p["protein_id"]] = True
        else:
            prot_to_enz[p["protein_id"]] = False
    protein_to_ec4 = {}
    for p in json.load(open(f"{ibis_dir}/ec_predictions.json")):
        protein_id = p["query_id"]
        if prot_to_enz[protein_id]:
            if len(p["predictions"]) == 0:
                continue
            ec4 = p["predictions"][0]["label"]
            ec_hs = p["predictions"][0]["homology"]
            if ec_hs >= ec_homol_score:
                protein_to_ec4[protein_id] = ec4
    protein_to_gene = {}
    for p in json.load(open(f"{ibis_dir}/gene_predictions.json")):
        protein_id = p["query_id"]
        gene_preds = p["predictions"]
        if len(gene_preds) == 0:
            continue
        gene = p["predictions"][0]["label"]
        gene_sim = p["predictions"][0]["similarity"]
        if gene_sim >= gene_similarity_score:
            protein_to_gene[protein_id] = gene
    prot_to_gtr = {}
    for p in json.load(open(f"{ibis_dir}/domain_predictions.json")):
        if len(p["regions"]) == 1 and p["regions"][0]["label"] == "GTr":
            protein_id = p["protein_id"]
            prot_to_gtr[protein_id] = "GTr"
    # load bgc data
    bgc_dat = json.load(open(f"{ibis_dir}/bgc_predictions.json"))
    if cluster_ids is None:
        cluster_ids = [
            f"{x['contig_id']}_{x['contig_start']}_{x['contig_stop']}"
            for x in bgc_dat
        ]
    # reorganize data to cluster level
    reorganized = {
        cluster_id: {
            "cluster_id": cluster_id,
            "ec4s": [],
            "genes": [],
            "gtrs": [],
        }
        for cluster_id in cluster_ids
    }
    for c in bgc_dat:
        contig_id = c["contig_id"]
        contig_start = c["contig_start"]
        contig_stop = c["contig_stop"]
        cluster_id = f"{contig_id}_{contig_start}_{contig_stop}"
        orfs = c["orfs"]
        if cluster_id not in reorganized:
            continue
        for orf in orfs:
            if orf not in orf_to_proteins:
                continue
            protein_id = orf_to_proteins[orf]
            if protein_id in protein_to_ec4:
                reorganized[cluster_id]["ec4s"].append(
                    protein_to_ec4[protein_id]
                )
            if protein_id in protein_to_gene:
                reorganized[cluster_id]["genes"].append(
                    protein_to_gene[protein_id]
                )
            if protein_id in prot_to_gtr:
                reorganized[cluster_id]["gtrs"].append(prot_to_gtr[protein_id])
    return list(reorganized.values())


def get_module_signatures_from_module_paths(module_paths):
    module_kmers = []
    for path in module_paths:
        path_len = len(path)
        module_lookup = {
            idx: module_tag for idx, module_tag in enumerate(path)
        }
        G = nx.Graph()
        G.add_nodes_from(module_lookup.keys())
        G.add_edges_from(list(zip(range(path_len), range(1, path_len + 1))))
        kmers = get_kmers(graph=G, nodes=set(module_lookup), kmer_length=3)
        module_kmers.extend(
            translate_kmers(kmers=kmers, label_lookup=module_lookup)
        )
    substrate_kmers = translate_kmers(
        kmers=module_kmers, label_lookup=module_to_substrate
    )
    substrate_family_kmers = translate_kmers(
        kmers=module_kmers, label_lookup=module_to_substrate_family
    )
    pks_rxn_kmers = translate_kmers(
        kmers=module_kmers, label_lookup=module_to_pks_reaction
    )
    # create signature
    module_signature = get_signature_from_kmers(kmers=module_kmers)
    substrate_signature = get_signature_from_kmers(kmers=substrate_kmers)
    substrate_family_signature = get_signature_from_kmers(
        kmers=substrate_family_kmers
    )
    pks_rxn_signature = get_signature_from_kmers(kmers=pks_rxn_kmers)
    return {
        "module_signature": module_signature,
        "substrate_signature": substrate_signature,
        "substrate_family_signature": substrate_family_signature,
        "pks_rxn_signature": pks_rxn_signature,
    }


def get_module_signatures_from_bgcs(ibis_dir: str, cluster_ids: List[str]):
    bgc_data = parse_ibis_dir_modular(
        ibis_dir=ibis_dir, cluster_ids=cluster_ids
    )
    out = {}
    for bgc in bgc_data:
        cluster_id = bgc["cluster_id"]
        module_signatures = get_module_signatures_from_module_paths(
            module_paths=bgc["module_paths"]
        )
        out[cluster_id] = {"cluster_id": cluster_id, **module_signatures}
    return out


def get_nonmodular_signatures_from_bgcs(ibis_dir: str, cluster_ids: List[str]):
    bgc_data = parse_ibis_dir_nonmodular(
        ibis_dir=ibis_dir, cluster_ids=cluster_ids
    )
    out = {}
    for bgc_dat in bgc_data:
        cluster_id = bgc_dat["cluster_id"]
        signatures = get_nonmodular_signatures_from_bgc_dat(bgc_dat=bgc_dat)
        out[cluster_id] = {"cluster_id": cluster_id, **signatures}
    return out


def get_module_signatures_from_molecule(bear_fp: str, nodes: List[str] = None):
    G = get_modular_bear_graph(bear_fp=bear_fp)
    if nodes != None:
        target_nodes = set(G.nodes) & set(nodes)
    else:
        target_nodes = set(G.nodes)
    kmers = get_kmers(graph=G, nodes=target_nodes, kmer_length=3)
    module_lookup = {}
    substrate_lookup = {}
    substrate_family_lookup = {}
    pks_rxn_lookup = {}
    for n in target_nodes:
        smarts_hash_id = G.nodes[n]["smarts_hash_id"]
        unit_id = smarts_to_unit[smarts_hash_id]
        if unit_id in unit_to_module_tag:
            module_lookup[n] = unit_to_module_tag[unit_id]
        if unit_id in unit_to_substrate:
            substrate_lookup[n] = unit_to_substrate[unit_id]
        if unit_id in unit_to_substrate_family:
            substrate_family_lookup[n] = unit_to_substrate_family[unit_id]
        if unit_id in unit_to_pks_reaction_tag:
            pks_rxn_lookup[n] = unit_to_pks_reaction_tag[unit_id]
    module_kmers = translate_kmers(kmers=kmers, label_lookup=module_lookup)
    substrate_kmers = translate_kmers(
        kmers=kmers, label_lookup=substrate_lookup
    )
    substrate_family_kmers = translate_kmers(
        kmers=kmers, label_lookup=substrate_family_lookup
    )
    pks_rxn_kmers = translate_kmers(kmers=kmers, label_lookup=pks_rxn_lookup)
    # create signature
    module_signature = get_signature_from_kmers(kmers=module_kmers)
    substrate_signature = get_signature_from_kmers(kmers=substrate_kmers)
    substrate_family_signature = get_signature_from_kmers(
        kmers=substrate_family_kmers
    )
    pks_rxn_signature = get_signature_from_kmers(kmers=pks_rxn_kmers)
    return {
        "module_signature": module_signature,
        "substrate_signature": substrate_signature,
        "substrate_family_signature": substrate_family_signature,
        "pks_rxn_signature": pks_rxn_signature,
    }


def get_nonmodular_signatures_from_bgc_dat(
    bgc_dat,
    min_sugar_rxn_freq: float = 0.7,
    min_rxn_anno_freq: float = 1.0,
    min_rule_rxn_freq: float = 0.7,
    apply_sugar_modular_cleanup: Literal["strict", "relaxed"] = "strict",
):
    unit_ids = set()
    # translate genes to tags
    prot_fam_tags = set()
    for gene in bgc_dat["genes"]:
        if gene in gene_to_protein_family_tag:
            prot_fam_tags.add(gene_to_protein_family_tag[gene])
    rxn_shortlist = set()
    sugar_tags = set()
    num_gtrs = len(bgc_dat["gtrs"])
    has_gtr = num_gtrs > 0
    ec4s = set(bgc_dat["ec4s"])
    for ec4 in bgc_dat["ec4s"]:
        # reduce the reaction search space for finalization
        if ec4 in ec4_to_reaction:
            rxn_shortlist.update(ec4_to_reaction[ec4])
        # translate reactions to sugar tags if GTr is present
        if ec4 in ec4_to_sugar_tag and has_gtr:
            sugar_tags.add("GTr")  # explicitly add GTr if already
            # found in domains
            sugar_tags_ = ec4_to_sugar_tag[ec4]
            # update number of GTrs
            if "GTr" in sugar_tags_:
                num_gtrs += 1
            sugar_tags.update(sugar_tags_)
        ec3 = ".".join(ec4.split(".")[:3])
        if ec3 in ec3_to_tailoring_unit:
            unit_ids.update(ec3_to_tailoring_unit[ec3])
    # determine final reaction complement
    rxns = set()
    for rxn in rxn_shortlist:
        ec_reqs = reaction_to_ec4.get(rxn, set())
        pf_reqs = reaction_to_protein_family_tag.get(rxn, set())
        rxn_annot_ratio = (
            len(ec_reqs & ec4s) + len(pf_reqs & prot_fam_tags)
        ) / (len(pf_reqs) + len(ec_reqs))
        if rxn_annot_ratio >= min_rxn_anno_freq:
            rxns.add(rxn)
    # add units based on final rxn complement
    for rule_id, req_rxns in rule_to_reaction.items():
        rule_ratio = len(req_rxns & rxns) / len(req_rxns)
        if rule_ratio >= min_rule_rxn_freq:
            unit_ids.add(rule_to_unit[rule_id])
    if apply_sugar_modular_cleanup is not None:
        # remove modular and sugar units added solely by reactions.
        # generally, these are mis-annotations derived from single-gene
        # reactions from the unit mapping curation
        unit_ids = (
            unit_ids
            - set(unit_to_module_tag.keys())
            - set(unit_to_sugar_tag.keys())
        )
        if apply_sugar_modular_cleanup == "strict":
            unit_ids = unit_ids - loose_pks_unit_ids - loose_sugar_unit_ids
    # determine final sugar complement
    if len(sugar_tags) > 0:
        sugar_cache = {}
        for unit_id, sugar_reqs in unit_to_sugar_tag.items():
            sugar_ratio = len(sugar_reqs & sugar_tags) / len(sugar_reqs)
            if sugar_ratio >= min_sugar_rxn_freq:
                sugar_cache[unit_id] = sugar_ratio
        sorted_sugar_cache = list(
            sorted(sugar_cache.items(), key=lambda x: x[1], reverse=True)
        )
        # add highest scoring sugars up to the number of available GTrs
        try:
            for i in range(num_gtrs):
                unit_ids.add(sorted_sugar_cache[i][0])
        except IndexError:
            # if sugars don't pass the threshold for the GTr, ignore.
            pass
    # create signature
    signature = {unit_id_to_unit[unit_id] for unit_id in unit_ids}
    return {"nonmodular_signature": signature}


def get_nonmodular_signature_from_molecule(
    bear_fp: str, nodes: List[str] = None
):
    G = get_nonmodular_bear_graph(bear_fp=bear_fp)
    if nodes != None:
        target_nodes = set(G.nodes) & set(nodes)
    else:
        target_nodes = set(G.nodes)
    nonmod_lookup = {}
    for n in target_nodes:
        smarts_hash_id = G.nodes[n]["smarts_hash_id"]
        unit_id = smarts_to_unit[smarts_hash_id]
        nonmod_lookup[n] = unit_id_to_unit[unit_id]
    # no kmers in this method.
    signature = set(nonmod_lookup.values())
    return {"nonmodular_signature": signature}
