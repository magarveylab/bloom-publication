import json
import pickle
from typing import List

import pandas as pd

from BearLinker import curdir
from BearLinker.neo4j.bgc_graph import (
    col_name_to_edge_properties,
    col_name_to_edge_type,
    col_name_to_node_properties,
    col_name_to_node_type,
)
from BearLinker.neo4j.common import create_graph_from_dags

########################################################################
# Load libraries
########################################################################


def load_library():
    library_dir = f"{curdir}/local/tables"
    library = {
        "module_to_substrate": {},
        "module_to_substrate_family": {},
        "gene_to_protein_family": {},
    }
    # module to pks reaction
    df = pd.read_csv(f"{library_dir}/module_to_pks_reaction.csv")
    library["module_to_pks_reaction"] = dict(
        zip(df.module_tag, df.pks_reaction_tag)
    )
    # module to substrate
    df = pd.read_csv(f"{library_dir}/module_to_substrate.csv")
    library["module_to_substrate"] = dict(zip(df.module_tag, df.substrate))
    # module to substrate family
    df = pd.read_csv(f"{library_dir}/module_to_substrate_family.csv")
    library["module_to_substrate_family"] = dict(
        zip(df.module_tag, df.substrate_family)
    )
    # gene to protein family
    df = pd.read_csv(f"{library_dir}/gene_to_protein_family.csv")
    library["gene_to_protein_family"] = dict(
        zip(df.gene, df.protein_family_tag)
    )
    return library


library = load_library()
module_to_pks_reaction = library["module_to_pks_reaction"]
module_to_substrate = library["module_to_substrate"]
module_to_substrate_family = library["module_to_substrate_family"]
gene_to_protein_family = library["gene_to_protein_family"]

########################################################################
# Helper Functions
########################################################################


def get_protein_to_orfs(prodigal_data: List[dict]):
    protein_to_orfs = {}
    for p in prodigal_data:
        protein_id = p["protein_id"]
        orf_id = f'{p["contig_id"]}_{p["contig_start"]}_{p["contig_stop"]}'
        if protein_id not in protein_to_orfs:
            protein_to_orfs[protein_id] = []
        protein_to_orfs[protein_id].append(orf_id)
    return protein_to_orfs


def get_orf_to_protein(prodigal_data: List[dict]):
    orf_to_protein = {}
    for p in prodigal_data:
        protein_id = p["protein_id"]
        orf_id = f'{p["contig_id"]}_{p["contig_start"]}_{p["contig_stop"]}'
        orf_to_protein[orf_id] = protein_id
    return orf_to_protein


def get_cluster_to_orfs(bgc_data: List[dict]):
    return {
        f'{c["contig_id"]}_{c["contig_start"]}_{c["contig_stop"]}': c["orfs"]
        for c in bgc_data
    }


def get_cluster_to_chemotypes(bgc_data: List[dict]):
    return {
        f'{c["contig_id"]}_{c["contig_start"]}_{c["contig_stop"]}': c[
            "internal_chemotypes"
        ]
        for c in bgc_data
    }


def get_protein_to_domains(domain_data: List[dict]):
    protein_to_domains = {}
    for p in domain_data:
        protein_id = p["protein_id"]
        domain_ids = [d["domain_id"] for d in p["regions"]]
        protein_to_domains[protein_id] = domain_ids
    return protein_to_domains


########################################################################
# DAGs to construct BGC graph
########################################################################


def get_module_adj_dag(module_data: List[dict], protein_to_orfs: dict):
    dags = []
    for p in module_data:
        protein_id = p["protein_id"]
        for m1, m2 in zip(p["modules"], p["modules"][1:]):
            m1_id = f'{protein_id}_{m1["protein_start"]}_{m1["protein_stop"]}'
            m2_id = f'{protein_id}_{m2["protein_start"]}_{m2["protein_stop"]}'
            for orf_id in protein_to_orfs[protein_id]:
                dags.append(
                    {
                        "orf_id": orf_id,
                        "module_id": m1_id,
                        "adj_module_id": m2_id,
                    }
                )
    return dags


def get_domain_dag(
    domain_data: List[dict], module_data: List[dict], protein_to_orfs: dict
):
    target_domains = ["A", "AT", "KS", "KR", "DH", "ER", "T"]
    dags = []
    # create domain hash lookup
    domain_hash_lookup = {}
    for p in domain_data:
        protein_id = p["protein_id"]
        for d in p["regions"]:
            domain_start = d["protein_start"]
            domain_stop = d["protein_stop"]
            domain_hash_id = d["domain_id"]
            domain_id = f"{protein_id}_{domain_start}_{domain_stop}"
            domain_label = d["label"]
            if domain_label not in target_domains:
                continue
            domain_hash_lookup[domain_id] = domain_hash_id
    # parse module data
    for p in module_data:
        protein_id = p["protein_id"]
        for m in p["modules"]:
            module_start = m["protein_start"]
            module_stop = m["protein_stop"]
            module_id = f"{protein_id}_{module_start}_{module_stop}"
            for domain_id in m["domains"]:
                if domain_id not in domain_hash_lookup:
                    continue
                domain_hash_id = domain_hash_lookup[domain_id]
                for orf_id in protein_to_orfs[protein_id]:
                    dags.append(
                        {
                            "orf_id": orf_id,
                            "module_id": module_id,
                            "domain_id": domain_id,
                            "domain_hash_id": domain_hash_id,
                        }
                    )
    return dags


def get_module_dag(module_data: List[dict], protein_to_orfs: dict):
    dags = []
    for p in module_data:
        protein_id = p["protein_id"]
        for m in p["modules"]:
            module_start = m["protein_start"]
            module_stop = m["protein_stop"]
            module_id = f"{protein_id}_{module_start}_{module_stop}"
            module_tag = m["tags"][0]["tag"]
            for orf_id in protein_to_orfs[protein_id]:
                dags.append(
                    {
                        "orf_id": orf_id,
                        "module_id": module_id,
                        "module_tag": module_tag,
                    }
                )
                if module_tag in module_to_substrate:
                    substrate = module_to_substrate[module_tag]
                    dags.append(
                        {
                            "orf_id": orf_id,
                            "module_id": module_id,
                            "substrate": substrate,
                        }
                    )
                if module_tag in module_to_substrate_family:
                    substrate_family_tag = module_to_substrate_family[
                        module_tag
                    ]
                    dags.append(
                        {
                            "orf_id": orf_id,
                            "module_id": module_id,
                            "substrate_family_tag": substrate_family_tag,
                        }
                    )
                if module_tag in module_to_pks_reaction:
                    pks_reaction_tag = module_to_pks_reaction[module_tag]
                    dags.append(
                        {
                            "orf_id": orf_id,
                            "module_id": module_id,
                            "pks_reaction_tag": pks_reaction_tag,
                        }
                    )
    return dags


def get_ec_dag(
    ec_data: List[dict], protein_to_orfs: dict, min_ec_homology: float = 0.6
):
    dags = []
    for query in ec_data:
        protein_id = query["query_id"]
        predictions = query["predictions"]
        if len(predictions) == 0:
            continue
        ec4 = predictions[0]["label"]
        ec3 = ".".join(ec4.split(".")[:3])
        homology = predictions[0]["homology"]
        if homology >= min_ec_homology:
            for orf_id in protein_to_orfs[protein_id]:
                dags.append({"orf_id": orf_id, "ec4": ec4, "ec3": ec3})
    return dags


def get_gtr_dag(domain_data: List[dict], protein_to_orfs: dict):
    dags = []
    for p in domain_data:
        if len(p["regions"]) == 1 and p["regions"][0]["label"] == "GTr":
            protein_id = p["protein_id"]
            for orf_id in protein_to_orfs[protein_id]:
                dags.append({"orf_id": orf_id, "domain_label": "GTr"})
    return dags


def get_protein_family_tag_dag(
    gene_data: List[dict],
    protein_to_orfs: dict,
    min_gene_similarity: float = 0.09,
):
    dags = []
    for query in gene_data:
        protein_id = query["query_id"]
        predictions = query["predictions"]
        if len(predictions) == 0:
            continue
        gene = predictions[0]["label"]
        if gene not in gene_to_protein_family:
            continue
        protein_family_tag = gene_to_protein_family[gene]
        similarity = predictions[0]["similarity"]
        if similarity >= min_gene_similarity:
            for orf_id in protein_to_orfs[protein_id]:
                dags.append(
                    {
                        "orf_id": orf_id,
                        "gene": gene,
                        "protein_family_tag": protein_family_tag,
                    }
                )
    return dags


def get_bgc_graphs(
    ibis_dir: str,
    min_orf_count: int = 4,
    min_module_count: int = 4,
    run_all: bool = False,
):
    # load input data
    input_data = {}
    input_data["prodigal"] = json.load(open(f"{ibis_dir}/prodigal.json"))
    input_data["bgc"] = json.load(open(f"{ibis_dir}/bgc_predictions.json"))
    input_data["module"] = json.load(
        open(f"{ibis_dir}/module_predictions.json")
    )
    input_data["domain"] = json.load(
        open(f"{ibis_dir}/domain_predictions.json")
    )
    input_data["ec"] = json.load(open(f"{ibis_dir}/ec_predictions.json"))
    input_data["gene"] = json.load(open(f"{ibis_dir}/gene_predictions.json"))
    # get protein to orfs
    protein_to_orfs = get_protein_to_orfs(input_data["prodigal"])
    orf_to_protein = get_orf_to_protein(input_data["prodigal"])
    # get cluster to orfs
    cluster_to_orfs = get_cluster_to_orfs(input_data["bgc"])
    # get cluster to chemotypes
    cluster_to_chemotypes = get_cluster_to_chemotypes(input_data["bgc"])
    # get all dags
    dags = []
    dags.extend(get_module_adj_dag(input_data["module"], protein_to_orfs))
    dags.extend(
        get_domain_dag(
            input_data["domain"], input_data["module"], protein_to_orfs
        )
    )
    dags.extend(get_module_dag(input_data["module"], protein_to_orfs))
    dags.extend(get_ec_dag(input_data["ec"], protein_to_orfs))
    dags.extend(get_gtr_dag(input_data["domain"], protein_to_orfs))
    dags.extend(
        get_protein_family_tag_dag(input_data["gene"], protein_to_orfs)
    )
    # sort dags by orfs
    orf_to_dags = {}
    for d in dags:
        orf_id = d["orf_id"]
        if orf_id not in orf_to_dags:
            orf_to_dags[orf_id] = []
        orf_to_dags[orf_id].append(d)
    # create graphs
    out = []
    for cluster_id, orfs in cluster_to_orfs.items():
        connected_orfs = [orf_id for orf_id in orfs if orf_id in orf_to_dags]
        dags = [d for orf_id in connected_orfs for d in orf_to_dags[orf_id]]
        dags.extend(
            [
                {
                    "cluster_id": cluster_id,
                    "orf_id": orf_id,
                    "orf_hash_id": orf_to_protein[orf_id],
                }
                for orf_id in orfs
            ]
        )
        G = create_graph_from_dags(
            dags=dags,
            col_name_to_node_type=col_name_to_node_type,
            col_name_to_node_properties=col_name_to_node_properties,
            col_name_to_edge_type=col_name_to_edge_type,
            col_name_to_edge_properties=col_name_to_edge_properties,
        )
        # check if graph passes quality control
        if run_all == True:
            quality = True
        else:
            quality = False
            chemotypes = cluster_to_chemotypes[cluster_id]
            # control for type I PKS and NRPS (need at least 4 modules)
            if (
                "TypeIPolyketide" in chemotypes
                or "NonRibosomalPeptide" in chemotypes
            ):
                module_count = len([n for n in G.nodes if n[0] == "Module"])
                if module_count >= min_module_count:
                    quality = True
                else:
                    quality = False
            elif "Ripp" in chemotypes or "Bacteriocin" in chemotypes:
                quality = False
            # control for other BGCs (need at least 4 orfs)
            else:
                orf_count = len([n for n in G.nodes if n[0] == "Orf"])
                if orf_count >= min_orf_count:
                    quality = True
                else:
                    quality = False
        if quality == True:
            out.append({"cluster_id": cluster_id, "graph": G})
    return out


def get_embeddings_for_bgc_graph(ibis_dir: str):
    # load input data
    input_data = {}
    input_data["prodigal"] = json.load(open(f"{ibis_dir}/prodigal.json"))
    input_data["bgc"] = json.load(open(f"{ibis_dir}/bgc_predictions.json"))
    input_data["domain"] = json.load(
        open(f"{ibis_dir}/domain_predictions.json")
    )
    input_data["protein_emb"] = pickle.load(
        open(f"{ibis_dir}/protein_embedding.pkl", "rb")
    )
    input_data["domain_emb"] = pickle.load(
        open(f"{ibis_dir}/domain_embedding.pkl", "rb")
    )
    orf_to_protein = get_orf_to_protein(input_data["prodigal"])
    protein_to_domains = get_protein_to_domains(input_data["domain"])
    prot_emb_lookup = {
        p["protein_id"]: p["embedding"] for p in input_data["protein_emb"]
    }
    dom_emb_lookup = {
        p["domain_id"]: p["embedding"] for p in input_data["domain_emb"]
    }
    # prepare cluster to embeddings
    cluster_to_embeddings = {}
    for c in input_data["bgc"]:
        contig_id = c["contig_id"]
        contig_start = c["contig_start"]
        contig_stop = c["contig_stop"]
        cluster_id = f"{contig_id}_{contig_start}_{contig_stop}"
        cluster_to_embeddings[cluster_id] = {
            "cluster_id": cluster_id,
            "orfs": {},
            "domains": {},
        }
        for orf_id in c["orfs"]:
            protein_id = orf_to_protein[orf_id]
            prot_emb = prot_emb_lookup[protein_id]
            cluster_to_embeddings[cluster_id]["orfs"][protein_id] = prot_emb
            if protein_id in protein_to_domains:
                for domain_id in protein_to_domains[protein_id]:
                    if domain_id in dom_emb_lookup:
                        dom_emb = dom_emb_lookup[domain_id]
                        cluster_to_embeddings[cluster_id]["domains"][
                            domain_id
                        ] = dom_emb
    return cluster_to_embeddings
