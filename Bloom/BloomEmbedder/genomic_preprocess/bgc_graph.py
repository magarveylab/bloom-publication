import json
import pickle
from typing import List

from Bloom.BloomEmbedder.genomic_preprocess.common import (
    create_graph_from_dags,
)

########################################################################
# Graph Schema
########################################################################

col_name_to_node_type = {
    "cluster_id": "BGC",
    "orf_id": "Orf",
    "module_id": "Module",
    "domain_id": "Domain",
    "adj_module_id": "Module",
}
col_name_to_node_properties = {
    "orf_id": ["orf_hash_id"],
    "domain_id": ["domain_hash_id"],
}
col_name_to_edge_type = {
    ("cluster_id", "orf_id"): "cluster_to_orf",
    ("orf_id", "module_id"): "orf_to_module",
    ("module_id", "adj_module_id"): "module_adj",
    ("module_id", "domain_id"): "module_to_domain",
}
col_name_to_edge_properties = {}

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


def get_bgc_graphs(
    ibis_dir: str,
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
    # get all dags
    dags = []
    dags.extend(get_module_adj_dag(input_data["module"], protein_to_orfs))
    dags.extend(
        get_domain_dag(
            input_data["domain"], input_data["module"], protein_to_orfs
        )
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


def get_node_labels_for_bgc_graph(ibis_dir: str, min_ec_homology: float = 0.8):
    # load input data
    orf_labels = {}
    module_labels = {}
    # add ed predictions
    for orf in json.load(open(f"{ibis_dir}/ec_predictions.json")):
        protein_id = orf["query_id"]
        if len(orf["predictions"]) == 0:
            continue
        top = orf["predictions"][0]
        if top["homology"] >= min_ec_homology:
            ec4 = top["label"]
            ec3 = ".".join(top["label"].split(".")[:3])
            ec2 = ".".join(top["label"].split(".")[:2])
            ec1 = top["label"].split(".")[0]
            orf_labels[protein_id] = {
                "ec1": ec1,
                "ec2": ec2,
                "ec3": ec3,
                "ec4": ec4,
            }
    # add module predictions
    for orf in json.load(open(f"{ibis_dir}/module_predictions.json")):
        protein_id = orf["protein_id"]
        if len(orf["modules"]) == 0:
            continue
        for m in orf["modules"]:
            protein_start = m["protein_start"]
            protein_stop = m["protein_stop"]
            module_id = f"{protein_id}_{protein_start}_{protein_stop}"
            module_tag = m["tags"][0]["tag"]
            module_labels[module_id] = {
                "module_tag": module_tag,
            }
    return {"Orf": orf_labels, "Module": module_labels}
