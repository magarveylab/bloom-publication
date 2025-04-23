import json
import pickle
from glob import glob

import networkx as nx
import numpy as np

from Bloom.BloomLNK import curdir
from Bloom.CommonUtils.HeteroGraph import HeteroGraph


def get_vocab(
    vocab_dir: str = f"{curdir}/preprocess/graph/node_vocab/*",
):
    vocab_filenames = glob(vocab_dir)
    vocab = {}
    for fp in vocab_filenames:
        node_type = fp.split("/")[-1].split(".")[0]
        vocab[node_type] = json.load(open(fp))
    return vocab


def get_preloaded_reaction_embedding(rxn_emb_fp: str):
    return pickle.load(open(rxn_emb_fp, "rb"))


preloaded_rxn_embedding = get_preloaded_reaction_embedding(
    rxn_emb_fp=f"{curdir}/preprocess/reaction_embedding.pkl"
)


class MetaboloGraph(HeteroGraph):

    def __init__(self, graph_id: str):
        # define schema
        schema = {}
        schema["node_types"] = [
            "Pair",
            "Orf",
            "Module",
            "Domain",
            "ModuleTag",
            "Substrate",
            "SubstrateFamilyTag",
            "PKSReactionTag",
            "EC4",
            "EC3",
            "Gene",
            "ProteinFamilyTag",
            "MetaboliteMolecularRegion",
            "UnitRule",
            "SugarReactionTag",
            "TailoringReactionTag",
            "Reaction",
        ]
        schema["edge_types"] = [
            # genomic edges
            ("Orf", "orf_to_module", "Module"),
            ("Module", "module_adj", "Module"),
            ("Module", "module_to_domain", "Domain"),
            # module edges
            ("Module", "module_to_tag", "ModuleTag"),
            ("Module", "module_to_substrate", "Substrate"),
            ("Module", "module_to_substrate_family_tag", "SubstrateFamilyTag"),
            ("Module", "module_to_pks_reaction_tag", "PKSReactionTag"),
            # orf edges
            ("Orf", "orf_to_ec4", "EC4"),
            ("EC4", "ec4_to_ec3", "EC3"),
            ("Orf", "orf_to_gene", "Gene"),
            ("Gene", "gene_to_protein_family_tag", "ProteinFamilyTag"),
            # metabolite edges
            (
                "MetaboliteMolecularRegion",
                "mol_region_adj",
                "MetaboliteMolecularRegion",
            ),
            # unit to module connections (module_tag, pks reaction, substrate, substrate family)
            ("MetaboliteMolecularRegion", "unit_to_module_tag", "ModuleTag"),
            (
                "MetaboliteMolecularRegion",
                "unit_to_pks_reaction_tag",
                "PKSReactionTag",
            ),
            ("MetaboliteMolecularRegion", "unit_to_substrate", "Substrate"),
            (
                "MetaboliteMolecularRegion",
                "unit_to_substrate_family_tag",
                "SubstrateFamilyTag",
            ),
            # unit to rules
            ("MetaboliteMolecularRegion", "unit_to_rule", "UnitRule"),
            # unit to sugar connections
            ("UnitRule", "rule_to_sugar_reaction_tag", "SugarReactionTag"),
            ("SugarReactionTag", "sugar_reaction_tag_to_ec4", "EC4"),
            # unit to tailoring connections
            (
                "UnitRule",
                "rule_to_tailoring_reaction_tag",
                "TailoringReactionTag",
            ),
            ("TailoringReactionTag", "tailoring_reaction_tag_to_ec3", "EC3"),
            # unit to reaction connections
            ("UnitRule", "rule_to_reaction", "Reaction"),
            ("Reaction", "reaction_to_ec4", "EC4"),
            ("Reaction", "reaction_to_protein_family_tag", "ProteinFamilyTag"),
        ]

        schema["node_embedding_dim"] = {
            "Domain": 1024,
            "Orf": 1024,
            "Reaction": 256,
        }
        schema["edge_embedding_dim"] = {}
        self.graph_id = graph_id
        super().__init__(schema=schema)

    @classmethod
    def build_graph(
        cls,
        graph_id: str,
        graph: nx.Graph,
        orf_embedding: dict,
        domain_embedding: dict,
        rxn_embedding: dict = preloaded_rxn_embedding,
    ):
        G = cls(graph_id=graph_id)
        # add nodes
        node_conversion = {}
        for n in graph.nodes:
            node_type, label = n[0], n[1]
            if node_type == "Orf":
                orf_hash_id = graph.nodes[n]["properties"]["orf_hash_id"]
                G_node_id = G.add_node(
                    node_type="Orf", embedding=orf_embedding[orf_hash_id]
                )
            elif node_type == "Domain":
                domain_hash_id = graph.nodes[n]["properties"]["domain_hash_id"]
                G_node_id = G.add_node(
                    node_type="Domain",
                    embedding=domain_embedding[domain_hash_id],
                )
            elif node_type == "Reaction":
                reaction_id = graph.nodes[n]["properties"]["reaction_id"]
                G_node_id = G.add_node(
                    node_type="Reaction",
                    embedding=rxn_embedding.get(reaction_id, np.zeros(256)),
                )
            elif node_type in [
                "Pair",
                "Module",
            ]:
                G_node_id = G.add_node(node_type=node_type, label="[CLS]")
            elif node_type == "MetaboliteMolecularRegion":
                unit_id = graph.nodes[n]["properties"].get("unit_id", "[UNK]")
                G_node_id = G.add_node(node_type=node_type, label=str(unit_id))
            elif node_type in G.node_types:
                G_node_id = G.add_node(node_type=node_type, label=str(label))
            node_conversion[n] = G_node_id
        # add edges
        edge_type_lookup = {e[1]: e for e in G.edge_types}
        for n1, n2, e in graph.edges(data=True):
            # note that the graph is not directional
            # need to account for that when adding edges
            edge_name = e["edge_type"]
            if edge_name == "orf_to_domain_label":
                continue
            edge_type = edge_type_lookup[edge_name]
            n1_node_type = edge_type[0]
            if n1_node_type == n1[0]:
                G_n1 = node_conversion[n1]
                G_n2 = node_conversion[n2]
            else:
                G_n1 = node_conversion[n2]
                G_n2 = node_conversion[n1]
            # note no edge properties (like similarity or homology score)
            # overcomplicates the graph
            G.add_edge(G_n1, G_n2, edge_type=edge_type)
        return G
