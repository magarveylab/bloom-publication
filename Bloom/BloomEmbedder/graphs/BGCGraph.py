import json
from glob import glob

from Bloom.BloomEmbedder.genomic_preprocess.bgc_graph import (
    get_bgc_graphs,
    get_embeddings_for_bgc_graph,
    get_node_labels_for_bgc_graph,
)
from Bloom.BloomEmbedder.utils import curdir
from Bloom.CommonUtils.HeteroGraph import HeteroGraph


def get_vocab(vocab_dir: str):
    vocab_filenames = glob(vocab_dir)
    vocab = {}
    for fp in vocab_filenames:
        key = fp.split("/")[-1].split(".")[0]
        vocab[key] = json.load(open(fp))
    return vocab


def get_node_vocab(
    vocab_dir: str = f"{curdir}/models/bgc/vocab/node_vocab/*",
):
    return get_vocab(vocab_dir=vocab_dir)


class BGCGraph(HeteroGraph):

    def __init__(self, graph_id: str):
        # define schema
        schema = {}
        schema["node_types"] = [
            "BGC",
            "Orf",
            "Module",
            "Domain",
        ]
        schema["edge_types"] = [
            ("Orf", "orf_to_module", "Module"),
            ("Module", "module_adj", "Module"),
            ("Module", "module_to_domain", "Domain"),
        ]
        schema["node_embedding_dim"] = {
            "Domain": 1024,
            "Orf": 1024,
        }
        schema["edge_embedding_dim"] = {}
        schema["node_labels"] = {
            "Orf": ["ec1", "ec2", "ec3", "ec4"],
            "Module": ["module_tag"],
        }
        self.graph_id = graph_id
        super().__init__(schema=schema)

    @classmethod
    def build_from_ibis_output(cls, ibis_dir: str):
        # get cluster graphs
        bgc_graphs = get_bgc_graphs(ibis_dir=ibis_dir)
        # get embeddings (orfs and domains)
        ibis_embeddings = get_embeddings_for_bgc_graph(ibis_dir=ibis_dir)
        # get node labels
        node_labels = get_node_labels_for_bgc_graph(ibis_dir=ibis_dir)
        # build individual graphs
        out = []
        for g in bgc_graphs:
            cluster_id = g["cluster_id"]
            orf_embedding = ibis_embeddings[cluster_id]["orfs"]
            domain_embedding = ibis_embeddings[cluster_id]["domains"]
            graph = g["graph"]
            G = cls(graph_id=cluster_id)
            # add nodes
            node_conversion = {}
            orf_traceback = {}
            module_traceback = {}
            for n in graph.nodes:
                node_type, _ = n[0], n[1]
                if node_type == "BGC":
                    G_node_id = G.add_node(node_type="BGC", label="[CLS]")
                elif node_type == "Orf":
                    orf_hash_id = graph.nodes[n]["properties"]["orf_hash_id"]
                    G_node_id = G.add_node(
                        node_type="Orf", embedding=orf_embedding[orf_hash_id]
                    )
                    # track node ids in graph
                    orf_traceback[G_node_id] = n[-1]
                    # add node labels
                    if orf_hash_id in node_labels["Orf"]:
                        for label_name in node_labels["Orf"][orf_hash_id]:
                            G.node_label_lookup[G_node_id][label_name] = (
                                node_labels["Orf"][orf_hash_id][label_name]
                            )
                elif node_type == "Domain":
                    domain_hash_id = graph.nodes[n]["properties"][
                        "domain_hash_id"
                    ]
                    G_node_id = G.add_node(
                        node_type="Domain",
                        embedding=domain_embedding[domain_hash_id],
                    )
                elif node_type == "Module":
                    G_node_id = G.add_node(node_type="Module", label="[CLS]")
                    module_traceback[G_node_id] = n[-1]
                    # add node labels
                    if n[-1] in node_labels["Module"]:
                        for label_name in node_labels["Module"][n[-1]]:
                            G.node_label_lookup[G_node_id][label_name] = (
                                node_labels["Module"][n[-1]][label_name]
                            )
                node_conversion[n] = G_node_id
            # add edges
            edge_type_lookup = {e[1]: e for e in G.edge_types}
            for n1, n2, e in graph.edges(data=True):
                # note that the graph is not directional
                # need to account for that when adding edges
                edge_name = e["edge_type"]
                if edge_name == "cluster_to_orf":
                    continue
                edge_type = edge_type_lookup[edge_name]
                n1_node_type = edge_type[0]
                if n1_node_type == n1[0]:
                    G_n1 = node_conversion[n1]
                    G_n2 = node_conversion[n2]
                else:
                    G_n1 = node_conversion[n2]
                    G_n2 = node_conversion[n1]
                G.add_edge(G_n1, G_n2, edge_type=edge_type)
            out.append(
                {
                    "graph": G,
                    "orf_traceback": orf_traceback,
                    "module_traceback": module_traceback,
                }
            )
        return out
