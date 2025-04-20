import itertools as it
from typing import Set

from rdkit import Chem

from Bloom.BloomDOS.Graph.AnnotationGraph import AnnotationGraph
from Bloom.BloomGraph.BloomGraph import BloomGraph


class MolecularGraph(BloomGraph):

    def __init__(self):
        super().__init__()

    @classmethod
    def load_graph_from_annotations(
        cls,
        mol: Chem.Mol,
        G_anno: AnnotationGraph,
        final_annotations: Set[int],
    ) -> "MolecularGraph":

        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))

        # construct unit graph
        graph = cls()
        unique_node_id = 1
        # add known annotations to unit graph
        for anno_id in final_annotations:
            prop = G_anno[anno_id]
            meta = {
                "description": prop["description"],
                "hash_id": prop.get("hash_id"),
                "anno_id": anno_id,
                "atom_indexes": list(prop["patt"]),
            }
            node = {
                "node_id": unique_node_id,
                "labels": [{"label": prop["notation"], "score": 1}],
                "enums": prop["enums"],
                "meta": meta,
            }
            graph.add_node(node)
            # annotate nodes of bottom layer
            for atom in prop["patt"]:

                mol.GetAtomWithIdx(atom).SetAtomMapNum(unique_node_id)
            unique_node_id += 1
        # add unknown annotations
        unknown_units = G_anno.get_unknown_units(final_annotations)
        for idx, anno_patt in enumerate(unknown_units):
            unknown_response = {
                "patt": anno_patt,
                "patt_len": len(anno_patt),
                "original_patt_len": len(anno_patt),
                "enums": {"unknown"},
                "avoid_dereplication": True,
                "can_dereplicate": False,
                "alpha_hit": set(),
                "variable_restrictions": {},
                "unknown_restrictions": set(),
                "priortization": 1,
            }
            # add unknown annotatoin to annotation graph
            anno_id = G_anno.add_node(unknown_response)
            # add node to top layer
            prop = G_anno[anno_id]
            label = "unk_{}".format(idx)
            description = "unknown_{}".format(idx)
            meta = {
                "description": description,
                "anno_id": anno_id,
                "atom_indexes": list(anno_patt),
            }
            node = {
                "node_id": unique_node_id,
                "labels": [{"label": label, "score": 1.0}],
                "enums": {"unknown"},
                "meta": meta,
            }
            graph.add_node(node)
            # annotate nodes of bottom layer
            for atom in prop["patt"]:
                mol.GetAtomWithIdx(atom).SetAtomMapNum(unique_node_id)
            unique_node_id += 1
        # add edges between units (based on molecular bonds)
        for n1, n2 in it.combinations(graph.nodes, 2):
            if G_anno.is_adjacent(
                graph[n1]["meta"]["anno_id"], graph[n2]["meta"]["anno_id"]
            ):
                edge = {"n1": n1, "n2": n2}
                graph.add_edge(edge)
        # add mol
        graph.mol = mol
        return graph
