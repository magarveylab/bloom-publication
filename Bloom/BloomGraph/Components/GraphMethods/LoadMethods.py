import json
from typing import List, Optional, Set

from rdkit import Chem

from Bloom.BloomGraph.BloomGraph import DataStructs


class LoadMethods:

    @classmethod
    def load_from_json(cls, json_fp: str):
        # load data
        data = json.load(open(json_fp))
        # initiate graph
        G = cls()
        G.add_nodes_from(data["nodes"])
        G.add_edges_from(data["edges"])
        G.smiles = data["smiles"]
        G.meta = data.get("meta", {})
        return G

    @classmethod
    def load_from_smiles(
        cls,
        node_id: int,
        label: str,
        smiles: str,
        meta: dict = {},
        labels: List[DataStructs.NodeLabelDict] = [],
        enums: Set[str] = set(),
    ):
        # add map number to smiles (conection to unit graph)
        mol = Chem.MolFromSmiles(smiles)
        [a.SetAtomMapNum(node_id) for a in mol.GetAtoms()]
        # initiate graph
        G = cls()
        if len(labels) == 0:
            labels = [{"label": label, "score": 1.0}]
        node = {
            "node_id": node_id,
            "labels": labels,
            "enums": enums,
            "meta": meta,
        }
        G.add_node(node)
        G.mol = mol
        return G

    @classmethod
    def load_from_nodes(
        cls,
        nodes: List[DataStructs.BloomNodeDict],
        smiles: str,
        edges: Optional[List[DataStructs.BloomEdgeDict]] = None,
    ):
        G = cls()
        # add smiles
        G.smiles = smiles
        # add nodes based on smiles
        nodes_to_keep = G.nodes_in_mol
        G.add_nodes_from([n for n in nodes if n["node_id"] in nodes_to_keep])
        # trace meta edge data
        edge_dict = {n: {} for n in nodes_to_keep}
        if edges != None:
            for e in edges:
                edge_dict[e["n1"]][e["n2"]] = e["meta"]
                edge_dict[e["n2"]][e["n1"]] = e["meta"]
        # add edges based on smiles
        edges_to_keep = G.edges_in_mol
        for n1, n2 in edges_to_keep:
            G.add_edge({"n1": n1, "n2": n2, "meta": edge_dict[n1].get(n2, {})})
        G.add_atomic_data_to_graph()
        return G
