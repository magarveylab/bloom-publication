from typing import Dict, List, Set, Tuple

from Bloom.BloomGraph.BloomGraph import DataStructs


class DataMethods:

    def add_node(self, node: DataStructs.BloomNodeDict):
        node_id = node["node_id"]
        # parse enums
        if "enums" in node and node["enums"] != None:
            enums = set(node["enums"])
        else:
            enums = set()
        # contains potential substitutions
        labels = {l["label"]: l["score"] for l in node["labels"]}
        # top label is unit reflected in the smiles
        top_label = (
            node["top_label"]
            if "top_label" in node and node["top_label"] in labels
            else max(labels.items(), key=lambda x: x[1])[0]
        )
        # parse meta
        meta = (
            node["meta"]
            if "meta" in node and isinstance(node["meta"], dict)
            else {}
        )
        # number of repeats
        count = (
            node["count"]
            if "count" in node and isinstance(node["count"], int)
            else 1
        )
        # cache node
        self.G.add_node(
            node_id,
            top_label=top_label,
            labels=labels,
            meta=meta,
            count=count,
            enums=enums,
            atoms=set(),
            intrabonds=set(),
        )

    def add_edge(self, edge: DataStructs.BloomEdgeDict):
        self.G.add_edge(edge["n1"], edge["n2"], meta=edge.get("meta", {}))

    def add_nodes_from(self, nodes: List[DataStructs.BloomNodeDict]):
        [self.add_node(n) for n in nodes]

    def add_edges_from(self, edges: List[DataStructs.BloomEdgeDict]):
        [self.add_edge(e) for e in edges]

    def add_enum(self, node: int, enums: Set[str]):
        if node in self.nodes:
            self[node]["enums"].update(enums)

    def add_enums_from(self, enum_dict: Dict[int, Set[str]]):
        [self.add_enum(node, enums) for node, enums in enum_dict.items()]

    def add_node_meta(self, node: int, meta: dict):
        if node in self.nodes:
            m = self[node]["meta"].copy()
            m.update(meta)
            self.G.nodes[node]["meta"] = m

    def add_node_meta_from(self, meta_to_add: Dict[int, dict]):
        [self.add_node_meta(node, meta) for node, meta in meta_to_add.items()]

    def add_edge_meta(self, n1: int, n2: int, meta: dict):
        if self.G.has_edge(n1, n2):
            m = self.G[n1][n2]["meta"].copy()
            m.update(meta)
            self.G[n1][n2]["meta"] = m

    def add_edge_meta_from(self, meta_to_add: Dict[Tuple[int, int], dict]):
        [
            self.add_edge_meta(n1, n2, meta)
            for (n1, n2), meta in meta_to_add.items()
        ]

    def add_atomic_data_to_graph(self):
        # clear existing atomic information
        for n in self.nodes:
            self[n]["atoms"] = set()
            self[n]["intrabonds"] = set()
        # add atom indexes to unit graph
        for atom in self.mol.GetAtoms():
            node_id = atom.GetAtomMapNum()
            if node_id in self.nodes:
                self[node_id]["atoms"].add(atom.GetIdx())
        # add bond indexes to unit graph
        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom().GetAtomMapNum()
            a2 = bond.GetEndAtom().GetAtomMapNum()
            if a1 == a2 and a1 in self.nodes:
                self[a1]["intrabonds"].add(bond.GetIdx())
        self.enum_to_nodes = self._enum_to_nodes
