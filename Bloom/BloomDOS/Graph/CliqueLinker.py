from typing import List, Tuple

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher


class CliqueLinker:
    # load enum pairings to connect groups during clique calculation
    def __init__(self, enum_1: str, enum_2: str):
        self.G = nx.Graph()
        self.G.add_node(1, enum=enum_1)
        self.G.add_node(2, enum=enum_2)
        self.G.add_edge(1, 2)

    def search(self, subject: nx.Graph) -> List[Tuple[int, int]]:
        # use isomorphic search to find specifc edges
        gm = GraphMatcher(subject, self.G, node_match=self.node_match)
        return [
            [x for x, y in observed_to_query.items()]
            for observed_to_query in gm.subgraph_isomorphisms_iter()
        ]

    @classmethod
    def _load_from_row(cls, row) -> "CliqueLinker":
        return cls(row["enum_1"], row["enum_2"])

    @staticmethod
    def node_match(a, b):
        # a -> subject
        # b -> query
        if b["enum"] in a["enums"]:
            return True
        else:
            return False
