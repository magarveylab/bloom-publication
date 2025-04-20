from typing import List, Set

import networkx as nx


class NetworkProperties:

    @property
    def disconnects(self) -> List[Set[int]]:
        return [set(g) for g in nx.connected_components(self.G)]

    @property
    def split(self):
        disconnects = self.disconnects
        return [
            self.subgraph(nodes=nodes, adjust_mol=True)
            for nodes in disconnects
        ]
