import itertools as it
from typing import List, Set, Tuple

import networkx as nx


class ModuleProperties:

    @property
    def longest_module_chain(self) -> List[int]:
        nonmodular_nodes = self.nonmodular_nodes
        G = self.G.copy()
        G.remove_nodes_from(nonmodular_nodes)
        paths = []
        for s, t in it.product(self.starter_nodes, self.terminal_nodes):
            if nx.has_path(G, s, t):
                paths.extend(nx.all_simple_paths(G, source=s, target=t))
        if len(paths) == 0:
            return []
        else:
            return max(paths, key=lambda x: len(x))

    @property
    def longest_module_chain_edges(self) -> Set[Tuple[int, int]]:
        chain = self.longest_module_chain
        edges = set(zip(chain, chain[1:]))
        reverse_edges = set(e[::-1] for e in edges)
        return edges | reverse_edges
