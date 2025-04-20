import networkx as nx


class ModuleGraphMethods:

    def get_pks_modular_graph(
        self, keep_longest_chain: bool = False
    ) -> nx.Graph:
        G = self.G.copy()
        G = G.subgraph(self.pks_nodes).copy()
        if keep_longest_chain:
            longest_module_chain_edges = self.longest_module_chain_edges
            edges_to_remove = set(G.edges) - longest_module_chain_edges
            G.remove_edges_from(edges_to_remove)
        return G

    def get_nrps_modular_graph(self) -> nx.Graph:
        G = self.G.copy()
        G = G.subgraph(self.nrps_nodes).copy()
        return G

    def get_nonmodular_graph(self) -> nx.Graph:
        G = self.G.copy()
        G = G.subgraph(self.nonmodular_nodes).copy()
        return G
