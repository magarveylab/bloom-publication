import logging

from Bloom.BloomDOS.Graph.AnnotationGraph import AnnotationGraph


class PeptideDereplication:

    def __init__(self, G: AnnotationGraph):
        self.G = G

    def remove_by_azoles(self):
        logging.info("Removing unconfident azole annotations")
        # extract subgraph for analysis
        edges_to_keep = self.G.get_edges(overlap=True, exact=True)
        subgraph = self.G.edge_subgraph(edges_to_keep)
        # remove azoles that overlap exactly with canonical amino acids
        to_remove = set()
        azole_nodes = self.G.get_nodes(enums={"azole"}) & set(subgraph.nodes)
        for n in azole_nodes:
            neighbors = subgraph.neighbors(n)
            neighbor_enums = self.G.get_enums_from_nodes(neighbors)
            if "aa" in neighbor_enums:
                to_remove.add(n)
        self.G.remove_nodes_from(to_remove)
