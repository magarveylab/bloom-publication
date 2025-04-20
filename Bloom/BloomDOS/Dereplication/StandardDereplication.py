import logging

from Bloom.BloomDOS.Graph.AnnotationGraph import AnnotationGraph


class StandardDereplication:

    def __init__(self, G: AnnotationGraph):
        self.G = G

    def remove_by_confident_annotations(self):
        logging.info("Removing annotations by user defined confidence")
        confident_enums = {"sugar", "fattyacid", "nucleoside"}
        unconfident_enums = {"pk"}
        keep_enums = {"curated_module", "DA"}
        unconfident_nodes = self.G.get_nodes(enums=unconfident_enums)
        keep_nodes = self.G.get_nodes(enums=keep_enums)
        unconfident_nodes = unconfident_nodes - keep_nodes
        # remove unconfident annotations that overlap with confident annotations
        to_remove = set()
        for n in unconfident_nodes:
            neighbors = self.G.neighbors(n)
            neighbor_enums = self.G.get_enums_from_nodes(neighbors)
            if len(neighbor_enums & confident_enums) > 0:
                to_remove.add(n)
        # remove units
        self.G.remove_nodes_from(to_remove)

    def remove_exact_annotations_by_rings(self):
        # helps resolve Piz annotations over Orn
        logging.info("Removing exact overlapping annotations by rings")
        edges = self.G.get_edges(overlap=True, exact=True)
        subgraph = self.G.edge_subgraph(edges)
        to_remove = set()
        for n1, n2 in subgraph.edges:
            if (
                "ring_aa" in self.G[n1]["enums"]
                and "ring_aa" not in self.G[n2]["enums"]
            ):
                to_remove.add(n2)
            elif (
                "ring_aa" not in self.G[n1]["enums"]
                and "ring_aa" in self.G[n2]["enums"]
            ):
                to_remove.add(n1)
        # remove units
        self.G.remove_nodes_from(to_remove)

    def remove_exact_annotations(self):
        logging.info("Removing exact overlapping annotations")
        edges = self.G.get_edges(overlap=True, exact=True)
        subgraph = self.G.edge_subgraph(edges)
        to_remove = set()
        for n1, n2 in subgraph.edges:
            # annotations must match by enums and variable restrictions
            if self.G[n1]["enums"] == self.G[n2]["enums"]:
                if (
                    self.G[n1]["variable_restrictions"]
                    == self.G[n2]["variable_restrictions"]
                ):
                    # consistent removal of annotation
                    if n1 > n2:
                        to_remove.add(n1)
                    else:
                        to_remove.add(n2)
        # remove units
        self.G.remove_nodes_from(to_remove)

    def remove_by_flexibility(self):
        logging.info("Removing annotations by flexibility of SMARTS query")
        edges = self.G.get_edges(overlap=True, exact=True)
        subgraph = self.G.edge_subgraph(edges)
        flexible_nodes = self.G.get_nodes(enums={"flexible"}) & subgraph.nodes
        # remove flexible nodes
        to_remove = set()
        for n in flexible_nodes:
            if len(list(subgraph.neighbors(n))) > 0:
                to_remove.add(n)
        # remove units
        self.G.remove_nodes_from(to_remove)

    def remove_by_priortization(self):
        logging.info("Removing annotations by defined priortization")
        # n1 and n2 overlap
        # n1 is subset of n2
        # edge can be used for dereplication
        edges_to_analyze = self.G.get_edges(
            overlap=True, subset=True, avoid_derep=False
        )
        # remove n1 if lower priortization score
        to_remove = set()
        for n1, n2 in edges_to_analyze:
            n1_priortization = self.G[n1]["priortization"]
            n2_priortization = self.G[n2]["priortization"]
            if n2_priortization > n1_priortization:
                to_remove.add(n1)
        # remove units
        self.G.remove_nodes_from(to_remove)

    def remove_by_coverage(self):
        logging.info("Removing annotations by atom coverage")
        # n1 and n2 overlap
        # n1 is subset of n2
        # edge can be used for dereplication
        edges_to_analyze = self.G.get_edges(
            overlap=True, subset=True, avoid_derep=False
        )
        # remove by atom coverage
        to_remove = set()
        for n1, n2 in edges_to_analyze:
            n1_len = self.G[n1]["patt_len"]
            n2_len = self.G[n2]["patt_len"]
            if n2_len > n1_len:
                to_remove.add(n1)
        # remove units
        self.G.remove_nodes_from(to_remove)
