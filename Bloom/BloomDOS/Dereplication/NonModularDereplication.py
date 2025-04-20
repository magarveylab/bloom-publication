import logging

from Bloom.BloomDOS.Graph.AnnotationGraph import AnnotationGraph


class NonModularDereplication:

    def __init__(self, G: AnnotationGraph):
        self.G = G

    def remove_nonmodular_by_exact_coverage(self):
        # enums that cannot remove non-modular units by exact dereplication
        skip_enums = {"hybrid", "variable_restriction"}
        logging.info(
            "Removing nonmodular annotations exactly mapping to another unit"
        )
        nonmodular_nodes = self.G.get_nodes(enums={"nonmodular"})
        edges = self.G.get_edges(overlap=True, exact=True)
        subgraph = self.G.edge_subgraph(edges)
        to_examine = set(subgraph.nodes) & nonmodular_nodes
        to_remove = set()
        for n in to_examine:
            # find units mapping to same region
            for m in subgraph.neighbors(n):
                # if there is a unit that is not nonmodular that maps to the same
                # region -> then remove nonmodlar unit
                if (
                    "nonmodular" not in self.G[m]["enums"]
                    and len(self.G[m]["enums"] & skip_enums) == 0
                ):
                    to_remove.add(n)
                    break
        # remove units
        self.G.remove_nodes_from(to_remove)

    def remove_nonmodular_by_combinatoric_coverage(self):
        logging.info(
            "Removing nonmodular annotations mapping to unit combinatorics"
        )
        nonmodular_nodes = self.G.get_nodes(enums={"nonmodular"})
        enums_to_consider = {
            "sugar",
            "confident",
            "fattyacid",
            "phosphate",
            "terpene",
        }
        # capture subset associations
        edges = self.G.get_edges(overlap=True, subset=True)
        subgraph = self.G.edge_subgraph(edges)
        to_examine = set(subgraph.nodes) & nonmodular_nodes
        to_remove = set()
        for n in to_examine:
            n_atoms = set(self.G[n]["patt"])
            # find units subset to nonmodular region
            for m in subgraph.neighbors(n):
                if "nonmodular" in self.G[m]["enums"]:
                    continue
                if len(self.G[m]["enums"] & enums_to_consider) > 0:
                    # subtract subset atoms from nonmodular units
                    n_atoms = n_atoms - self.G[m]["patt"]
            # this method is not truly combinatorics
            # as subset annotations can overlap
            # but it is ok since we are only considering enums such as sugars
            # (which do not overlap with each other)
            if len(n_atoms) == 0:
                to_remove.add(n)
        # remove units
        self.G.remove_nodes_from(to_remove)
