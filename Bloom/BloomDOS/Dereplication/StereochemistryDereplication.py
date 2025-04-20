from Bloom.BloomDOS.Graph.AnnotationGraph import AnnotationGraph


class StereochemistryDereplication:

    def __init__(self, G: AnnotationGraph):
        self.G = G

    def remove_sugars(self):
        # remove stereoless versions of sugars
        sugar_nodes = self.G.get_nodes(enums={"sugar"})
        stereo_nodes = self.G.get_nodes(enums={"stereochemistry"})
        sugar_with_stereo_nodes = sugar_nodes & stereo_nodes
        sugar_without_stereo_nodes = sugar_nodes - stereo_nodes
        if len(sugar_with_stereo_nodes) > 0:
            edges = self.G.get_edges(overlap=True, exact=True)
            subgraph = self.G.edge_subgraph(edges)
            to_examine = set(subgraph.nodes) & sugar_without_stereo_nodes
            to_remove = set()
            for n in to_examine:
                # find units mapping to same region
                for m in subgraph.neighbors(n):
                    if m in sugar_with_stereo_nodes:
                        to_remove.add(n)
                        break
            # remove units
            self.G.remove_nodes_from(to_remove)
