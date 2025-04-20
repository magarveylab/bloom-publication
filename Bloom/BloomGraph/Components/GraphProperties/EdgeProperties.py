from typing import Set, Tuple


class EdgeProperties:

    @property
    def edges(self) -> Set[Tuple[int]]:
        return self.G.edges

    @property
    def ambiguous_edges(self) -> Set[Tuple[int]]:
        ambiguous_nodes = self.ambiguous_nodes
        return set(e for e in self.edges if len(set(e) & ambiguous_nodes) > 0)

    @property
    def unambiguous_edges(self) -> Set[Tuple[int]]:
        return self.edges - self.ambiguous_edges

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def unambiguous_edge_count(self) -> int:
        return len(self.unambiguous_edges)

    @property
    def edge_data(self) -> Tuple[int, int, dict]:
        return self.G.edges(data=True)

    @property
    def reverse_edges(self) -> Set[Tuple[int]]:
        return set(e[::-1] for e in self.edges)

    @property
    def edges_in_mol(self) -> Set[Tuple[int]]:
        edges = set()
        for b in self.mol.GetBonds():
            a1 = b.GetBeginAtom().GetAtomMapNum()
            a2 = b.GetEndAtom().GetAtomMapNum()
            if a1 != a2 and a1 != 0 and a2 != 0:
                edges.add((a1, a2))
        return edges
