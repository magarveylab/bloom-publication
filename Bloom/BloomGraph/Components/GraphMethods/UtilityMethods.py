import copy
from typing import Dict, Set

import networkx as nx


class UtilityMethods:

    def __getitem__(self, key) -> dict:
        return self.G.nodes[key]

    def __contains__(self, name) -> bool:
        # check if unit is present
        return True if name in self.top_labels else False

    def __len__(self) -> Set[int]:
        return len(self.nodes)

    def __add__(self, other):
        # new graph
        new = self.EmptyGraph()
        # merge graphs (x + y)
        if len(self.nodes & other.nodes) == 0:
            new.G = nx.union(self.G, other.G)
        else:
            # make sure node starts with 1 and not 0 for mapped smiles
            new_G = nx.disjoint_union(self.G, other.G)
            mapping = {n: n + 1 for n in new_G.nodes}
            new.G = nx.relabel_nodes(new_G, mapping)
        # combine smiles
        new_smiles = [self.smiles, other.smiles]
        new_smiles = [s for s in new_smiles if s != None]
        if len(new_smiles) > 0:
            new.smiles = ".".join(new_smiles)
        return new

    def __radd__(self, other):
        # recursive add with "sum"
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __copy__(self):
        new = self.EmptyGraph()
        new.G = self.G
        new.smiles = self.smiles
        new.meta = self.meta
        return new

    def __eq__(self, other) -> bool:
        # check if graphs are equal
        if isinstance(other, self.ClassType):
            # check if the graph is the same size
            if self.__len__() == len(other):
                # check if the graph has the same composition
                if set(self.all_labels) == set(other.all_labels):
                    # exact node matching function
                    node_match = lambda a, b: (
                        True if a["labels"] == b["labels"] else False
                    )
                    # check if the graph is isomorphic (same unit connectivity)
                    gm = self.GetIsomorphismGraphMatcher(other)
                    if gm.subgraph_is_isomorphic():
                        return True
                    else:
                        return False
                else:
                    False
            else:
                False
        else:
            return False

    def issubset(self, nodes: Set[int]) -> bool:
        return self.nodes.issubset(nodes)

    def contains(self, nodes: Set[int]) -> bool:
        return True if len(self.nodes & nodes) > 0 else False

    def get_atoms_from_nodes(self, nodes: Set[int]) -> Set[int]:
        return {a for n in nodes for a in self[n]["atoms"]}

    def relabel(self, mapping: Dict[int, int]):
        new = copy.deepcopy(self)
        # adjust unit graph
        new.G = nx.relabel_nodes(new.G, mapping)
        # adjust smiles
        for a in new.mol.GetAtoms():
            a.SetAtomMapNum(mapping.get(a.GetAtomMapNum(), 0))
        return new
