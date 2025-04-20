import copy
from typing import Dict, List, Set, Tuple

import networkx as nx
from rdkit import Chem


class NetworkMethods:

    def bfs_tree(
        self, n: int, radius: int, remove_edges: Set[Tuple[int, int]] = set()
    ):
        if radius > 0:
            # create copy of graph - remove target egdes
            G = self.G.copy()
            G.remove_edges_from(remove_edges)
            return set(nx.bfs_tree(G, source=n, depth_limit=radius).nodes)
        else:
            return {n}

    def subgraph(self, nodes: Set[int], adjust_mol: bool = False):
        subgraph = self.EmptyGraph()
        # deep copy is needed so we can edit node meta data
        subgraph.G = copy.deepcopy(self.G.subgraph(nodes))
        if adjust_mol == True:
            new_smiles = self.extract_smiles(nodes)
            if Chem.MolFromSmiles(new_smiles) == None:
                subgraph.smiles = self.smiles
            else:
                subgraph.smiles = new_smiles
        else:
            subgraph.smiles = self.smiles
        return subgraph

    def subgraph_from_smiles(self, smiles: str):
        # pull node ids from smiles
        mol = Chem.MolFromSmiles(smiles)
        node_ids = set(a.GetAtomMapNum() for a in mol.GetAtoms()) & self.nodes
        if len(node_ids) == 0:
            return None
        node_exports_dict = self.node_exports_dict
        nodes = [node_exports_dict[n] for n in node_ids]
        return self.load_from_nodes(nodes=nodes, smiles=smiles)

    def subtract(self, nodes: Set[int], adjust_mol: bool = False):
        return self.subgraph(set(self.nodes) - nodes, adjust_mol=adjust_mol)

    def relabel_nodes(self, mapping: Dict[int, int]):
        self.G = nx.relabel_nodes(self.G, mapping)

    def split(self, adjust_mol: bool = True):
        return [
            self.subgraph(d["nodes"], adjust_mol=adjust_mol)
            for d in self.mol_disconnects
        ]

    def GetIsomorphismGraphMatcher(self, other) -> nx.isomorphism.GraphMatcher:
        # exact node matching function
        node_match = lambda a, b: True if a["labels"] == b["labels"] else False
        # check if the graph is isomorphic (same unit connectivity)
        return nx.isomorphism.GraphMatcher(
            self.G, other.G, node_match=node_match
        )

    def FindIsomorphicSubgraph(self, other) -> Dict[int, int]:
        gm = self.GetIsomorphismGraphMatcher(other)
        for r in gm.subgraph_isomorphisms_iter():
            return r
