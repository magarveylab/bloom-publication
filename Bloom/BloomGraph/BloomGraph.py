from typing import Optional

import networkx as nx
import xxhash
from rdkit import Chem

from Bloom.BloomGraph.Components.GraphMethods import (
    AtomMethods,
    ChainMethods,
    DataMethods,
    ExportMethods,
    LoadMethods,
    ModuleGraphMethods,
    NetworkMethods,
    SubsearchMethods,
    UtilityMethods,
    VisualizationMethods,
)
from Bloom.BloomGraph.Components.GraphProperties import (
    AtomProperties,
    BondProperties,
    EdgeProperties,
    EnumProperties,
    ExportProperties,
    LabelProperties,
    ModuleProperties,
    MolProperties,
    NetworkProperties,
    NodeProperties,
    PolyketideProperties,
    TableProperties,
)


class BloomGraph(
    # load all properties
    AtomProperties.AtomProperties,
    BondProperties.BondProperties,
    EdgeProperties.EdgeProperties,
    EnumProperties.EnumProperties,
    ExportProperties.ExportProperties,
    LabelProperties.LabelProperties,
    MolProperties.MolProperties,
    NetworkProperties.NetworkProperties,
    NodeProperties.NodeProperties,
    TableProperties.TableProperties,
    ModuleProperties.ModuleProperties,
    PolyketideProperties.PolyketideProperties,
    # load all methods
    AtomMethods.AtomMethods,
    DataMethods.DataMethods,
    ExportMethods.ExportMethods,
    LoadMethods.LoadMethods,
    NetworkMethods.NetworkMethods,
    UtilityMethods.UtilityMethods,
    VisualizationMethods.VisualizationMethods,
    ModuleGraphMethods.ModuleGraphMethods,
    ChainMethods.ChainMethods,
    SubsearchMethods.SubsearchMethods,
):

    def __init__(self):
        super().__init__()
        # unit graph (updated by adding nodes and edges)
        self._G = nx.Graph()
        # mol graph
        # special update methods triggered when setting smiles or mol
        self._mol = None
        # meta (track user defined properties)
        self.meta = {}
        self.enum_to_nodes = {}

    @property
    def graph_type(self) -> Optional[str]:
        # properties exist if data is pulled from cactus database
        if self.meta.get("cluster_id") != None:
            return "cluster"
        elif self.meta.get("smallmolecule_id") != None:
            return "smallmolecule"
        else:
            return None

    @property
    def graph_id(self) -> Optional[int]:
        if self.meta.get("cluster_id") != None:
            return self.meta["cluster_id"]
        elif self.meta.get("smallmolecule_id") != None:
            return self.meta["smallmolecule_id"]
        else:
            return None

    @property
    def G(self) -> nx.Graph:
        return self._G

    @G.setter
    def G(self, g: nx.Graph):
        self._G = g
        if self.mol != None:
            self.add_atomic_data_to_graph()

    @property
    def smiles(self) -> Optional[str]:
        return (
            Chem.MolToSmiles(self.mol, canonical=False)
            if self.mol != None
            else None
        )

    @smiles.setter
    def smiles(self, s: str):
        self.mol = Chem.MolFromSmiles(s)

    @property
    def hash_id(self) -> int:
        smiles = self.smiles  # not this is a mapped smiles
        return xxhash.xxh32(smiles).intdigest() if smiles != None else None

    @property
    def mol(self) -> Optional[Chem.Mol]:
        return self._mol

    @mol.setter
    def mol(self, m: Chem.Mol):
        self._mol = m
        self.add_atomic_data_to_graph()

    @classmethod
    def EmptyGraph(cls):
        return cls()

    @property
    def ClassType(self):
        return BloomGraph
