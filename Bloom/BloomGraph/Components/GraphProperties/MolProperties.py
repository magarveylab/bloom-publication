from typing import List, Set, TypedDict

from rdkit import Chem


class MolDisconnectlDict(TypedDict):
    mol: Chem.Mol
    nodes: Set[int]


class MolProperties:

    @property
    def mol_without_map_nums(self) -> Chem.Mol:
        mol = Chem.Mol(self.mol)
        [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
        return mol

    @property
    def mol_with_unique_map_nums(self) -> Chem.Mol:
        mol = Chem.Mol(self.mol)
        [a.SetAtomMapNum(a.GetIdx() + 1) for a in mol.GetAtoms()]
        return mol

    @property
    def mol_disconnects(self) -> List[MolDisconnectlDict]:
        cache = []
        for s in self.smiles.split("."):
            m = Chem.MolFromSmiles(s)
            nodes = set(a.GetAtomMapNum() for a in m.GetAtoms()) - {0}
            cache.append({"mol": m, "nodes": nodes})
        return cache
