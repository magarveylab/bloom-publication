from typing import Dict, List

from rdkit import Chem


def get_natural_smiles(smiles):
    # render the smiles with rdkit to fix indexing
    # do not canonicalize
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=False)


def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return False if mol == None else True


def map_smiles(smiles: str, map_dict: Dict[int, int]) -> str:
    mol = Chem.MolFromSmiles(smiles)
    for atom_idx, atom_num in map_dict.items():
        mol.GetAtomWithIdx(atom_idx).SetAtomMapNum(atom_num)
    return Chem.MolToSmiles(mol)


def get_variable_atoms(smiles: str) -> List[int]:
    mol = Chem.MolFromSmarts(smiles)
    return [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0
    ]


def get_idx_with_map_num(smiles: str, map_num: int) -> List[int]:
    mol = Chem.MolFromSmiles(smiles)
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == map_num]


def get_unmapped_atom_idx(smiles: str) -> List[int]:
    mol = Chem.MolFromSmiles(smiles)
    return [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0
    ]


def update_unmapped_atoms(smiles: str, atom_num: int) -> str:
    mol = Chem.MolFromSmiles(smiles)
    [
        a.SetAtomMapNum(atom_num)
        for a in mol.GetAtoms()
        if a.GetAtomMapNum() == 0
    ]
    return Chem.MolToSmiles(mol)


def get_ring_system_count(mol: Chem.Mol, includeSpiro=False) -> int:
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ringAts = set(ring)
        nSystems = []
        for system in systems:
            nInCommon = len(ringAts.intersection(system))
            if nInCommon and (includeSpiro or nInCommon > 1):
                ringAts = ringAts.union(system)
            else:
                nSystems.append(system)
        nSystems.append(ringAts)
        systems = nSystems
    return len(systems)
