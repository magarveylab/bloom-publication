from rdkit import Chem
from typing import Dict

def get_nodes_to_atoms_from_smiles(mapped_smiles: str):
    mol = Chem.MolFromSmiles(mapped_smiles)
    nodes_to_atoms = {a.GetAtomMapNum(): set() for a in mol.GetAtoms() if a.GetAtomMapNum() != 0}
    for a in mol.GetAtoms():
        nodes_to_atoms[a.GetAtomMapNum()].add(a.GetIdx())
    return nodes_to_atoms

def get_atoms_to_colour_from_nodes_to_colour(mapped_smiles: str, nodes_to_colour: Dict[int, tuple]):
    mol = Chem.MolFromSmiles(mapped_smiles)
    return {a.GetIdx(): nodes_to_colour[a.GetAtomMapNum()] \
                for a in mol.GetAtoms() if a.GetAtomMapNum() in nodes_to_colour}