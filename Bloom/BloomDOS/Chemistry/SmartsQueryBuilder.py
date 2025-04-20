from typing import Dict, List

from rdkit import Chem
from rdkit.Chem.rdchem import Atom

from Bloom.BloomDOS.GeneralUtils import other_variable_atomic_nums


class SmartsQueryBuilder:

    def __init__(
        self,
        smiles: str,
        stereochemistry: bool = False,
        limit_to_carbon_queries: bool = False,
        do_H_queries: bool = False,
        ignore_CH3: bool = False,
        ignore_CH2: bool = False,
        ignore_CH: bool = False,
        do_hybridization_queries: bool = False,
        do_heteroatom_n_queries: bool = False,
        do_atom_n_queries: bool = False,
        do_charge_queries: bool = False,
        ignore_hydroxyls: bool = False,
        ignore_phosphate_oxygens: bool = False,
        consider_carboxylic: bool = False,
        ignore_nitrogens: bool = False,
        consider_amide: bool = False,
        ignore_thiols: bool = False,
        ignore_variable_groups: bool = True,
        consider_ring: bool = False,
        atoms_to_ignore: List[int] = [],
        add_variable_n_map: Dict[int, int] = {},
        ring_atoms_to_ignore: List[int] = [],
        variable_bond_queries: bool = False,
    ):
        # molecule
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(self.smiles)
        # variables to build queries
        self.stereochemistry = stereochemistry
        self.limit_to_carbon_queries = limit_to_carbon_queries
        self.do_H_queries = do_H_queries
        self.ignore_CH3 = ignore_CH3
        self.ignore_CH2 = ignore_CH2
        self.ignore_CH = ignore_CH
        self.do_hybridization_queries = do_hybridization_queries
        self.do_atom_n_queries = do_atom_n_queries
        self.do_heteroatom_n_queries = do_heteroatom_n_queries
        self.do_charge_queries = do_charge_queries
        self.ignore_hydroxyls = ignore_hydroxyls
        self.ignore_phosphate_oxygens = ignore_phosphate_oxygens
        self.consider_carboxylic = consider_carboxylic
        self.ignore_nitrogens = ignore_nitrogens
        self.consider_amide = consider_amide
        self.ignore_thiols = ignore_thiols
        self.ignore_variable_groups = ignore_variable_groups
        self.consider_ring = consider_ring
        self.atoms_to_annotate = set()
        self.atoms_to_ignore = set(atoms_to_ignore)
        self.ring_atoms_to_ignore = set(ring_atoms_to_ignore)
        self.add_variable_n_map = add_variable_n_map
        self.variable_bond_queries = variable_bond_queries
        # dictionaries to cache variables
        self.stereochemistry_map = {}
        self.atom_H_map = {}
        self.atom_hybrid_map = {}
        self.atom_n_map = {}
        self.atom_heteroatom_n_map = {}
        self.atom_charge_map = {}
        self.atom_variable_n_map = {}
        self.ring_membership_map = {}
        self.flagged_variable_atoms = {}
        # build queries
        self.prepare_ignore_atoms()
        if self.stereochemistry:
            self.prepare_stereochemistry_queries()
        if self.do_H_queries:
            self.prepare_H_queries()
        if self.do_hybridization_queries:
            self.prepare_hybridization_queries()
        if self.do_atom_n_queries:
            self.prepare_atom_n_queries()
        if self.do_heteroatom_n_queries:
            self.prepare_heteroatom_n_queries()
        if self.do_charge_queries:
            self.prepare_charge_queries()
        if self.consider_ring:
            self.prepare_ring_membership_queries()
        self.prepare_variable_n_queries()
        self.get_flagged_variable_atoms()

    def prepare_ignore_atoms(self):
        # atoms to force annotation
        if self.consider_carboxylic:
            matching_atoms = self.get_matching_atoms(
                smarts="[CX3](=O)[OX2H1]", index_to_ignore=[0, 1]
            )
            self.atoms_to_annotate.update(matching_atoms)
        if self.consider_amide:
            matching_atoms = self.get_matching_atoms(
                smarts="[NX3][CX3](=[OX1])[#6]", index_to_ignore=[1, 2, 3]
            )
            self.atoms_to_annotate.update(matching_atoms)
        # atoms to ignore
        if self.ignore_hydroxyls:
            matching_atoms = self.get_matching_atoms(
                smarts="[#6][OX2H]", index_to_ignore=[0]
            )
            self.atoms_to_ignore.update(matching_atoms)
        if self.ignore_phosphate_oxygens:
            matching_atoms = self.get_matching_atoms(
                smarts="[#15]-[#8D1]", index_to_ignore=[0]
            )
            self.atoms_to_ignore.update(matching_atoms)
        if self.ignore_nitrogens:
            matching_atoms = self.get_matching_atoms(smarts="[#7]")
            self.atoms_to_ignore.update(matching_atoms)
        if self.ignore_thiols:
            matching_atoms = self.get_matching_atoms(smarts="[#16X2H]")
            self.atoms_to_ignore.update(matching_atoms)
        if self.ignore_variable_groups:
            for atom in self.mol.GetAtoms():
                if (
                    atom.GetAtomicNum() == 0
                    or atom.GetAtomicNum() in other_variable_atomic_nums
                ):
                    self.atoms_to_ignore.add(atom.GetIdx())
        if self.limit_to_carbon_queries:
            for atom in self.mol.GetAtoms():
                if atom.GetAtomicNum() != 6:
                    self.atoms_to_ignore.add(atom.GetIdx())
        # remove any atoms to annotate from atoms to ignore
        self.atoms_to_ignore = self.atoms_to_ignore - self.atoms_to_annotate

    def get_matching_atoms(self, smarts: str, index_to_ignore: List[int] = []):
        matching_atoms = set()
        smarts = Chem.MolFromSmarts(smarts)
        matches = self.mol.GetSubstructMatches(smarts)
        for m in matches:
            # for certain patts (such as hydroxyl) - ignore certain indexes
            m = [a for idx, a in enumerate(m) if idx not in index_to_ignore]
            matching_atoms.update(m)
        return matching_atoms

    def consider_atom(self, atom: Atom):
        if atom.GetIdx() in self.atoms_to_ignore:
            return False
        else:
            return True

    def consider_ring_atom(self, atom: Atom):
        if atom.GetIdx() in self.ring_atoms_to_ignore:
            return False
        else:
            return True

    def prepare_stereochemistry_queries(self):
        for atom in self.mol.GetAtoms():
            if self.consider_atom(atom) is False:
                continue
            if str(atom.GetChiralTag()) == "CHI_TETRAHEDRAL_CCW":
                self.stereochemistry_map[atom.GetIdx()] = "@"
            elif str(atom.GetChiralTag()) == "CHI_TETRAHEDRAL_CW":
                self.stereochemistry_map[atom.GetIdx()] = "@@"

    def prepare_H_queries(self):
        for atom in self.mol.GetAtoms():
            if self.consider_atom(atom) == False:
                continue
            # skip different states of hydrogenated carbons
            H_count = atom.GetTotalNumHs()
            if self.ignore_CH3 and H_count == 3:
                continue
            if self.ignore_CH2 and H_count == 2:
                continue
            if self.ignore_CH and H_count == 1:
                continue
            # cache hydrogen information
            self.atom_H_map[atom.GetIdx()] = H_count

    def prepare_hybridization_queries(self):
        for atom in self.mol.GetAtoms():
            if self.consider_atom(atom) == False:
                continue
            # calculate hybridization state
            hybridization = str(atom.GetHybridization())
            # cache hybridization information
            self.atom_hybrid_map[atom.GetIdx()] = hybridization

    def prepare_atom_n_queries(self):
        for atom in self.mol.GetAtoms():
            if (
                self.consider_atom(atom) == False
                and self.variable_bond_queries is False
            ):
                continue
            # calculate number of neighbors
            n = len(atom.GetNeighbors())
            # cache neighbor information
            self.atom_n_map[atom.GetIdx()] = n

    def prepare_heteroatom_n_queries(self):
        for atom in self.mol.GetAtoms():
            if self.consider_atom(atom) == False:
                continue
            # calculate number of non-carbon neighbors
            n = len(
                [
                    a
                    for a in atom.GetNeighbors()
                    if a.GetAtomicNum() != 6
                    and a.GetAtomicNum() != 0
                    and a.GetAtomicNum() not in other_variable_atomic_nums
                ]
            )
            self.atom_heteroatom_n_map[atom.GetIdx()] = n

    def prepare_charge_queries(self):
        for atom in self.mol.GetAtoms():
            if self.consider_atom(atom) == False:
                continue
            charge = atom.GetFormalCharge()
            self.atom_charge_map[atom.GetIdx()] = charge

    def prepare_variable_n_queries(self):
        for atom in self.mol.GetAtoms():
            if self.consider_atom(atom) == False:
                continue
            # calculate number of variable neighbors
            n = len(
                [
                    a
                    for a in atom.GetNeighbors()
                    if a.GetAtomicNum() == 0
                    or a.GetAtomicNum() in other_variable_atomic_nums
                ]
            )
            self.atom_variable_n_map[atom.GetIdx()] = n

    def prepare_ring_membership_queries(self):
        for atom in self.mol.GetAtoms():
            if (
                self.consider_atom(atom) is False
                or self.consider_ring_atom(atom) is False
            ):
                continue
            self.ring_membership_map[atom.GetIdx()] = atom.IsInRing()

    def get_flagged_variable_atoms(self):
        for atom in self.mol.GetAtoms():
            a_num = atom.GetAtomicNum()
            if a_num in other_variable_atomic_nums:
                if a_num not in self.flagged_variable_atoms:
                    self.flagged_variable_atoms[a_num] = []
                self.flagged_variable_atoms[a_num].append(atom.GetIdx())
