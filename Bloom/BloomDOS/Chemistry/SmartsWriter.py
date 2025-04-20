import re
from typing import Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem.rdchem import Atom

from Bloom.BloomDOS.GeneralUtils import other_variable_atomic_nums


class SmartsWriter:

    def __init__(
        self,
        smiles: str,
        stereochemistry_map: Dict[int, str] = {},
        atom_H_map: Dict[int, int] = {},
        atom_hybrid_map: Dict[int, int] = {},
        atom_n_map: Dict[int, int] = {},
        atom_heteroatom_n_map: Dict[int, int] = {},
        atom_charge_map: Dict[int, int] = {},
        atom_variable_n_map: Dict[int, int] = {},
        add_variable_n_map: Dict[int, int] = {},
        atom_restriction_map: Dict[int, List[str]] = {},
        ring_membership_map: Dict[int, bool] = {},
        bond_map: Dict[Tuple[int, int], str] = {},
    ):
        # smarts and mol
        self.smiles = smiles
        self.atoms_to_queries = {}
        self.atoms_to_str = {}
        self.bonds_to_str = {}
        # queries
        self.stereochemistry_map = stereochemistry_map
        self.atom_H_map = atom_H_map
        self.atom_n_map = atom_n_map
        self.atom_hybrid_map = atom_hybrid_map
        self.atom_heteroatom_n_map = atom_heteroatom_n_map
        self.atom_charge_map = atom_charge_map
        self.atom_variable_n_map = atom_variable_n_map
        self.add_variable_n_map = add_variable_n_map
        self.atom_restriction_map = atom_restriction_map
        self.ring_membership_map = ring_membership_map
        self.bond_map = bond_map
        # calculate search piece
        self.load_mol()
        self.load_atomic_queries()
        self.load_smarts_template()
        self.create_smarts()

    def load_mol(self):
        # erase existing map numbers
        # we are going to use atom map index to map queries
        self.mol = Chem.MolFromSmiles(self.smiles)
        for atom in self.mol.GetAtoms():
            atom.SetNumExplicitHs(0)
            atom.SetAtomMapNum(0)
            atom.SetAtomMapNum(atom.GetIdx() + 1)

    def load_atomic_queries(self):
        # map atom indexes to queries
        # atom flag is 1 greater than atom_idx to avoid atom map number of 0
        for atom in self.mol.GetAtoms():
            atom_flag = atom.GetIdx() + 1
            self.atoms_to_queries[atom_flag] = {}
            self.get_custom_queries(atom, atom_flag)

    def load_smarts_template(self):
        self.smarts_template = Chem.MolToSmarts(self.mol)
        # trace atoms
        search_funct = re.compile("\[(.*?)\]")
        for atom_flag, r in enumerate(
            search_funct.findall(self.smarts_template), 1
        ):
            original_term = "[{}]".format(r)
            new_term = "[{}]".format(atom_flag)
            self.atoms_to_str[atom_flag] = new_term
            self.smarts_template = self.smarts_template.replace(
                original_term, new_term
            )
        # trace bonds
        search_funct = re.compile("\](.*?)\[")
        bonds = search_funct.findall(self.smarts_template)
        last_bond = self.smarts_template[
            self.smarts_template.rindex("]") + 1 :
        ]
        if last_bond != "":
            bonds.append(last_bond)
        for bond, bond_term in zip(self.mol.GetBonds(), bonds):
            b = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            diff = b[1] - b[0]
            if diff == 1:
                new_term = (
                    "[{}]".format(bond.GetBeginAtomIdx() + 1)
                    + "{}"
                    + "[{}]".format(bond.GetEndAtomIdx() + 1)
                )
            elif diff > 1:
                new_term = "{}" + "[{}]".format(bond.GetEndAtomIdx() + 1)
            else:
                new_term = "[{}]".format(bond.GetBeginAtomIdx() + 1) + "{}"
            self.bonds_to_str[tuple(sorted(b))] = {
                "root": new_term,
                "bond": bond_term,
            }

    def get_custom_queries(self, atom: Atom, atom_flag: int):
        # atom id is used to pull queries
        atom_idx = atom.GetIdx()
        # get variable n state
        variable_n_state = self.atom_variable_n_map.get(atom_idx, 0)
        # get explicitly-defined n state (only for flexible units)
        add_variable_n_state = self.add_variable_n_map.get(atom_idx, 0)
        # define stereochemistry
        stereo = self.stereochemistry_map.get(atom_idx)
        if stereo is not None:
            self.atoms_to_queries[atom_flag]["stereo_query"] = stereo
        # define explicit hydrogens
        H_state = self.atom_H_map.get(atom_idx)
        if H_state != None:
            hygroden_query = self.get_hydrogen_query(
                H_state, add_variable_n_state
            )
            self.atoms_to_queries[atom_flag]["hydrogen_query"] = hygroden_query
        # define atom neighbors
        atom_n = self.atom_n_map.get(atom_idx)
        if atom_n != None:
            range_query = self.get_range_query(
                atom_n, variable_n_state, add_variable_n_state
            )
            self.atoms_to_queries[atom_flag]["range_query"] = range_query
        # define heteroatom neighbors
        heteroatom_n = self.atom_heteroatom_n_map.get(atom_idx)
        if heteroatom_n != None:
            hetero_n_query = self.get_heteroatom_query(
                heteroatom_n, variable_n_state, add_variable_n_state
            )
            self.atoms_to_queries[atom_flag]["hetero_n_query"] = hetero_n_query
        # define hybridization
        # note a hybridization query wont be made if there are variable atoms attached
        hybridization_state = self.atom_hybrid_map.get(atom_idx)
        if hybridization_state != None and variable_n_state == 0:
            hybrid_query = self.get_hybridization_query(hybridization_state)
            self.atoms_to_queries[atom_flag]["hybrid_query"] = hybrid_query
        # define charge
        atom_charge = self.atom_charge_map.get(atom_idx)
        if atom_charge != None:
            charge_query = self.get_charge_query(atom_charge)
            self.atoms_to_queries[atom_flag]["charge_query"] = charge_query
        # atom type restriction
        atom_restriction = self.atom_restriction_map.get(atom_idx)
        if atom_restriction != None:
            atom_restriction = self.get_atom_restriction_query(
                atom_restriction
            )
            self.atoms_to_queries[atom_flag][
                "atom_restriction"
            ] = atom_restriction
        # atom ring membership
        ring_membership = self.ring_membership_map.get(atom_idx)
        if ring_membership != None:
            ring_membership = self.get_ring_membership_query(ring_membership)
            self.atoms_to_queries[atom_flag][
                "ring_membership"
            ] = ring_membership

    def create_smarts(self):
        self.smarts = str(self.smarts_template)

        # add bond terms
        for bond, bond_term in self.bond_map.items():
            bond = tuple(sorted(bond))
            if bond in self.bonds_to_str:
                old_term = self.bonds_to_str[bond]["root"].format(
                    self.bonds_to_str[bond]["bond"]
                )
                new_bond_term = self.bonds_to_str[bond]["bond"].replace(
                    "-", "*"
                )
                new_bond_term = new_bond_term.replace("=", "*")
                new_bond_term = new_bond_term.replace("/", "*")
                new_bond_term = new_bond_term + " "
                new_bond_term = new_bond_term.replace("\ ", "*")
                new_bond_term = new_bond_term.replace(" ", "")
                new_bond_term = new_bond_term.replace("\\", "*")
                new_bond_term = new_bond_term.replace("#", "*")
                new_bond_term = new_bond_term.replace(":", "*")
                new_bond_term = new_bond_term.replace("*", bond_term)
                new_term = self.bonds_to_str[bond]["root"].format(
                    new_bond_term
                )
                self.smarts = self.smarts.replace(old_term, new_term)

        # add atomic terms
        for atom in self.mol.GetAtoms():
            atom_flag = atom.GetAtomMapNum()
            atomic_num = atom.GetAtomicNum()

            queries = [
                i
                for i in self.atoms_to_queries[atom_flag].values()
                if i != None
            ]
            if atomic_num == 0:
                term = "*"
            elif atomic_num in other_variable_atomic_nums:
                term = other_variable_atomic_nums[atomic_num]["replace"]
            else:
                term = "#{}".format(atomic_num)
            # add queries
            if len(queries) > 0:
                term = term + "".join(queries)
            # add square brackets
            term = "[{}]".format(term)
            self.smarts = self.smarts.replace(
                self.atoms_to_str[atom_flag], term
            )

    def get_variable_atoms(self):
        mol = Chem.MolFromSmarts(self.smarts)
        return [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 0
        ]

    @staticmethod
    def get_hybridization_query(hybridization_state: str):
        # determine hybridization state
        if hybridization_state == "S":
            return "^0"
        elif hybridization_state == "SP":
            return "^1"
        elif hybridization_state == "SP2":
            return "^2"
        elif hybridization_state == "SP3":
            return "^3"
        elif hybridization_state == "SP3D":
            return "^4"
        elif hybridization_state == "SP3D2":
            return "^5"
        else:
            return None

    @staticmethod
    def get_range_query(atom_n: int, variable_n: int, add_variable_n: int):
        if variable_n == 0:
            return "D" + str(atom_n)
        else:
            return (
                "D{"
                + str(atom_n)
                + "-"
                + str(atom_n + variable_n + add_variable_n)
                + "}"
            )

    @staticmethod
    def get_heteroatom_query(
        heteroatom_n: int, variable_n: int, add_variable_n: int
    ):
        if variable_n == 0:
            return "z" + str(heteroatom_n)
        else:
            return (
                "z{"
                + str(heteroatom_n)
                + "-"
                + str(heteroatom_n + variable_n + add_variable_n)
                + "}"
            )

    @staticmethod
    def get_hydrogen_query(h_state: int, add_variable_n_state: int):
        if add_variable_n_state == 0:
            return "H{}".format(h_state)
        else:
            min_H = (
                0
                if h_state - add_variable_n_state < 0
                else h_state - add_variable_n_state
            )
            h_states = ["H{}".format(i) for i in range(min_H, h_state + 1)]
            return ",".join(h_states)

    @staticmethod
    def get_charge_query(charge: int):
        return "+{}".format(charge) if charge >= 0 else str(charge)

    @staticmethod
    def get_atom_restriction_query(restriction: List[str]):
        return "!" + ";!".join(restriction)

    @staticmethod
    def get_ring_membership_query(ring_membership: bool):
        return "!R0" if ring_membership is True else "R0"
