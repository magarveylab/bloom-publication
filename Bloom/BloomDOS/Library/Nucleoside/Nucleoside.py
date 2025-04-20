import os

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from Bloom.BloomDOS.Chemistry.ChemUtils import get_variable_atoms
from Bloom.BloomDOS.Chemistry.SmartsCompiler import get_smarts
from Bloom.BloomDOS.GeneralUtils import curdir
from Bloom.BloomDOS.Library.Hasher import get_hash_id

nucleoside_dir = os.path.join(curdir, "Library/Nucleoside/Tables")

#########################################################
# Main Function
#########################################################


class Nucleoside:

    def __init__(self):
        self.library = []
        self.load_data()
        self.create_library()
        self.add_hash_ids()

    def load_data(self):
        # load nucleoside units
        nucleoside_fp = "{}/nucleoside_units.csv".format(nucleoside_dir)
        self.units = pd.read_csv(nucleoside_fp).to_dict("records")

    def create_library(self):
        self.library = []
        for unit in tqdm(self.units, desc="Generating nucleoside units"):
            # find carbon atoms with SP2 - ignore smarts queries
            atoms_to_ignore = []
            mol = Chem.MolFromSmiles(unit["smiles"])
            for atom in mol.GetAtoms():
                if (
                    atom.GetAtomicNum() == 6
                    and str(atom.GetHybridization()) == "SP2"
                ):
                    atoms_to_ignore.append(atom.GetIdx())
            smarts = get_smarts(
                unit["smiles"],
                atoms_to_ignore=atoms_to_ignore,
                ignore_nitrogens=True,
            )
            variable_atoms = get_variable_atoms(unit["smiles"])
            self.library.append(
                {
                    "notation": unit["notation"],
                    "description": unit["description"],
                    "ignore_patt_idx": variable_atoms,
                    "module_decomposition": False,
                    "smarts": smarts,
                    "smiles": unit["smiles"],
                    "avoid_dereplication": True,
                    "enums": ["nucleoside"],
                }
            )

    def add_hash_ids(self):
        for unit in self.library:
            unit["hash_id"] = get_hash_id(unit)
