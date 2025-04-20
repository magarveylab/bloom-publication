import ast
import os

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from Bloom.BloomDOS.Chemistry.SmartsCompiler import get_smarts
from Bloom.BloomDOS.GeneralUtils import curdir
from Bloom.BloomDOS.Library.Hasher import get_hash_id

# define dir where substrates/reactions can be found
pk_dir = os.path.join(curdir, "Library/Polyketide/Tables")


class AlphaChains:
    # these chains are used to annotate alpha carbons for polyketide detection
    # this ensures the units are mapped with consideration to directionality
    def __init__(self, max_length: int = 60):
        self.terminal_units = []
        self.library = []
        self.max_length = max_length
        self.load_data()
        self.create_library()
        self.add_hash_ids()

    def load_data(self):
        terminal_fp = "{}/alpha_terminals.csv".format(pk_dir)
        terminal_units = pd.read_csv(terminal_fp).to_dict("records")
        # create smarts
        for unit in terminal_units:
            atoms_to_ignore = [
                int(i)
                for i in ast.literal_eval(unit["smarts_atoms_to_ignore"])
            ]
            smarts = get_smarts(
                unit["smiles"],
                do_H_queries=True,
                do_hybridization_queries=True,
                do_heteroatom_n_queries=True,
                do_atom_n_queries=True,
                do_charge_queries=True,
                ignore_hydroxyls=False,
                ignore_nitrogens=False,
                ignore_thiols=False,
                atoms_to_ignore=atoms_to_ignore,
            )
            self.terminal_units.append(
                {"notation": unit["notation"], "smarts": smarts}
            )

    def create_library(self):
        alpha_carbon = "[#6:1]-,=,:"
        beta_carbon = "[#6]-,=,:"
        for unit in tqdm(self.terminal_units, desc="Creating Alpha Chains"):
            smarts = unit["smarts"]
            smarts = smarts.replace("[*]-", "")
            add_alpha = False
            for n in range(self.max_length):
                # construct smarts
                notation = "C{}-{}".format(n, unit["notation"])
                if add_alpha == True:
                    smarts = "{}{}".format(alpha_carbon, smarts)
                    add_alpha = False
                else:
                    smarts = "{}{}".format(beta_carbon, smarts)
                    add_alpha = True
                # construct ignore patt idx
                alpha_carbons = []
                mol = Chem.MolFromSmarts(smarts)
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() == 1:
                        alpha_carbons.append(atom.GetIdx())
                self.library.append(
                    {
                        "notation": notation,
                        "description": notation,
                        "smarts": smarts,
                        "alpha_restriction_map": list(alpha_carbons),
                    }
                )

    def add_hash_ids(self):
        for unit in self.library:
            unit["hash_id"] = get_hash_id(unit)
