import ast
import os

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from Bloom.BloomDOS.Chemistry.ChemUtils import (
    get_natural_smiles,
    get_variable_atoms,
)
from Bloom.BloomDOS.Chemistry.SmartsCompiler import get_smarts
from Bloom.BloomDOS.GeneralUtils import curdir
from Bloom.BloomDOS.Library.Hasher import get_hash_id

known_smarts = {"expoxide_linker": "[*]1[#8H0D2z{0-2}][*]1"}

simple_smarts_notations = ["phosphate", "chlorine"]

tailoring_dir = os.path.join(curdir, "Library/Tailoring/Tables")


class Tailoring:

    def __init__(self):
        self.library = []
        self.load_data()
        self.create_library()
        self.add_hash_ids()

    def load_data(self):
        # load tailoring units
        tailoring_fp = "{}/tailoring.csv".format(tailoring_dir)
        self.tailoring = pd.read_csv(tailoring_fp).to_dict("records")
        for x in self.tailoring:
            x["ignore_patt_idx"] = ast.literal_eval(x["ignore_patt_idx"])
            x["avoid_dereplication"] = True

    def create_library(self):
        self.add_curated()
        self.add_phosphates()

    def add_curated(self):
        for unit in tqdm(self.tailoring, desc="Generate tailoring units"):
            smiles = get_natural_smiles(unit["smiles"])
            variable_map = {}
            # create smarts queries
            if unit["notation"] in simple_smarts_notations:
                smarts = get_smarts(
                    smiles,
                    do_H_queries=False,
                    do_hybridization_queries=False,
                    do_heteroatom_n_queries=False,
                    do_atom_n_queries=False,
                    do_charge_queries=False,
                    ignore_hydroxyls=True,
                    ignore_phosphate_oxygens=True,
                    ignore_nitrogens=False,
                    limit_to_carbon_queries=False,
                    consider_ring_membership=False,
                )
            elif unit["notation"] in known_smarts:
                smarts = known_smarts[unit["notation"]]
            else:
                linker_atoms = ast.literal_eval(unit["linker_atoms"])
                if len(linker_atoms) > 0:
                    linker_atom_enums = [
                        ast.literal_eval(i)
                        for i in unit["linker_atom_enums"].split(" | ")
                    ]
                    mol = Chem.MolFromSmiles(smiles)
                    map2idx = {
                        a.GetAtomMapNum(): a.GetIdx()
                        for a in mol.GetAtoms()
                        if a.GetAtomMapNum() > 0
                    }
                    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
                    smiles = Chem.MolToSmiles(mol, canonical=False)
                    variable_map = {
                        map2idx[x]: y
                        for x, y in zip(linker_atoms, linker_atom_enums)
                    }
                smarts = get_smarts(smiles=smiles, ignore_hydroxyls=True)
            variable_atoms = get_variable_atoms(unit["smiles"])
            ignore_patt_idx = list(
                set(unit["ignore_patt_idx"] + variable_atoms)
            )
            # unknown restrictions - are these units allowed to be adjacent to unknown
            if unit["apply_unknown_restrictions"]:
                unknown_restriction_map = variable_atoms
            else:
                unknown_restriction_map = []
            # cache
            self.library.append(
                {
                    "description": unit["description"],
                    "notation": unit["notation"],
                    "smiles": smiles,
                    "smarts": smarts,
                    "avoid_dereplication": False,
                    "enums": ["tailoring"],
                    "unknown_restriction_map": unknown_restriction_map,
                    "ignore_patt_idx": ignore_patt_idx,
                    "variable_map": variable_map,
                }
            )

    def add_phosphates(self):
        # general phosphate patterns
        # ambiguous bonds
        phosphates = [
            "[#8]~[#15](~[#8])(~[#8])~[#8]",
            "[#8]~[#15](~[#8])~[#8]",
            "[#8]~[#15]~[#8]",
        ]
        for smarts in phosphates:
            self.library.append(
                {
                    "notation": "phosphate",
                    "description": "phosphate",
                    "smarts": smarts,
                    "avoid_dereplication": True,
                    "enums": ["tailoring", "phosphate"],
                    "ignore_patt_idx": [],
                }
            )

    def add_hash_ids(self):
        for unit in self.library:
            unit["hash_id"] = get_hash_id(unit)
