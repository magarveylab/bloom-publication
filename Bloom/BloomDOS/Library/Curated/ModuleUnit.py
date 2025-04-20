import ast
import os

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from Bloom.BloomDOS.Chemistry.ChemUtils import get_natural_smiles
from Bloom.BloomDOS.Chemistry.SmartsCompiler import get_smarts
from Bloom.BloomDOS.GeneralUtils import curdir
from Bloom.BloomDOS.Library.Hasher import get_hash_id

miscellaneous_dir = os.path.join(curdir, "Library/Curated/Tables")


class ModuleUnit:
    # these are modular units (and breakdown) curated by users
    def __init__(self):
        self.library = []
        self.load_data()
        self.create_library()
        self.add_hash_ids()

    def load_data(self):
        # load units
        curated_fp = "{}/modular_units.csv".format(miscellaneous_dir)
        self.units = pd.read_csv(curated_fp).to_dict("records")

    def create_library(self):
        ignore_notations = ["attacker", "reciever"]
        for u in tqdm(self.units, desc="Prepare curated modular units"):
            notation = u["module_name"]
            if notation in ignore_notations:
                continue
            # prepare description
            molecule = u["molecule"]
            unit_name = u["unit_name"]
            description = "{} | {} | {}".format(notation, unit_name, molecule)
            # prepare smiles
            smiles = get_natural_smiles(u["module_smiles"])
            mol = Chem.MolFromSmiles(smiles)
            # ignore atoms without map numbers
            ignore_atoms = [
                a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 0
            ]
            [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
            smiles = Chem.MolToSmiles(mol, canonical=False)
            # generate smarts pattern
            smarts = get_smarts(
                smiles,
                do_H_queries=True,
                do_hybridization_queries=True,
                do_heteroatom_n_queries=True,
                do_atom_n_queries=True,
                ignore_hydroxyls=False,
                consider_carboxylic=True,
                ignore_nitrogens=False,
                consider_amide=True,
                ignore_variable_groups=True,
            )
            # prepare enums
            module_types = ast.literal_eval(u["module_types"])
            substrates = ast.literal_eval(u["substrates"])
            reactions = ast.literal_eval(u["reactions"])
            auxillary_genes = ast.literal_eval(u["auxillary_genes"])
            if "nrps" in module_types:
                enums = ["aa", "curated_module"]
            elif "pks" in module_types:
                enums = ["pk", "curated_module"]
            else:
                enums = ["curated_module"]
            enums = enums + [unit_name]
            # cache unit
            self.library.append(
                {
                    "notation": notation,
                    "description": description,
                    "smarts": smarts,
                    "smiles": smiles,
                    "ignore_patt_idx": ignore_atoms,
                    "enums": enums,
                    "alpha_restriction_map": [],
                    "avoid_dereplication": True,
                    "variable_map": {},
                }
            )

    def add_hash_ids(self):
        for unit in self.library:
            unit["hash_id"] = get_hash_id(unit)
