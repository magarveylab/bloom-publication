import os

import pandas as pd
from tqdm import tqdm

from Bloom.BloomDOS.Chemistry.ChemUtils import get_variable_atoms
from Bloom.BloomDOS.Chemistry.Reaction import anomeric_clevage
from Bloom.BloomDOS.Chemistry.SmartsCompiler import get_smarts
from Bloom.BloomDOS.GeneralUtils import curdir
from Bloom.BloomDOS.Library.Hasher import get_hash_id

sugar_dir = os.path.join(curdir, "Library/Sugar/Tables")


class Sugar:

    def __init__(self):
        self.library = []
        self.load_data()
        self.create_library()
        self.add_sugars_with_stereochemistry()
        self.add_hash_ids()
        self.dereplicate_units()

    def load_data(self):
        # load sugars
        sugar_fp = "{}/sugars.csv".format(sugar_dir)
        self.sugars = pd.read_csv(sugar_fp).to_dict("records")

    def create_library(self):
        # calculate incorporated form for sugars
        for sugar in tqdm(self.sugars, desc="Generating sugar units"):
            products = anomeric_clevage.RunReactant(sugar["smiles"])
            if len(products) > 0:
                smiles = products[0]
            else:
                # if there is no anomeric site, incorporate the sugar as it is
                smiles = sugar["smiles"]
            # smarts
            variable_atoms = get_variable_atoms(smiles)
            smarts = get_smarts(
                smiles=smiles,
                do_H_queries=True,
                do_heteroatom_n_queries=True,
                do_atom_n_queries=True,
                limit_to_carbon_queries=False,
                consider_ring_membership=False,
                ignore_nitrogens=True,
                ignore_phosphate_oxygens=True,
            )
            self.library.append(
                {
                    "notation": sugar["notation"],
                    "description": sugar["description"],
                    "smiles": smiles,
                    "smarts": smarts,
                    "ignore_patt_idx": variable_atoms,
                    "enums": ["sugar"],
                    "avoid_dereplication": True,
                }
            )

    def add_sugars_with_stereochemistry(self):

        to_add = []
        for unit in self.library:
            if "@" in unit["smiles"]:
                new_unit = {k: v for k, v in unit.items()}
                new_smarts = get_smarts(
                    smiles=unit["smiles"],
                    stereochemistry=True,
                    do_H_queries=True,
                    do_heteroatom_n_queries=True,
                    do_atom_n_queries=True,
                    limit_to_carbon_queries=False,
                    consider_ring_membership=False,
                    ignore_nitrogens=True,
                    ignore_phosphate_oxygens=True,
                )
                new_unit["smarts"] = new_smarts
                new_unit["enums"] = unit["enums"] + ["stereochemistry"]
                to_add.append(new_unit)
        self.library = self.library + to_add

    def add_hash_ids(self):
        for unit in self.library:
            unit["hash_id"] = get_hash_id(unit)

    def dereplicate_units(self):
        self.library = list(
            {unit["hash_id"]: unit for unit in self.library}.values()
        )
