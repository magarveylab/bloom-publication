import ast
import json
import os

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from Bloom.BloomDOS.Chemistry.ChemUtils import get_variable_atoms
from Bloom.BloomDOS.Chemistry.SmartsCompiler import (
    get_simple_smarts,
    get_smarts,
)
from Bloom.BloomDOS.GeneralUtils import curdir
from Bloom.BloomDOS.Library.Hasher import get_hash_id

metabolite_dir = os.path.join(curdir, "Library/Metabolite/Tables")


class ExtractedUnits:

    def __init__(
        self,
        units_fp: str,
        enum: str,
        ring_size: int = 6,
        ring_atom_count: int = 10,
    ):
        self.library = []
        self.units_fp = units_fp
        self.enum = enum
        # paramters to determine if flexible unit should be created
        self.ring_size = ring_size
        self.ring_atom_count = ring_atom_count
        self.load_data()
        self.create_library()
        self.add_hash_ids()
        self.dereplicate_units()

    def load_data(self):
        # load extracted units from secondary pathways
        self.units = pd.read_csv(self.units_fp).to_dict("records")

    def create_library(self):
        for unit in tqdm(
            self.units, desc=f"Prepare extracted units for {self.enum}"
        ):
            mol = Chem.MolFromSmiles(unit["smiles"])
            atom_count = len(mol.GetAtoms())
            ring_atom_count = len(
                [a for a in mol.GetAtoms() if a.IsInRingSize(self.ring_size)]
            )
            # calculate smarts
            ignore_hydroxyls = True if atom_count > 4 else False
            smarts = get_smarts(
                unit["smiles"],
                do_charge_queries=True,
                ignore_hydroxyls=ignore_hydroxyls,
                ignore_phosphate_oxygens=False,
            )
            # find atoms to ignore
            variable_atoms = get_variable_atoms(unit["smiles"])
            original_enums = ast.literal_eval(unit["enum"])
            enums = original_enums + ["nonmodular", self.enum]
            avoid_dereplication = True
            self.library.append(
                {
                    "description": unit["description"],
                    "notation": unit["notation"],
                    "smiles": unit["smiles"],
                    "smarts": smarts,
                    "avoid_dereplication": avoid_dereplication,
                    "can_dereplicate": False,
                    "enums": enums,
                    "ignore_patt_idx": list(variable_atoms),
                }
            )
            # add flexible if ring paramter is met
            if ring_atom_count >= self.ring_atom_count:
                smarts = get_simple_smarts(unit["core_smiles"])
                variable_atoms = get_variable_atoms(smarts)
                enums = original_enums + ["nonmodular", "flexible", self.enum]
                avoid_dereplication = False
                self.library.append(
                    {
                        "description": unit["description"],
                        "notation": unit["notation"],
                        "smiles": unit["smiles"],
                        "smarts": smarts,
                        "avoid_dereplication": avoid_dereplication,
                        "can_dereplicate": False,
                        "enums": enums,
                        "ignore_patt_idx": list(variable_atoms),
                    }
                )

    def add_hash_ids(self):
        for unit in self.library:
            unit["hash_id"] = get_hash_id(unit)

    def dereplicate_units(self):
        self.library = list(
            {unit["hash_id"]: unit for unit in self.library}.values()
        )


class SecondaryMetabolite(ExtractedUnits):

    def __init__(self):
        super().__init__(
            units_fp=f"{metabolite_dir}/secondary_units.csv",
            enum="secondary_metabolite",
        )


class PrimaryMetabolite(ExtractedUnits):

    def __init__(self):
        super().__init__(
            units_fp=f"{metabolite_dir}/primary_units.csv",
            enum="primary_metabolite",
        )
