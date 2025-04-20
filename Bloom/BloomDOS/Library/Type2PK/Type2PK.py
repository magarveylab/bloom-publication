import os

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from Bloom.BloomDOS.Chemistry.ChemUtils import get_natural_smiles
from Bloom.BloomDOS.GeneralUtils import curdir
from Bloom.BloomDOS.Library.Hasher import get_hash_id

type2pk_dir = os.path.join(curdir, "Library/Type2PK/Tables")


class Type2PK:
    # curated type 2 polyketide scaffolds
    def __init__(self):
        self.library = []
        self.load_data()
        self.create_library()
        self.add_hash_ids()

    def load_data(self):
        # load units
        type2pk_fp = "{}/type2pk.csv".format(type2pk_dir)
        self.units = pd.read_csv(type2pk_fp).to_dict("records")

    def create_library(self):
        for u in tqdm(self.units, desc="Prepare Type2PK scaffolds"):
            notation = u["notation"]
            description = u["notation"]
            smiles = get_natural_smiles(u["smiles"])
            mol = Chem.MolFromSmiles(smiles)
            ignore_patt_idx = [
                a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0
            ]
            smarts = Chem.MolToSmarts(mol)
            self.library.append(
                {
                    "description": description,
                    "notation": notation,
                    "smiles": smiles,
                    "smarts": smarts,
                    "atoms_to_ignore": [],
                    "avoid_dereplication": True,
                    "can_dereplicate": False,
                    "enums": ["type2pk"],
                    "ignore_patt_idx": ignore_patt_idx,
                }
            )

    def add_hash_ids(self):
        for unit in self.library:
            unit["hash_id"] = get_hash_id(unit)
