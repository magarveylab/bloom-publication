import os

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from Bloom.BloomDOS.Chemistry.SmartsQueryBuilder import SmartsQueryBuilder
from Bloom.BloomDOS.Chemistry.SmartsWriter import SmartsWriter
from Bloom.BloomDOS.GeneralUtils import curdir
from Bloom.BloomDOS.Library.Hasher import get_hash_id

fattyacid_dir = os.path.join(curdir, "Library/FattyAcid/Tables")


class FattyAcid:

    def __init__(self):
        self.library = []
        self.load_data()
        self.create_library()
        self.add_hash_ids()
        self.dereplicate_units()

    def load_data(self):
        # load fatty acids
        fattyacid_fp = "{}/fattyacids.csv".format(fattyacid_dir)
        self.fattyacids = pd.read_csv(fattyacid_fp).to_dict("records")

    def create_library(self):
        for unit in tqdm(self.fattyacids, desc="Generating Fatty Acids"):
            # build smarts
            smarts_query = SmartsQueryBuilder(
                unit["smiles"],
                do_H_queries=True,
                do_heteroatom_n_queries=True,
                do_atom_n_queries=True,
                ignore_hydroxyls=True,
                consider_ring=True,
            )
            # add heteroatom n to carbon atom in carbonyl
            mol = Chem.MolFromSmiles(unit["smiles"])
            carbon_atoms = []
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 6:
                    n = atom.GetNeighbors()
                    n_atom_types = [a.GetAtomicNum() for a in n]
                    if 0 in n_atom_types and 8 in n_atom_types:
                        carbon_atoms.append(atom.GetIdx())
            for c in carbon_atoms:
                smarts_query.atom_heteroatom_n_map[c] += 1
            # format smarts string
            smarts = SmartsWriter(
                unit["smiles"],
                atom_H_map=smarts_query.atom_H_map,
                atom_hybrid_map=smarts_query.atom_hybrid_map,
                atom_n_map=smarts_query.atom_n_map,
                atom_heteroatom_n_map=smarts_query.atom_heteroatom_n_map,
                atom_charge_map=smarts_query.atom_charge_map,
                atom_variable_n_map=smarts_query.atom_variable_n_map,
                ring_membership_map=smarts_query.ring_membership_map,
            )
            # cache
            self.library.append(
                {
                    "description": unit["description"],
                    "notation": unit["notation"],
                    "smiles": unit["smiles"],
                    "smarts": smarts.smarts,
                    "ignore_patt_idx": smarts.get_variable_atoms(),
                    "avoid_dereplication": False,
                    "enums": ["fattyacid"],
                }
            )

    def add_hash_ids(self):
        for unit in self.library:
            unit["hash_id"] = get_hash_id(unit)

    def dereplicate_units(self):
        keep = []
        hash_ids = []
        for unit in self.library:
            if unit["hash_id"] not in hash_ids:
                keep.append(unit)
                hash_ids.append(unit["hash_id"])
        self.library = [x for x in keep]
