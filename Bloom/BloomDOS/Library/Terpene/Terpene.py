import os

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from Bloom.BloomDOS.Chemistry.Reaction import ReactionSmarts
from Bloom.BloomDOS.Chemistry.SmartsCompiler import get_smarts
from Bloom.BloomDOS.GeneralUtils import curdir
from Bloom.BloomDOS.Library.Hasher import get_hash_id

terpene_dir = os.path.join(curdir, "Library/Terpene/Tables")


class Terpene:

    def __init__(self):

        self.n_polymers = 20
        self.n_hydroxylations = 2

        self.library = []
        self.load_data()
        self.create_polymers()
        self.incorporate_polymers()
        self.hydroxylate_polymers()
        self.create_library()
        self.add_hash_ids()

    def load_data(self):
        units_fp = "{}/monomer_patterns.csv".format(terpene_dir)
        self.monomers = pd.read_csv(units_fp).to_dict("records")
        self.monomers = {x["notation"]: x["smiles"] for x in self.monomers}

        reactions_fp = "{}/terpene_reactions.csv".format(terpene_dir)
        self.reactions = pd.read_csv(reactions_fp).to_dict("records")
        self.reactions = {x["reaction"]: x["smarts"] for x in self.reactions}

    def create_polymers(self):
        rxn_smarts = self.reactions.get("isoprene_polymerization")
        rxn = ReactionSmarts(rxn_smarts)

        self.terpene_polymers = []
        reactant = self.monomers.get("isoprene")
        for i in range(1, self.n_polymers + 1):
            reactant = rxn.RunReactant(reactant)
            reactant = Chem.MolToSmiles(Chem.MolFromSmiles(reactant[0]))
            self.terpene_polymers.append((reactant, i + 1))
        self.terpene_polymers = list(set(self.terpene_polymers))

    def incorporate_polymers(self):

        rxn_1 = ReactionSmarts(self.reactions.get("terminal_incorporation_1"))
        rxn_2 = ReactionSmarts(self.reactions.get("terminal_incorporation_2"))

        incorporated_polymers = []
        for p, i in self.terpene_polymers:
            prod = rxn_1.RunReactant(p)
            prod = Chem.MolToSmiles(Chem.MolFromSmiles(prod[0]))
            incorporated_polymers.append((prod, i))
            prod = rxn_2.RunReactant(prod)
            prod = Chem.MolToSmiles(Chem.MolFromSmiles(prod[0]))
            incorporated_polymers.append((prod, i))

        self.terpene_polymers = list(
            set(self.terpene_polymers + incorporated_polymers)
        )

    def hydroxylate_polymers(self):

        rxn = ReactionSmarts(self.reactions.get("hydroxylation"))

        hydroxylated_polymers = []
        for reactant, n in tqdm(
            self.terpene_polymers, desc="Hydroyxlating terpene polymers"
        ):
            for i in range(1, self.n_hydroxylations + 1):
                reactant = rxn.RunReactant(reactant)
                reactant = Chem.MolToSmiles(Chem.MolFromSmiles(reactant[0]))
                hydroxylated_polymers.append((reactant, n))
        self.terpene_polymers = self.terpene_polymers + hydroxylated_polymers

    def create_library(self):

        for smiles, i in tqdm(
            self.terpene_polymers, desc="Creating terpene library"
        ):

            smarts = get_smarts(smiles)

            self.library.append(
                {
                    "notation": "isoprene-{}".format(i),
                    "description": "isoprene-{}".format(i),
                    "smiles": smiles,
                    "smarts": smarts,
                    "enums": ["terpene"],
                    "avoid_dereplication": False,
                    "can_dereplicate": True,
                    "ignore_patt_idx": [],
                }
            )

    def add_hash_ids(self):
        for unit in self.library:
            unit["hash_id"] = get_hash_id(unit)
