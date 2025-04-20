import ast
import os

import pandas as pd
from tqdm import tqdm

from Bloom.BloomDOS.Chemistry.ChemUtils import (
    get_idx_with_map_num,
    get_variable_atoms,
    map_smiles,
)
from Bloom.BloomDOS.Chemistry.Reaction import ReactionSmarts
from Bloom.BloomDOS.Chemistry.SmartsCompiler import get_smarts
from Bloom.BloomDOS.GeneralUtils import curdir
from Bloom.BloomDOS.Library.Hasher import get_hash_id

miscellaneous_dir = os.path.join(curdir, "Library/Curated/Tables")

#########################################################
# Helper Functions
#########################################################


def get_misc_unit(smiles: str, u: dict, add_enums: list):
    i1 = get_idx_with_map_num(smiles, 3)
    i2 = get_idx_with_map_num(smiles, 4)
    i3 = get_idx_with_map_num(smiles, 5)
    variable_atoms = get_variable_atoms(smiles)
    ignore_patt_idx = list(set(i1 + i3 + variable_atoms))
    atoms_to_ignore = list(set(i2 + i3))
    # create smarts
    smarts = get_smarts(
        smiles,
        do_H_queries=u["do_H_queries"],
        do_hybridization_queries=u["do_hybridization_queries"],
        do_heteroatom_n_queries=u["do_heteroatom_n_queries"],
        do_atom_n_queries=u["do_atom_n_queries"],
        do_charge_queries=u["do_charge_queries"],
        ignore_hydroxyls=True,
        ignore_nitrogens=True,
        atoms_to_ignore=atoms_to_ignore,
    )
    enums = u["enums"]
    variable_enums = u["variable_enums"]
    if len(variable_enums) > 0:
        variable_map = {a: variable_enums for a in variable_atoms}
    else:
        variable_map = {}
    # cache
    return {
        "description": u["description"],
        "notation": u["notation"],
        "smiles": smiles,
        "smarts": smarts,
        "ignore_patt_idx": ignore_patt_idx,
        "enums": enums
        + ["miscellaneous", "starter", "pk", "pk_adj"]
        + add_enums,
        "avoid_dereplication": True,
        "variable_map": variable_map,
        "uniquify": u["uniquify"],
    }


#########################################################
# Main Function
#########################################################


class Miscellaneous:
    # these are units curated by users
    def __init__(self):
        self.library = []
        self.load_data()
        self.create_library()
        self.add_CoA()
        self.add_hash_ids()
        self.dereplicate_units()

    def load_data(self):
        # load miscellaneous units
        miscellaneous_fp = "{}/miscellaneous_units.csv".format(
            miscellaneous_dir
        )
        self.units = pd.read_csv(miscellaneous_fp).to_dict("records")
        for x in self.units:
            x["ignore_patt_idx"] = ast.literal_eval(x["ignore_patt_idx"])
            x["smarts_atoms_to_ignore"] = ast.literal_eval(
                x["smarts_atoms_to_ignore"]
            )
            x["enums"] = ast.literal_eval(x["enums"])
            x["variable_enums"] = ast.literal_eval(x["variable_enums"])
        # load reactions
        rxn_fp = "{}/miscellaneous_reactions.csv".format(miscellaneous_dir)
        df = pd.read_csv(rxn_fp)
        df["reaction_smarts"] = df["reaction_smarts"].apply(
            lambda x: ReactionSmarts(x)
        )
        self.rxn_dict = dict(zip(df.reaction_id, df.reaction_smarts))

    def create_library(self):
        for u in tqdm(self.units, desc="Prepare miscellaneous units"):
            overlap_flags = set(u["ignore_patt_idx"]) & set(
                u["smarts_atoms_to_ignore"]
            )
            ignore_patt_flags = set(u["ignore_patt_idx"]) - overlap_flags
            ignore_smarts_flags = (
                set(u["smarts_atoms_to_ignore"]) - overlap_flags
            )
            flag_atoms = {}
            flag_atoms.update({i: 5 for i in overlap_flags})
            flag_atoms.update({i: 3 for i in ignore_patt_flags})
            flag_atoms.update({i: 4 for i in ignore_smarts_flags})
            # flag atoms to ignore after smarts search
            if len(flag_atoms) > 0:
                smiles = map_smiles(u["smiles"], flag_atoms)
            else:
                smiles = u["smiles"]
            # calculate atoms to ignore
            self.library.append(get_misc_unit(smiles, u, []))
            # add derivatized units
            if u["derivatize"] == False:
                continue
            for rxn_id, rxn in self.rxn_dict.items():
                products = rxn.RecursiveRunReactant(smiles)
                for p_smiles in products:
                    self.library.append(
                        get_misc_unit(p_smiles, u, ["incorporated"])
                    )

    def add_CoA(self):

        smiles = "CC(COP(O)(O[PH](=O)(OCC1OC(C(C1O[PH](=O)(O)=O)O)N2C=NC3=C2NC=NC3=N)=O)=O)(C(C(NCCC(NCCS)=O)=O)O)C"
        smarts = "[#6]-[#6](-[#6]-[#8]-[#15](~[#8])(-[#8]-[#15](~[#8])(-[#8]-[#6]-[#6]1-[#8]-[#6](-[#6](-[#6]-1-[#8]-[#15](~[#8])(~[#8])~[#8])-[#8])-[#7]1~[#6]~[#7]~[#6]2~[#6]~1~[#7]~[#6]~[#7]~[#6]~2~[#7])~[#8])~[#8])(-[#6](-[#6](-[#7]-[#6]-[#6]-[#6](-[#7]-[#6]-[#6]-[#16])=[#8])=[#8])-[#8])-[#6]"

        out = {
            "description": "CoenzymeA",
            "notation": "CoA",
            "smiles": smiles,
            "smarts": smarts,
            "ignore_patt_idx": [],
            "enums": ["miscellaneous", "starter", "pk", "pk_adj", "confident"],
            "avoid_dereplication": True,
            "variable_map": {},
            "uniquify": True,
        }
        self.library.append(out)

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
