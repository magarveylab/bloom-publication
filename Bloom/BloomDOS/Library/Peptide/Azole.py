import ast
import os
from typing import List, Optional, Tuple

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from Bloom.BloomDOS.Chemistry.Reaction import ReactionSmarts
from Bloom.BloomDOS.Chemistry.SmartsCompiler import get_smarts
from Bloom.BloomDOS.DataStructs import EnhancedSmartsDict
from Bloom.BloomDOS.GeneralUtils import curdir
from Bloom.BloomDOS.Library.Hasher import get_hash_id
from Bloom.BloomDOS.Library.Polyketide.PolyketideModule import PolyketideModule

# define dir where substrates/reactions can be found
aa_dir = os.path.join(curdir, "Library/Peptide/Tables")

#########################################################
# Helper Function
#########################################################


def get_azole_component(
    rxn: ReactionSmarts,
    azole_type: str,
    smiles: str,
    azole_patt: Chem.Mol,
    keep_idx: List[int],
    azole_bonds: List[Tuple[int, int]],
    description: str,
    notation: str,
    enums: List[str],
    substrate_family: str,
    substrate_id: str,
    aa_id: str,
    domain_id: Optional[str] = None,
    domain_subclass_id: Optional[str] = None,
) -> List[EnhancedSmartsDict]:
    to_add = []
    products = rxn.RunReactant(smiles)
    for p_smiles in products:
        mol = Chem.MolFromSmiles(p_smiles)
        hit = mol.GetSubstructMatch(azole_patt)
        # these index do not correspond to orginal amino acid
        ignore_patt_idx = [
            atom for idx, atom in enumerate(hit) if idx not in keep_idx
        ]
        # indexes that correspond to azole bonds
        bond_map = {(hit[i[0]], hit[i[1]]): "-,=,:" for i in azole_bonds}
        # create smarts
        smarts = get_smarts(
            p_smiles,
            do_H_queries=False,
            do_hybridization_queries=False,
            do_heteroatom_n_queries=False,
            do_atom_n_queries=False,
            do_charge_queries=False,
            ignore_hydroxyls=False,
            ignore_nitrogens=False,
            ignore_thiols=False,
            bond_map=bond_map,
        )
        # cache
        to_add.append(
            {
                "description": "{} | {}".format(description, azole_type),
                "notation": "{} | {}".format(notation, azole_type),
                "smiles": p_smiles,
                "smarts": smarts,
                "avoid_dereplication": False,
                "enums": list(enums) + ["azole", azole_type],
                "ignore_patt_idx": ignore_patt_idx,
                "substrate_family": substrate_family,
                "substrate_id": substrate_id,
                "domain_id": domain_id,
                "domain_subclass_id": domain_subclass_id,
                "aa_id": aa_id,
            }
        )
    return to_add


#########################################################
# Main Function
#########################################################


class Azole:

    def __init__(self):
        self.library = []
        self.load_data()
        self.create_library()
        self.add_hash_ids()
        self.dereplicate_units()

    def load_data(self):
        # Load patterns
        self.patts = {}
        patts_fp = "{}/aminoacid_patterns.csv".format(aa_dir)
        df = pd.read_csv(patts_fp)
        df["smarts"] = df["smarts"].apply(lambda x: Chem.MolFromSmarts(x))
        self.patts = dict(zip(df.patt_id, df.smarts))
        # load alpha amino acids
        aa_fp = "{}/aminoacids.csv".format(aa_dir)
        aminoacids = pd.read_csv(aa_fp).to_dict("records")
        self.alpha_aminoacids = []
        self.azole_req_aminoacids = []
        for aa in aminoacids:
            aa_mol = Chem.MolFromSmiles(aa["smiles"])
            if aa_mol.HasSubstructMatch(self.patts["alpha_terminal"]):
                aa["smarts_atoms_to_ignore"] = ast.literal_eval(
                    aa["smarts_atoms_to_ignore"]
                )
                aa["enums"] = ["aa", aa["aminoacid_type"]]
                self.alpha_aminoacids.append(aa)
                if aa["notation"] in ["Ser", "Thr", "Cys"]:
                    self.azole_req_aminoacids.append(aa)
        # load ca aminoacids
        ca_fp = "{}/ca_aminoacids.csv".format(aa_dir)
        self.ca_aminoacids = pd.read_csv(ca_fp).to_dict("records")
        for aa in self.ca_aminoacids:
            aa["enums"] = ["aa", "ca"]
        # Load reactions
        reactions_fp = "{}/aminoacid_reactions.csv".format(aa_dir)
        df = pd.read_csv(reactions_fp)
        df["reaction_smarts"] = df["reaction_smarts"].apply(
            lambda x: ReactionSmarts(x)
        )
        self.reactions = dict(zip(df.reaction_id, df.reaction_smarts))
        # Load polyketide units
        self.pk_module = PolyketideModule()
        self.pk_module.load_starters()
        self.pk_module.load_substrates()
        self.pk_module.load_module_reactions()
        self.pk_module.compute_module_monomers()

    def create_library(self):
        self.add_starter_aa_azoles()
        self.add_starter_ca_azoles()
        self.add_extender_azoles()
        self.add_terminal_azoles()
        self.add_hybrid_azoles()
        self.add_pk_azoles()

    def add_starter_aa_azoles(self):
        azole_type = "azole_starter"
        # azole starter reaction
        rxn = self.reactions["aa_azole_starter"]
        # index 0, 1, 2 -> correspond to amino acid, ignore rest
        azole_patt = self.patts["aa_azole_starter"]
        keep_idx = [0, 1, 2]
        # indexes that correspond to azole bonds
        azole_bonds = [(2, 3), (3, 4), (4, 5), (5, 7), (2, 7)]
        # parse amino acids
        for aa in tqdm(
            self.alpha_aminoacids, desc="Generating alpha aa azole starters"
        ):
            self.library.extend(
                get_azole_component(
                    rxn,
                    azole_type,
                    aa["smiles"],
                    azole_patt,
                    keep_idx,
                    azole_bonds,
                    aa["description"],
                    aa["notation"],
                    aa["enums"],
                    aa["substrate_family"],
                    aa["substrate_id"],
                    aa["aa_id"],
                )
            )

    def add_starter_ca_azoles(self):
        azole_type = "azole_starter"
        rxn = self.reactions["ca_azole_starter"]
        # index 0, 1 -> correspond to amino acid, ignore rest
        azole_patt = self.patts["ca_azole_starter"]
        keep_idx = [0, 1]
        # indexes that correspond to azole bonds
        azole_bonds = [(1, 2), (2, 3), (3, 4), (4, 6), (6, 1)]
        # parse amino acids
        for aa in tqdm(
            self.ca_aminoacids, desc="Generating ca azole starters"
        ):
            self.library.extend(
                get_azole_component(
                    rxn,
                    azole_type,
                    aa["smiles"],
                    azole_patt,
                    keep_idx,
                    azole_bonds,
                    aa["description"],
                    aa["notation"],
                    aa["enums"],
                    aa["substrate_family"],
                    aa["substrate_id"],
                    aa["aa_id"],
                )
            )

    def add_extender_azoles(self):
        azole_type = "azole_extender"
        rxn = self.reactions["azole_extender"]
        # index 4, 6, 11, 10, 7 -> correspond to amino acid, ignore rest
        azole_patt = self.patts["azole_extender"]
        keep_idx = [4, 6, 11, 10, 7]
        # indexes that correspond to azole bonds
        azole_bonds = [
            (6, 11),
            (11, 10),
            (10, 8),
            (8, 7),
            (6, 7),
            (6, 4),
            (4, 3),
            (3, 2),
            (2, 1),
            (1, 5),
            (4, 5),
        ]
        # parse amino acids
        for aa in tqdm(
            self.azole_req_aminoacids, desc="Generating azole extenders"
        ):
            self.library.extend(
                get_azole_component(
                    rxn,
                    azole_type,
                    aa["smiles"],
                    azole_patt,
                    keep_idx,
                    azole_bonds,
                    aa["description"],
                    aa["notation"],
                    aa["enums"],
                    aa["substrate_family"],
                    aa["substrate_id"],
                    aa["aa_id"],
                )
            )

    def add_terminal_azoles(self):
        azole_type = "azole_terminal"
        rxn = self.reactions["azole_terminal"]
        # index 8, 1, 2, 7, 6, 3 -> correspond to amino acid
        azole_patt = self.patts["azole_terminal"]
        keep_idx = [8, 1, 2, 7, 6, 3]
        # indexes that correspond to azole bonds
        azole_bonds = [(2, 7), (7, 6), (6, 4), (4, 3), (3, 2)]
        # parse amino acids
        for aa in tqdm(
            self.azole_req_aminoacids, desc="Generating azole terminals"
        ):
            self.library.extend(
                get_azole_component(
                    rxn,
                    azole_type,
                    aa["smiles"],
                    azole_patt,
                    keep_idx,
                    azole_bonds,
                    aa["description"],
                    aa["notation"],
                    aa["enums"],
                    aa["substrate_family"],
                    aa["substrate_id"],
                    aa["aa_id"],
                )
            )

    def add_hybrid_azoles(self):
        azole_type = "azole_hybrid"
        rxn = self.reactions["azole_hybrid"]
        # index 1, 6, 5, 2 -> correspond to amino acid
        azole_patt = self.patts["azole_hybrid"]
        keep_idx = [1, 6, 5, 2]
        # indexes that correspond to azole bonds
        azole_bonds = [(1, 6), (6, 5), (5, 3), (3, 2), (1, 2)]
        # parse amino acids
        for aa in tqdm(
            self.azole_req_aminoacids, desc="Generating hybrid azoles"
        ):
            self.library.extend(
                get_azole_component(
                    rxn,
                    azole_type,
                    aa["smiles"],
                    azole_patt,
                    keep_idx,
                    azole_bonds,
                    aa["description"],
                    aa["notation"],
                    aa["enums"],
                    aa["substrate_family"],
                    aa["substrate_id"],
                    aa["aa_id"],
                )
            )

    def add_pk_azoles(self):
        azole_type = "azole_starter"
        rxn = self.reactions["pk_azole_starter"]
        # index 0 -> correspond to original pk unit
        azole_patt = self.patts["pk_azole_starter"]
        keep_idx = [0]
        # indexes that correspond to azole bonds
        azole_bonds = [(0, 1), (1, 2), (2, 3), (3, 5), (0, 5)]
        # parse pk units
        pk_units = self.pk_module.starters + self.pk_module.module_monomers
        for unit in tqdm(pk_units, desc="Generating PK azoles"):
            mol = Chem.MolFromSmiles(unit["smiles"])
            [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
            [a.SetIsotope(0) for a in mol.GetAtoms()]
            smiles = Chem.MolToSmiles(mol)
            self.library.extend(
                get_azole_component(
                    rxn,
                    azole_type,
                    smiles,
                    azole_patt,
                    keep_idx,
                    azole_bonds,
                    unit["description"],
                    unit["notation"],
                    unit["enums"],
                    unit.get("substrate_family"),
                    unit.get("substrate_id"),
                    unit.get("aa_id"),
                    domain_id=unit.get("domain_id"),
                    domain_subclass_id=unit.get("domain_subclass_id"),
                )
            )

    def add_hash_ids(self):
        for unit in self.library:
            unit["hash_id"] = get_hash_id(unit)

    def dereplicate_units(self):
        self.library = list(
            {unit["hash_id"]: unit for unit in self.library}.values()
        )
