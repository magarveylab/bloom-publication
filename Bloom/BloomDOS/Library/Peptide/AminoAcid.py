import ast
import itertools as it
import os
from typing import List

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from Bloom.BloomDOS.Chemistry.ChemUtils import (
    get_idx_with_map_num,
    get_ring_system_count,
    get_variable_atoms,
    map_smiles,
)
from Bloom.BloomDOS.Chemistry.Reaction import ReactionSmarts
from Bloom.BloomDOS.Chemistry.SmartsCompiler import get_smarts
from Bloom.BloomDOS.DataStructs import EnhancedSmartsDict
from Bloom.BloomDOS.GeneralUtils import curdir
from Bloom.BloomDOS.Library.Hasher import get_hash_id

# define dir where substrates/reactions can be found
aa_dir = os.path.join(curdir, "Library/Peptide/Tables")

#########################################################
# Helper Functions
#########################################################


def get_aa_derivatives(
    rxn: ReactionSmarts,
    rxn_name: str,
    smiles: str,
    description: str,
    notation: str,
    enums: List[str],
    substrate_family: str,
    substrate_id: str,
    aa_id: str,
) -> List[EnhancedSmartsDict]:
    derivatives = []
    products = rxn.RunReactant(smiles)
    for p_smiles in products:
        atoms_to_ignore = get_idx_with_map_num(p_smiles, 3)
        # Add carbons in benzene rings to atoms_to_ignore (ex. ring in Tyrosine)
        benzene_hits = Chem.MolFromSmiles(p_smiles).GetSubstructMatches(
            Chem.MolFromSmarts("[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1")
        )
        benzene_hits = sorted(list(set([x for y in benzene_hits for x in y])))
        atoms_to_ignore = atoms_to_ignore + benzene_hits
        variable_atoms = get_variable_atoms(p_smiles)
        # create smarts
        smarts = get_smarts(
            p_smiles,
            do_H_queries=True,
            do_hybridization_queries=True,
            do_heteroatom_n_queries=True,
            do_atom_n_queries=True,
            do_charge_queries=False,
            ignore_hydroxyls=True,
            ignore_nitrogens=True,
            ignore_thiols=True,
            atoms_to_ignore=atoms_to_ignore,
        )
        # cache
        derivatives.append(
            {
                "description": "{} | {}".format(description, rxn_name),
                "notation": "{} | {}".format(notation, rxn_name),
                "smiles": p_smiles,
                "smarts": smarts,
                "avoid_dereplication": False,
                "enums": enums + [rxn_name],
                "ignore_patt_idx": variable_atoms,
                "substrate_family": substrate_family,
                "substrate_id": substrate_id,
                "aa_id": aa_id,
            }
        )
    return derivatives


#########################################################
# Main Function
#########################################################


class AminoAcid:

    def __init__(self):
        # dereplicate
        self.library = []
        self.load_data()
        self.create_library()
        self.add_misc_units()
        self.add_hash_ids()
        self.dereplicate_units()

    def load_data(self):
        # load tailoring enums
        aa_tailoring_fp = "{}/aminoacid_tailoring.csv".format(aa_dir)
        tailoring_enums = pd.read_csv(aa_tailoring_fp).to_dict("records")
        for t in tailoring_enums:
            t["smarts"] = Chem.MolFromSmarts(t["smarts"])
        # load amino acids
        aa_fp = "{}/aminoacids.csv".format(aa_dir)
        self.aminoacids = pd.read_csv(aa_fp).to_dict("records")
        for aa in self.aminoacids:
            aa["smarts_atoms_to_ignore"] = ast.literal_eval(
                aa["smarts_atoms_to_ignore"]
            )
            aa["ignore_patt_idx"] = ast.literal_eval(aa["ignore_patt_idx"])
            aa["enums"] = ["aa", aa["aminoacid_type"]]
            mol = Chem.MolFromSmiles(aa["smiles"])
            if get_ring_system_count(mol) > 0:
                aa["enums"].append("ring_aa")
            # add tailoring enums
            for t in tailoring_enums:
                if mol.HasSubstructMatch(t["smarts"]):
                    aa["enums"].append(t["enum"])
        # load ca amino acids
        ca_fp = "{}/ca_aminoacids.csv".format(aa_dir)
        self.ca_aminoacids = pd.read_csv(ca_fp).to_dict("records")
        for ca in self.ca_aminoacids:
            ca["enums"] = ["aa", "ca"]
            # add tailoring enums
            for t in tailoring_enums:
                if mol.HasSubstructMatch(t["smarts"]):
                    aa["enums"].append(t["enum"])
        # Load reactions
        reaction_fp = "{}/aminoacid_reactions.csv".format(aa_dir)
        df = pd.read_csv(reaction_fp)
        df["reaction_smarts"] = df["reaction_smarts"].apply(
            lambda x: ReactionSmarts(x)
        )
        self.reactions = dict(zip(df.reaction_id, df.reaction_smarts))
        # Load patterns
        self.patts = {}
        patts_fp = "{}/aminoacid_patterns.csv".format(aa_dir)
        df = pd.read_csv(patts_fp)
        df["smarts"] = df["smarts"].apply(lambda x: Chem.MolFromSmarts(x))
        self.patts = dict(zip(df.patt_id, df.smarts))

    def create_library(self):
        self.add_base_smiles()
        self.add_incorporated_aminoacids()
        self.add_amino_aldehydes()
        self.add_alkanolamines()
        self.add_hydroxy_acids()
        self.add_cyclo_aminoacids()
        self.add_dehydro_aminoacids()
        self.add_ketoacids()
        self.add_decarboxylated_terminals()
        self.add_hybrid_starters()
        self.add_hpg_variations()
        self.add_imine_aminoacids()
        self.add_ca_aminoacids()
        self.add_condensed_cys_ser_thr()
        self.add_avoid_dereplication()
        self.add_can_dereplicate()

    def add_misc_units(self):
        # load variable map
        var_fp = "{}/aminoacid_misc_variable_map.csv".format(aa_dir)
        lookup = {}
        for row in pd.read_csv(var_fp).to_dict("records"):
            notation = row["notation"]
            if notation not in lookup:
                lookup[notation] = {}
            atom_list = ast.literal_eval(row["atom"])
            for atom in atom_list:
                if atom not in lookup[notation]:
                    lookup[notation][atom] = []
                lookup[notation][atom].extend(ast.literal_eval(row["enums"]))
        # load misc units
        misc_fp = "{}/aminoacid_misc.csv".format(aa_dir)
        misc_units = pd.read_csv(misc_fp)
        misc_units["enums"] = misc_units["enums"].apply(
            lambda x: ast.literal_eval(x)
        )
        misc_units["ignore_patt_idx"] = misc_units["ignore_patt_idx"].apply(
            lambda x: ast.literal_eval(x)
        )
        misc_units["variable_map"] = misc_units["notation"].apply(
            lambda x: lookup.get(x, {})
        )
        misc_units = misc_units.to_dict("records")
        self.library.extend(misc_units)

    def add_base_smiles(self):
        for aa in tqdm(self.aminoacids, desc="Generating initial amino acids"):
            # flag atoms to ignore queries - the flag is used to keep track of modifications
            flag_atoms = {
                atom_idx: 3 for atom_idx in aa["smarts_atoms_to_ignore"]
            }
            smiles = map_smiles(aa["smiles"], flag_atoms)
            variable_atoms = get_variable_atoms(smiles)
            atoms_to_ignore = (
                get_idx_with_map_num(smiles, 3) + aa["ignore_patt_idx"]
            )
            # Add carbons in benzene rings to atoms_to_ignore (ex. ring in Tyrosine)
            benzene_hits = Chem.MolFromSmiles(smiles).GetSubstructMatches(
                self.patts["benzene"]
            )
            benzene_hits = sorted(
                list(set([x for y in benzene_hits for x in y]))
            )
            atoms_to_ignore = atoms_to_ignore + benzene_hits
            # create smarts
            smarts = get_smarts(
                smiles,
                do_H_queries=True,
                do_hybridization_queries=True,
                do_heteroatom_n_queries=True,
                do_atom_n_queries=True,
                do_charge_queries=False,
                ignore_hydroxyls=True,
                ignore_nitrogens=True,
                ignore_thiols=True,
                atoms_to_ignore=atoms_to_ignore,
            )
            ignore_patt_idx = list(set(variable_atoms + aa["ignore_patt_idx"]))
            enums = list(aa["enums"] + ["terminal"])
            # cache
            self.library.append(
                {
                    "description": aa["description"],
                    "notation": aa["notation"],
                    "smiles": smiles,
                    "smarts": smarts,
                    "avoid_dereplication": True,
                    "enums": aa["enums"] + ["terminal"],
                    "ignore_patt_idx": ignore_patt_idx,
                    "substrate_id": aa["substrate_id"],
                    "substrate_family": aa["substrate_family"],
                    "aa_id": aa["aa_id"],
                }
            )

    def add_incorporated_aminoacids(self):
        to_add = []
        for aa in tqdm(
            self.library, desc="Generating incorporated amino acids"
        ):
            # get incorporated forms
            rxn = self.reactions["incorporated"]
            incorporated_smiles = rxn.RecursiveRunReactant(aa["smiles"])
            for smiles in incorporated_smiles:
                # get atoms to ignore
                atoms_to_ignore = (
                    get_idx_with_map_num(smiles, 3) + aa["ignore_patt_idx"]
                )
                # Add carbons in benzene rings to atoms_to_ignore (ex. ring in Tyrosine)
                benzene_hits = Chem.MolFromSmiles(smiles).GetSubstructMatches(
                    self.patts["benzene"]
                )
                benzene_hits = sorted(
                    list(set([x for y in benzene_hits for x in y]))
                )
                atoms_to_ignore = atoms_to_ignore + benzene_hits
                # create smarts
                smarts = get_smarts(
                    smiles,
                    do_H_queries=True,
                    do_hybridization_queries=True,
                    do_heteroatom_n_queries=True,
                    do_atom_n_queries=True,
                    do_charge_queries=False,
                    ignore_hydroxyls=True,
                    ignore_nitrogens=True,
                    ignore_thiols=True,
                    atoms_to_ignore=atoms_to_ignore,
                )
                variable_atoms = get_variable_atoms(smiles)
                ignore_patt_idx = list(
                    set(variable_atoms + aa["ignore_patt_idx"])
                )
                enums = aa["enums"] + ["incorporated"]
                enums.remove("terminal")
                # cache
                to_add.append(
                    {
                        "description": aa["description"],
                        "notation": aa["notation"],
                        "smiles": smiles,
                        "smarts": smarts,
                        "avoid_dereplication": True,
                        "enums": enums,
                        "ignore_patt_idx": ignore_patt_idx,
                        "substrate_id": aa["substrate_id"],
                        "substrate_family": aa["substrate_family"],
                        "aa_id": aa["aa_id"],
                    }
                )
        self.library.extend(to_add)

    def add_amino_aldehydes(self):
        to_add = []
        for aa in tqdm(self.library, desc="Generating amino aldehydes"):
            mol = Chem.MolFromSmiles(aa["smiles"])
            if mol.HasSubstructMatch(self.patts["alpha_terminal"]):
                derivatives = get_aa_derivatives(
                    self.reactions["amino_aldehyde"],
                    "AmAl",
                    aa["smiles"],
                    aa["description"],
                    aa["notation"],
                    aa["enums"],
                    aa["substrate_family"],
                    aa["substrate_id"],
                    aa["aa_id"],
                )
                to_add.extend(derivatives)
        self.library.extend(to_add)

    def add_alkanolamines(self):
        to_add = []
        for aa in tqdm(self.library, desc="Generating alkanolamines"):
            if aa["notation"] != "Mal-NH3":
                derivatives = get_aa_derivatives(
                    self.reactions["alkanolamine"],
                    "Alk",
                    aa["smiles"],
                    aa["description"],
                    aa["notation"],
                    aa["enums"],
                    aa["substrate_family"],
                    aa["substrate_id"],
                    aa["aa_id"],
                )
                to_add.extend(derivatives)
        self.library.extend(to_add)

    def add_hydroxy_acids(self):
        to_add = []
        for aa in tqdm(self.library, desc="Generating hydroxy acids"):
            derivatives = get_aa_derivatives(
                self.reactions["hydroxy_acid"],
                "HA",
                aa["smiles"],
                aa["description"],
                aa["notation"],
                aa["enums"],
                aa["substrate_family"],
                aa["substrate_id"],
                aa["aa_id"],
            )
            to_add.extend(derivatives)
        self.library.extend(to_add)

    def add_cyclo_aminoacids(self):
        to_add = []
        for aa in tqdm(self.library, desc="Generating cyclo amino acids"):
            mol = Chem.MolFromSmiles(aa["smiles"])
            if mol.HasSubstructMatch(self.patts["alpha"]):
                derivatives = get_aa_derivatives(
                    self.reactions["alpha_cyclo"],
                    "cyclo",
                    aa["smiles"],
                    aa["description"],
                    aa["notation"],
                    aa["enums"],
                    aa["substrate_family"],
                    aa["substrate_id"],
                    aa["aa_id"],
                )
            else:
                derivatives = get_aa_derivatives(
                    self.reactions["beta_cyclo"],
                    "cyclo",
                    aa["smiles"],
                    aa["description"],
                    aa["notation"],
                    aa["enums"],
                    aa["substrate_family"],
                    aa["substrate_id"],
                    aa["aa_id"],
                )
            to_add.extend(derivatives)
        self.library.extend(to_add)

    def add_dehydro_aminoacids(self):
        to_add = []
        for aa in tqdm(self.library, desc="Generating dehydro amino acids"):
            derivatives = get_aa_derivatives(
                self.reactions["dehydro"],
                "dH",
                aa["smiles"],
                aa["description"],
                aa["notation"],
                aa["enums"],
                aa["substrate_family"],
                aa["substrate_id"],
                aa["aa_id"],
            )
            to_add.extend(derivatives)
        self.library.extend(to_add)

    def add_ketoacids(self):
        to_add = []
        for aa in tqdm(self.library, desc="Generating alpha ketoacids"):
            derivatives = get_aa_derivatives(
                self.reactions["ketoacid"],
                "aKT",
                aa["smiles"],
                aa["description"],
                aa["notation"],
                aa["enums"],
                aa["substrate_family"],
                aa["substrate_id"],
                aa["aa_id"],
            )
            to_add.extend(derivatives)
        self.library.extend(to_add)

    def add_decarboxylated_terminals(self):
        to_add = []
        decarb_patt = self.patts["decarboxylation"]
        for aa in tqdm(
            self.library, desc="Generating decarboxylated terminals"
        ):
            if set(aa["enums"]) == {"aa", "alpha", "incorporated"}:
                derivatives = get_aa_derivatives(
                    self.reactions["decarboxylation"],
                    "decarb",
                    aa["smiles"],
                    aa["description"],
                    aa["notation"],
                    aa["enums"],
                    aa["substrate_family"],
                    aa["substrate_id"],
                    aa["aa_id"],
                )
                # update index to ignore with imine atom which is at index 3
                for y in derivatives:
                    mol = Chem.MolFromSmiles(y["smiles"])
                    hit = mol.GetSubstructMatch(decarb_patt)
                    y["ignore_patt_idx"] = y["ignore_patt_idx"] + [hit[3]]
                to_add.extend(derivatives)
        self.library.extend(to_add)

    def add_hybrid_starters(self):
        hybrid_rxns = [
            "hybrid_KS",
            "hybrid_KS_beta",
            "hybrid_KS_dehydro",
            "hybrid_KR",
            "hybrid_KR_beta",
            "hybrid_KR_dehydro",
            "hybrid_DH",
            "hybrid_DH_beta",
            "hybrid_DH_dehydro",
            "hybrid_ER",
            "hybrid_ER_beta",
            "hybrid_ER_dehydro",
            "hybrid_B1",
            "hybrid_B1_beta",
            "hybrid_B1_dehydro",
            "hybrid_B2",
            "hybrid_B2_beta",
            "hybrid_B2_dehydro",
            "hybrid_BER",
            "hybrid_BER_beta",
            "hybrid_BER_dehydro",
        ]
        to_add = []
        for rxn_id in hybrid_rxns:
            rxn_patt = self.patts[rxn_id]
            for aa in tqdm(
                self.library, desc="Generating {} starters".format(rxn_id)
            ):
                # restriction
                if "Mal-NH3" not in aa["notation"]:
                    rxn = self.reactions[rxn_id]
                    products = rxn.RunReactant(aa["smiles"])
                    for p_smiles in products:
                        # get atoms to ignore
                        atoms_to_ignore = get_idx_with_map_num(p_smiles, 3)
                        variable_atoms = get_variable_atoms(p_smiles)
                        mol = Chem.MolFromSmiles(p_smiles)
                        # Last index of SMARTS pattern in B1 reactions corresponds to flexible PK atom
                        if "B1" in rxn_id:
                            flexible_pk_index = -1
                        # Second last index of SMARTS pattern in B1 reactions corresponds to flexible PK atom
                        elif "B2" in rxn_id:
                            flexible_pk_index = -2
                        # First index of SMARTS pattern in DH_dehydro reactions corresponds to flexible PK atom
                        elif rxn_id == "hybrid_DH_dehydro":
                            flexible_pk_index = 0
                        # Last index of SMARTS pattern in alpha reactions corresponds to flexible PK atom
                        elif "beta" not in rxn_id:
                            flexible_pk_index = -1
                        # Second last index of SMARTS pattern in hybrid_B1_beta/hybrid_BER_beta reactions corresponds
                        # to flexible PK atom
                        elif rxn_id in ["hybrid_B1_beta", "hybrid_BER_beta"]:
                            flexible_pk_index = -2
                        # First index of SMARTS pattern in all other beta reactions corresponds to flexible PK atom
                        else:
                            flexible_pk_index = 0
                        hit = mol.GetSubstructMatch(rxn_patt)
                        atoms_to_ignore = (
                            [hit[flexible_pk_index]]
                            + list(variable_atoms)
                            + atoms_to_ignore
                        )
                        # Add carbons in benzene rings to atoms_to_ignore (ex. ring in Tyrosine)
                        benzene_hits = Chem.MolFromSmiles(
                            p_smiles
                        ).GetSubstructMatches(self.patts["benzene"])
                        benzene_hits = sorted(
                            list(set([x for y in benzene_hits for x in y]))
                        )
                        atoms_to_ignore = atoms_to_ignore + benzene_hits
                        # Atoms IDs 0 and 1 always correspond to PK unit in ER/DH reactions
                        # For all other reactions, Atom IDs 0, 1 and 2 correspond to PK unit
                        if rxn_id in [
                            "hybrid_ER",
                            "hybrid_ER_beta",
                            "hybrid_ER_dehydro",
                            "hybrid_DH",
                            "hybrid_DH_beta",
                            "hybrid_DH_dehydro",
                        ]:
                            pk_atoms = [0, 1]
                        else:
                            pk_atoms = [0, 1, 2]
                        # Ignore PK atoms and variable atoms
                        ignore_patt_idx = pk_atoms + variable_atoms
                        # create smarts
                        smarts = get_smarts(
                            p_smiles,
                            do_H_queries=True,
                            do_hybridization_queries=True,
                            do_heteroatom_n_queries=True,
                            do_atom_n_queries=True,
                            do_charge_queries=False,
                            ignore_hydroxyls=True,
                            ignore_nitrogens=True,
                            ignore_thiols=True,
                            atoms_to_ignore=atoms_to_ignore,
                        )
                        # assign an enum to all the pk atoms
                        variable_map = {}
                        for atom_idx in ignore_patt_idx:
                            variable_map[atom_idx] = ["pk"]
                        # cache
                        desc = "{} | {}".format(aa["description"], rxn_id)
                        notation = "{} | {}".format(aa["notation"], rxn_id)
                        to_add.append(
                            {
                                "description": desc,
                                "notation": notation,
                                "smiles": p_smiles,
                                "smarts": smarts,
                                "avoid_dereplication": False,
                                "enums": aa["enums"] + ["hybrid", rxn_id],
                                "ignore_patt_idx": ignore_patt_idx,
                                "variable_map": variable_map,
                                "substrate_family": aa["substrate_family"],
                                "substrate_id": aa["substrate_id"],
                                "aa_id": aa["aa_id"],
                            }
                        )
        self.library.extend(to_add)

    def add_hpg_variations(self):
        rxn_name = "hpg"
        hpg_patt = self.patts["hpg"]
        to_add = []
        for aa in tqdm(self.library, desc="Generating Hpg variations"):
            if "hpg" not in aa["notation"].lower():
                continue
            if "hybrid" in aa["enums"]:
                continue
            mol = Chem.MolFromSmiles(aa["smiles"])
            hits = mol.GetSubstructMatches(hpg_patt)
            if len(hits) == 0:
                continue
            # get all hydrogen sites from the patts
            # OH corresponds to index 0
            hydroxyl_hits = set(h[0] for h in hits)
            # carbon sites correspond to the rest
            carbon_hits = set()
            for h in hits:
                carbon_hits.update(h[1:])
            carbon_hits = list(carbon_hits)
            # find variable atoms
            variable_hits = get_variable_atoms(smiles=aa["smiles"])
            # find all hydroxyl combinations
            hydroxyl_combs = []
            for x in range(len(hydroxyl_hits) + 1):
                hydroxyl_combs.extend(it.combinations(hydroxyl_hits, x))
            # create entry for each hydroxyl comb
            for comb in hydroxyl_combs:
                ignore_patt_idx = variable_hits + list(comb)
                atoms_to_ignore = list(comb) + carbon_hits + variable_hits
                # Add carbons in benzene rings to atoms_to_ignore (ex. ring in Tyrosine)
                benzene_hits = Chem.MolFromSmiles(
                    aa["smiles"]
                ).GetSubstructMatches(self.patts["benzene"])
                benzene_hits = sorted(
                    list(set([x for y in benzene_hits for x in y]))
                )
                atoms_to_ignore = atoms_to_ignore + benzene_hits
                smarts = get_smarts(
                    aa["smiles"],
                    do_H_queries=True,
                    do_hybridization_queries=True,
                    do_heteroatom_n_queries=True,
                    do_atom_n_queries=True,
                    do_charge_queries=False,
                    ignore_hydroxyls=True,
                    ignore_nitrogens=True,
                    ignore_thiols=True,
                    atoms_to_ignore=atoms_to_ignore,
                )
                to_add.append(
                    {
                        "description": aa["description"],
                        "notation": aa["notation"],
                        "smiles": aa["smiles"],
                        "smarts": smarts,
                        "curated": False,
                        "enums": aa["enums"] + [rxn_name],
                        "ignore_patt_idx": ignore_patt_idx,
                        "substrate_family": aa["substrate_family"],
                        "substrate_id": aa["substrate_id"],
                        "aa_id": aa["aa_id"],
                    }
                )
        self.library.extend(to_add)

    def add_imine_aminoacids(self):
        to_add = []
        for aa in tqdm(self.library, desc="Generating imine amino acids"):
            derivatives = get_aa_derivatives(
                self.reactions["imine"],
                "imine",
                aa["smiles"],
                aa["description"],
                aa["notation"],
                aa["enums"],
                aa["substrate_family"],
                aa["substrate_id"],
                aa["aa_id"],
            )
            to_add.extend(derivatives)
        self.library.extend(to_add)

    def add_ca_aminoacids(self):
        to_add = []
        for aa in tqdm(
            self.ca_aminoacids, desc="Generating incorporated carboxylic acids"
        ):
            # get incorporated forms
            rxn = self.reactions["incorporated"]
            incorporated_smiles = rxn.RecursiveRunReactant(aa["smiles"])
            for smiles in incorporated_smiles:
                # create smarts
                smarts = get_smarts(
                    smiles,
                    do_H_queries=True,
                    do_hybridization_queries=True,
                    do_heteroatom_n_queries=True,
                    do_atom_n_queries=True,
                    do_charge_queries=False,
                    ignore_hydroxyls=True,
                    ignore_nitrogens=True,
                    ignore_thiols=True,
                )
                # cache
                enums = aa["enums"] + ["incorporated"]
                variable_atoms = get_variable_atoms(smiles)
                to_add.append(
                    {
                        "description": aa["description"] + " | CA",
                        "notation": aa["notation"] + " | CA",
                        "smiles": aa["smiles"],
                        "smarts": smarts,
                        "avoid_dereplication": True,
                        "enums": enums,
                        "ignore_patt_idx": variable_atoms,
                        "substrate_family": aa["substrate_family"],
                        "substrate_id": aa["substrate_id"],
                        "aa_id": aa["aa_id"],
                    }
                )
        self.library.extend(to_add)

    def add_condensed_cys_ser_thr(self):
        to_add = []
        combs = [("Cys", 16), ("Ser", 8), ("Thr", 8)]
        for comb in tqdm(combs, desc="Generating condensed Cys/Ser/Thr"):
            for aa in self.library:
                if comb[0] in aa["notation"] and len(aa["enums"]) == 3:
                    mol = Chem.MolFromSmiles(aa["smiles"])
                    sulf_index = [
                        x.GetIdx()
                        for x in mol.GetAtoms()
                        if x.GetAtomicNum() == comb[1]
                    ]
                    if len(sulf_index) == 0:
                        break
                    else:
                        sulf_index = sulf_index[0]
                        ignore_patt_idx = tuple(
                            sorted(
                                list(
                                    set(
                                        list(aa["ignore_patt_idx"])
                                        + [sulf_index]
                                    )
                                )
                            )
                        )
                        to_add.append(
                            {
                                "description": aa["description"],
                                "notation": aa["notation"],
                                "smiles": aa["smiles"],
                                "smarts": aa["smarts"],
                                "avoid_dereplication": True,
                                "enums": aa["enums"],
                                "ignore_patt_idx": ignore_patt_idx,
                                "substrate_family": aa["substrate_family"],
                                "substrate_id": aa["substrate_id"],
                                "aa_id": aa["aa_id"],
                            }
                        )
        self.library.extend(to_add)

    def add_avoid_dereplication(self):
        aa_notation_queries = ["Ahp", "Asn", "Gln"]
        for k in self.library:
            notation = k["notation"]
            for query in aa_notation_queries:
                if query in notation:
                    k["avoid_dereplication"] = True

    def add_can_dereplicate(self):
        for k in self.library:
            if len({"aKT", "dH_2"}.intersection(set(k["enums"]))) > 0:
                k["can_dereplicate"] = False
            elif "gly" in k["notation"].lower():
                k["can_dereplicate"] = False
            else:
                k["can_dereplicate"] = True

    def add_hash_ids(self):
        for unit in self.library:
            unit["hash_id"] = get_hash_id(unit)

    def dereplicate_units(self):
        cache = {}
        for unit in self.library:
            hash_id = unit["hash_id"]
            if hash_id not in cache:
                cache[hash_id] = []
            cache[hash_id].append(unit["notation"])
        for x, y in cache.items():
            if "LA" in y:
                print(y)
            if "Ph-Ser" in y:
                print(y)
            if "bMe-Phe" in y:
                print(y)
            if "4PePro" in y:
                print(y)
        self.library = list(
            {unit["hash_id"]: unit for unit in self.library}.values()
        )
