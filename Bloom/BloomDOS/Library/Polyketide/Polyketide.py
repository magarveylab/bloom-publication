from typing import Dict, List, Set

from rdkit import Chem
from tqdm import tqdm

from Bloom.BloomDOS.Chemistry.ChemUtils import (
    get_idx_with_map_num,
    get_natural_smiles,
    get_variable_atoms,
    update_unmapped_atoms,
)
from Bloom.BloomDOS.Chemistry.Reaction import dethiolation
from Bloom.BloomDOS.Chemistry.SmartsCompiler import get_smarts
from Bloom.BloomDOS.Library.Hasher import get_hash_id
from Bloom.BloomDOS.Library.Polyketide.PolyketideModule import PolyketideModule

#########################################################
# Helper Functions
#########################################################


def get_variable_map(
    smiles: str, linkers: List[dict] = []
) -> Dict[int, Set[str]]:
    variable_map = {}
    mol = Chem.MolFromSmiles(smiles)
    # add enums for linkers in specialized motifs
    if len(linkers) > 0:
        for l in linkers:
            # these smarts identify specific atoms in molecule
            # that need to be associated to defined enums
            hits = mol.GetSubstructMatches(l["linker_smarts"])
            if len(hits) > 0:
                patt = hits[0]
                for patt_idx in l["linker_atoms"]:
                    atom_idx = patt[patt_idx]
                    if atom_idx not in variable_map:
                        variable_map[atom_idx] = set(l["linker_enums"])
                    else:
                        variable_map[atom_idx].update(l["linker_enums"])
    # add normal pk linkers
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_num = atom.GetAtomicNum()
        # only add entry if linker enum is not present
        if atom_idx not in variable_map:
            if atom_num == 0:
                variable_map[atom_idx] = {"pk", "aa", "starter"}
            elif atom_num == 85:
                variable_map[atom_idx] = {"tailoring"}
    # format map
    variable_map = {x: list(y) for x, y in variable_map.items()}
    return variable_map


def get_alpha_restriction_map(smiles: str) -> List[int]:
    # alpha carbons are indicated with C13 isotope
    mol = Chem.MolFromSmiles(smiles)
    return [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsotope() == 13
    ]


def get_TE_atom(smiles: str) -> int:
    # terminal variable atom in cyclic system should not be carbon
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetSubstructMatch(Chem.MolFromSmarts("[#8]=[#6]-[#0]"))[2]


def get_bond_map(smiles: str, bonds: List[dict]) -> Dict[int, str]:
    # for specialized motifs, indicacte which bonds are aromatic for smarts creation
    bond_map = {}
    mol = Chem.MolFromSmiles(smiles)
    for b in bonds:
        hits = mol.GetSubstructMatches(b["smarts"])
        for patt in hits:
            for a1, a2 in b["bonds"]:
                bid = tuple(sorted([patt[a1], patt[a2]]))
                bond_map[bid] = b["bond_descriptor"]
    return bond_map


#########################################################
# Main Function
#########################################################


class Polyketide:

    def __init__(self):
        self.library = []
        self.load_data()
        self.create_library()
        self.add_hash_ids()

    def load_data(self):
        self.pk_module = PolyketideModule()
        self.pk_module.load_starters()
        self.pk_module.load_substrates()
        self.pk_module.load_module_reactions()
        self.pk_module.compute_module_monomers()
        self.pk_module.load_terminal_reactions()
        self.pk_module.load_specialized_motifs()
        self.pk_module.load_standalone_motifs()

    def create_library(self):
        self.add_starter()
        self.add_extender()
        self.add_terminal()
        self.add_specialized()
        self.add_specialized_terminal()
        self.add_standalone_motifs()

    def add_starter(self):
        for s in tqdm(
            self.pk_module.starters, desc="Generating polyketide starters"
        ):
            # dethiolate starter units
            product = dethiolation.RunReactant(s["smiles"])[0]
            # any atoms mapped with 1 (no queries - flexible for modifications)
            mol = Chem.MolFromSmiles(product)
            atoms_to_ignore = [
                a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 1
            ]
            [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
            product = Chem.MolToSmiles(mol, canonical=False)
            # prepare smarts
            smarts = get_smarts(
                product,
                do_H_queries=True,
                do_hybridization_queries=True,
                do_heteroatom_n_queries=True,
                do_atom_n_queries=True,
                ignore_hydroxyls=True,
                consider_carboxylic=True,
                ignore_nitrogens=True,
                consider_amide=True,
                ignore_variable_groups=True,
                atoms_to_ignore=atoms_to_ignore,
            )
            variable_map = get_variable_map(product)
            # ensure that variable atom maps to carbon
            # this is to distinguish methyl starter unit for methyl tailoring
            variable_to_atomic_num_map = {a: [6] for a in variable_map}
            # get unknown restrictions
            # variable atoms correspond to atoms in adjacent units
            # these units cannot be unknown
            alpha_restriction_map = get_alpha_restriction_map(product)
            # cache
            self.library.append(
                {
                    "notation": s["notation"],
                    "description": s["description"] + " | starter",
                    "smarts": smarts,
                    "smiles": product,
                    "ignore_patt_idx": list(variable_map.keys()),
                    "enums": list(s["enums"]),
                    "variable_map": variable_map,
                    "variable_to_atomic_num_map": variable_to_atomic_num_map,
                    "alpha_restriction_map": alpha_restriction_map,
                    "substrate_id": s["substrate_id"],
                    "substrate_family": s["substrate_family"],
                }
            )

    def add_extender(self):
        # apply reaction to monomer units
        for m in tqdm(
            self.pk_module.module_monomers,
            desc="Generating polyketide extenders",
        ):
            description = m["description"] + " | extender"
            product = dethiolation.RunReactant(m["smiles"])[0]
            # prepare smarts
            smarts = get_smarts(
                product,
                do_H_queries=True,
                do_hybridization_queries=True,
                do_heteroatom_n_queries=True,
                do_atom_n_queries=True,
                ignore_hydroxyls=True,
                consider_carboxylic=True,
                ignore_nitrogens=True,
                consider_amide=True,
                ignore_variable_groups=True,
            )
            variable_map = get_variable_map(product)
            alpha_restriction_map = get_alpha_restriction_map(product)
            # cache
            self.library.append(
                {
                    "notation": m["notation"],
                    "description": description,
                    "smarts": smarts,
                    "smiles": product,
                    "ignore_patt_idx": list(variable_map.keys()),
                    "enums": list(m["enums"]) + ["extender"],
                    "variable_map": variable_map,
                    "alpha_restriction_map": alpha_restriction_map,
                    "priortization": m["priortization"],
                    "uniquify": m["uniquify"],
                    "domain_id": m["domain_id"],
                    "domain_subclass_id": m["domain_subclass_id"],
                    "substrate_id": m["substrate_id"],
                    "substrate_family": m["substrate_family"],
                }
            )

    def add_terminal(self):
        for m in tqdm(
            self.pk_module.module_monomers,
            desc="Generating polyketide terminal",
        ):
            for t in self.pk_module.terminal_rxns:
                product = t["reaction_smarts"].RunReactant(m["smiles"])[0]
                notation = m["notation"] + " | {}".format(t["reaction_id"])
                description = m["description"] + " | {}".format(
                    t["reaction_id"]
                )
                enums = m["enums"] | {"TE"}
                if t["reaction_id"] == "TE":
                    # terminal variable atom should not be carbon
                    # it should be oxygen or nitrogen for cyclic
                    terminal_var_atom = get_TE_atom(product)
                    atom_restriction_map = {terminal_var_atom: ["C"]}
                else:
                    atom_restriction_map = {}
                # prepare smarts
                smarts = get_smarts(
                    product,
                    do_H_queries=True,
                    do_hybridization_queries=True,
                    do_heteroatom_n_queries=True,
                    do_atom_n_queries=True,
                    ignore_hydroxyls=True,
                    consider_carboxylic=True,
                    ignore_nitrogens=True,
                    consider_amide=True,
                    ignore_variable_groups=True,
                    atom_restriction_map=atom_restriction_map,
                )
                variable_map = get_variable_map(product)
                if t["reaction_id"] == "TE":
                    variable_map[terminal_var_atom].extend(
                        ["miscellaneous", "nonmodular"]
                    )
                alpha_restriction_map = get_alpha_restriction_map(product)
                # cache
                self.library.append(
                    {
                        "notation": notation,
                        "description": description,
                        "smarts": smarts,
                        "smiles": product,
                        "ignore_patt_idx": list(variable_map.keys()),
                        "enums": list(enums) + ["terminal"],
                        "variable_map": variable_map,
                        "alpha_restriction_map": alpha_restriction_map,
                        "priortization": m["priortization"],
                        "uniquify": m["uniquify"],
                        "domain_id": m["domain_id"],
                        "domain_subclass_id": m["domain_subclass_id"],
                        "substrate_id": m["substrate_id"],
                        "substrate_family": m["substrate_family"],
                    }
                )

    def add_specialized(self):
        for s in tqdm(
            self.pk_module.specialized_motifs,
            desc="Generating polyketide specialized units",
        ):
            for m in self.pk_module.module_monomers:
                if (
                    len(s["ignore_substrate_ids"]) > 0
                    and m["substrate"] in s["ignore_substrate_ids"]
                ):
                    continue
                if (
                    len(s["predecessor_reaction_ids"]) > 0
                    and m["reaction_id"] not in s["predecessor_reaction_ids"]
                ):
                    continue
                if (
                    len(s["predecessor_substrate_ids"]) > 0
                    and m["substrate"] not in s["predecessor_substrate_ids"]
                ):
                    continue
                notation = m["notation"] + " | {}".format(s["reaction_name"])
                description = m["description"] + " | {}".format(
                    s["reaction_name"]
                )
                enums = m["enums"] | s["enums"]
                product = s["reaction_smarts"].RunReactant(m["smiles"])[0]
                # annotate unmapped atoms with index 2 (to identify linker atoms)
                product = update_unmapped_atoms(product, 2)
                # some reactions require thiol removal to generate motif
                if s["consider_thiol_removal"] == True:
                    product = dethiolation.RunReactant(product)[0]
                # determine atoms_to_ignore
                atoms_to_ignore = get_idx_with_map_num(product, 2)
                variable_atoms = get_variable_atoms(product)
                atoms_to_ignore = list(set(atoms_to_ignore + variable_atoms))
                # determine bond descriptors
                bond_map = get_bond_map(product, s["bonds"])
                # determine variable map
                variable_map = get_variable_map(product, linkers=s["linkers"])
                # modify variable map
                if s["consider_module_neighbors"] == False:
                    to_remove = set()
                    for x in variable_map:
                        # linker to next module is defined by {'pk', 'starter', 'aa'}
                        next_module_enums = {"pk", "starter", "aa"}
                        if next_module_enums.issubset(set(variable_map[x])):
                            to_remove.add(x)
                    for n in to_remove:
                        del variable_map[n]
                # prepare smarts
                smarts = get_smarts(
                    product,
                    do_H_queries=True,
                    do_hybridization_queries=True,
                    do_heteroatom_n_queries=True,
                    do_atom_n_queries=True,
                    ignore_hydroxyls=True,
                    consider_carboxylic=True,
                    ignore_nitrogens=True,
                    consider_amide=True,
                    ignore_variable_groups=True,
                    atoms_to_ignore=atoms_to_ignore,
                    bond_map=bond_map,
                )
                alpha_restriction_map = get_alpha_restriction_map(product)
                self.library.append(
                    {
                        "notation": notation,
                        "description": description,
                        "smarts": smarts,
                        "smiles": product,
                        "ignore_patt_idx": list(atoms_to_ignore),
                        "enums": list(enums),
                        "variable_map": variable_map,
                        "alpha_restriction_map": alpha_restriction_map,
                        "priortization": m["priortization"],
                        "avoid_dereplication": s["avoid_dereplication"],
                        "uniquify": m["uniquify"],
                        "domain_id": m["domain_id"],
                        "domain_subclass_id": m["domain_subclass_id"],
                        "substrate_id": m["substrate_id"],
                        "substrate_family": m["substrate_family"],
                    }
                )

    def add_specialized_terminal(self):
        for s in tqdm(
            self.pk_module.specialized_motifs,
            desc="Generating polyketide specialized terminal units",
        ):
            if s["consider_terminal"] == False:
                continue
            for m in self.pk_module.module_monomers:
                if (
                    len(s["ignore_substrate_ids"]) > 0
                    and m["substrate"] in s["ignore_substrate_ids"]
                ):
                    continue
                if (
                    len(s["predecessor_reaction_ids"]) > 0
                    and m["reaction_id"] not in s["predecessor_reaction_ids"]
                ):
                    continue
                if (
                    len(s["predecessor_substrate_ids"]) > 0
                    and m["substrate"] not in s["predecessor_substrate_ids"]
                ):
                    continue
                # base notation
                base_notation = m["notation"] + " | {}".format(
                    s["reaction_name"]
                )
                base_description = m["description"] + " | {}".format(
                    s["reaction_name"]
                )
                # calculate product
                base_product = s["reaction_smarts"].RunReactant(m["smiles"])[0]
                base_product = update_unmapped_atoms(base_product, 2)
                enums = m["enums"] | s["enums"] | {"TE"}
                # calculate terminal products
                for t in self.pk_module.terminal_rxns:
                    notation = base_notation + " | {}".format(t["reaction_id"])
                    description = base_description + " | {}".format(
                        t["reaction_id"]
                    )
                    # run reaction
                    product = t["reaction_smarts"].RunReactant(base_product)[0]
                    # determine atom restriction map
                    if t["reaction_id"] == "TE":
                        # terminal variable atom should not be carbon
                        # it should be oxygen or nitrogen for cyclic
                        terminal_var_atom = get_TE_atom(product)
                        atom_restriction_map = {terminal_var_atom: ["C"]}
                    else:
                        atom_restriction_map = {}
                    # determine atoms_to_ignore
                    atoms_to_ignore = get_idx_with_map_num(product, 2)
                    variable_atoms = get_variable_atoms(product)
                    atoms_to_ignore = list(
                        set(atoms_to_ignore + variable_atoms)
                    )
                    # determine bond descriptors
                    bond_map = get_bond_map(product, s["bonds"])
                    # determine variable map
                    variable_map = get_variable_map(
                        product, linkers=s["linkers"]
                    )
                    if s["consider_module_neighbors"] == False:
                        to_remove = set()
                        for x in variable_map:
                            # linker to next module is defined by {'pk', 'starter', 'aa'}
                            next_module_enums = {"pk", "starter", "aa"}
                            if next_module_enums.issubset(
                                set(variable_map[x])
                            ):
                                to_remove.add(x)
                        for n in to_remove:
                            del variable_map[n]
                    # prepare smarts
                    smarts = get_smarts(
                        product,
                        do_H_queries=True,
                        do_hybridization_queries=True,
                        do_heteroatom_n_queries=True,
                        do_atom_n_queries=True,
                        ignore_hydroxyls=True,
                        consider_carboxylic=True,
                        ignore_nitrogens=True,
                        consider_amide=True,
                        ignore_variable_groups=True,
                        atoms_to_ignore=atoms_to_ignore,
                        atom_restriction_map=atom_restriction_map,
                        bond_map=bond_map,
                    )
                    alpha_restriction_map = get_alpha_restriction_map(product)
                    self.library.append(
                        {
                            "notation": notation,
                            "description": description,
                            "smarts": smarts,
                            "smiles": product,
                            "ignore_patt_idx": list(atoms_to_ignore),
                            "enums": list(enums),
                            "variable_map": variable_map,
                            "alpha_restriction_map": alpha_restriction_map,
                            "priortization": m["priortization"],
                            "avoid_dereplication": s["avoid_dereplication"],
                            "uniquify": m["uniquify"],
                            "domain_id": m["domain_id"],
                            "domain_subclass_id": m["domain_subclass_id"],
                            "substrate_id": m["substrate_id"],
                            "substrate_family": m["substrate_family"],
                        }
                    )

    def add_standalone_motifs(self):
        for s in tqdm(
            self.pk_module.standalone_motifs,
            desc="Generating polyketide standalone units",
        ):
            smiles = get_natural_smiles(s["smiles"])
            mol = Chem.MolFromSmiles(smiles)
            map2idx = {
                a.GetAtomMapNum(): a.GetIdx()
                for a in mol.GetAtoms()
                if a.GetAtomMapNum() > 0
            }
            [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
            product = Chem.MolToSmiles(mol, canonical=False)
            variable_map = {
                map2idx[x]: y
                for x, y in zip(s["linker_atoms"], s["linker_atom_enums"])
            }
            smarts = get_smarts(
                product,
                do_H_queries=True,
                do_hybridization_queries=True,
                do_heteroatom_n_queries=True,
                do_atom_n_queries=True,
                ignore_hydroxyls=True,
                consider_carboxylic=True,
                ignore_nitrogens=True,
                consider_amide=True,
                ignore_variable_groups=True,
            )
            variable_atoms = get_variable_atoms(product)
            self.library.append(
                {
                    "notation": s["notation"],
                    "description": s["description"],
                    "smarts": smarts,
                    "smiles": product,
                    "ignore_patt_idx": list(variable_atoms),
                    "enums": list(s["enums"]) + ["standalone_pk"],
                    "alpha_restriction_map": [],
                    "variable_map": variable_map,
                }
            )

    def add_hash_ids(self):
        for unit in self.library:
            unit["hash_id"] = get_hash_id(unit)
