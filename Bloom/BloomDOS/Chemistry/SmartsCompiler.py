from typing import Dict, List, Tuple

from rdkit import Chem

from Bloom.BloomDOS.Chemistry.SmartsQueryBuilder import SmartsQueryBuilder
from Bloom.BloomDOS.Chemistry.SmartsWriter import SmartsWriter


def get_simple_smarts(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    smarts = Chem.MolToSmarts(mol)
    smarts = smarts.replace("#0", "*")
    return smarts


def get_smarts(
    smiles: str,
    stereochemistry: bool = False,
    do_H_queries: bool = True,
    do_hybridization_queries: bool = True,
    do_heteroatom_n_queries: bool = True,
    do_atom_n_queries: bool = True,
    do_charge_queries: bool = False,
    ignore_hydroxyls: bool = True,
    ignore_phosphate_oxygens: bool = True,
    ignore_nitrogens: bool = False,
    ignore_thiols: bool = False,
    consider_carboxylic: bool = False,
    consider_amide: bool = False,
    limit_to_carbon_queries: bool = False,
    atoms_to_ignore: List[int] = [],
    ring_atoms_to_ignore: List[int] = [],
    consider_ring_membership: bool = False,
    ignore_variable_groups: bool = True,
    add_variable_n_map: Dict[int, int] = {},
    atom_restriction_map: Dict[int, List[str]] = {},
    bond_map: Dict[Tuple[int, int], str] = {},
    variable_bond_queries: bool = False,
):
    # extract smarts queries from structure
    smarts_query = SmartsQueryBuilder(
        smiles=smiles,
        stereochemistry=stereochemistry,
        limit_to_carbon_queries=limit_to_carbon_queries,
        do_H_queries=do_H_queries,
        do_hybridization_queries=do_hybridization_queries,
        do_heteroatom_n_queries=do_heteroatom_n_queries,
        do_atom_n_queries=do_atom_n_queries,
        ignore_hydroxyls=ignore_hydroxyls,
        ignore_phosphate_oxygens=ignore_phosphate_oxygens,
        ignore_nitrogens=ignore_nitrogens,
        ignore_thiols=ignore_thiols,
        atoms_to_ignore=atoms_to_ignore,
        ring_atoms_to_ignore=ring_atoms_to_ignore,
        consider_carboxylic=consider_carboxylic,
        consider_amide=consider_amide,
        ignore_variable_groups=ignore_variable_groups,
        consider_ring=consider_ring_membership,
        add_variable_n_map=add_variable_n_map,
        variable_bond_queries=variable_bond_queries,
    )
    # write smarts with extracted queries
    smarts = SmartsWriter(
        smiles=smiles,
        stereochemistry_map=smarts_query.stereochemistry_map,
        atom_H_map=smarts_query.atom_H_map,
        atom_hybrid_map=smarts_query.atom_hybrid_map,
        atom_n_map=smarts_query.atom_n_map,
        atom_heteroatom_n_map=smarts_query.atom_heteroatom_n_map,
        atom_charge_map=smarts_query.atom_charge_map,
        atom_variable_n_map=smarts_query.atom_variable_n_map,
        add_variable_n_map=smarts_query.add_variable_n_map,
        ring_membership_map=smarts_query.ring_membership_map,
        atom_restriction_map=atom_restriction_map,
        bond_map=bond_map,
    )
    return smarts.smarts
