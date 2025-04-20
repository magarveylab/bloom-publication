import timeout_decorator
from molvs import Standardizer
from rdkit import Chem
from rdkit.Chem import rdmolops

from Bloom.BloomDOS.Chemistry.Reaction import (
    aa_tautomerization,
    keto_tautomerization_1,
    keto_tautomerization_2,
)

####################################################################
# Standardization
####################################################################


class MoleculeStandardizer:

    def __init__(
        self,
        standardize_canonical: bool = True,
        fast_standardize: bool = True,
        standardize_tautomers: bool = True,
        neutralize: bool = False,
        remove_radicals: bool = False,
        remove_map_numbers: bool = False,
        remove_stereochemistry: bool = False,
    ):
        self.standardize_canonical = standardize_canonical
        self.fast_standardize = fast_standardize
        self.standardize_tautomers = standardize_tautomers
        self.neutralize = neutralize
        self.remove_radicals = remove_radicals
        self.remove_map_numbers = remove_map_numbers
        self.remove_stereochemistry = remove_stereochemistry
        self.standardizer = Standardizer()

    def standardize_smiles(self, smiles: str):
        if self.standardize_canonical:
            smiles = self.canonicalize(smiles)
        if self.remove_map_numbers:
            smiles = self.remove_map_numbers_from_smiles(smiles)
        if self.remove_radicals:
            smiles = self.eliminate_radicals(smiles)
        if self.neutralize:
            smiles = self.neutralize_atoms(smiles)
        if self.remove_stereochemistry:
            smiles = self.remove_stereochemistry_from_smiles(smiles)
        # option to run other tautomerization reactions, recommended for bear
        if self.standardize_tautomers:
            tmp_smiles = self.standardize_tautomerization(smiles)
            if tmp_smiles != None:
                smiles = tmp_smiles
        # tautomerize amide bonds and pk bonds (using smarts reactions)
        if self.fast_standardize:
            smiles = self.run_fast_standardization(smiles)
        return smiles

    def canonicalize(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        smol = self.standardizer.standardize(mol)
        return Chem.MolToSmiles(smol)

    def remove_map_numbers_from_smiles(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        for a in mol.GetAtoms():
            a.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol)

    def neutralize_atoms(self, smiles: str):
        new_smiles = []
        smiles_tokens = smiles.split(".")
        for smiles_tok in smiles_tokens:
            mol = Chem.MolFromSmiles(smiles_tok)
            pattern = Chem.MolFromSmarts(
                "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]"
            )
            at_matches = mol.GetSubstructMatches(pattern)
            at_matches_list = [y[0] for y in at_matches]
            if len(at_matches_list) > 0:
                for at_idx in at_matches_list:
                    atom = mol.GetAtomWithIdx(at_idx)
                    chg = atom.GetFormalCharge()
                    hcount = atom.GetTotalNumHs()
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(hcount - chg)
                    atom.UpdatePropertyCache()
            new_smiles_tok = Chem.MolToSmiles(mol)
            if Chem.MolFromSmiles(new_smiles_tok) != None:
                new_smiles.append(new_smiles_tok)
            else:
                new_smiles.append(smiles_tok)
        return ".".join(new_smiles)

    def eliminate_radicals(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        for a in mol.GetAtoms():
            if a.GetNumRadicalElectrons() == 1:
                a.SetNumRadicalElectrons(0)
                a.SetFormalCharge(a.GetFormalCharge() - 1)
        return Chem.MolToSmiles(mol)

    def remove_stereochemistry_from_smiles(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        rdmolops.RemoveStereochemistry(mol)
        mol = Chem.RemoveHs(mol)
        return Chem.MolToSmiles(mol)

    def run_fast_standardization(self, smiles: str):
        reactions = [
            aa_tautomerization,
            keto_tautomerization_1,
            keto_tautomerization_2,
        ]
        for rxn in reactions:
            new_smiles_tokens = []
            smiles_tokens = smiles.split(".")
            for smiles_tok in smiles_tokens:
                new_smiles_tok = rxn.RecursiveRunReactant(
                    smiles_tok, return_last=True
                )
                if new_smiles_tok == None:
                    new_smiles_tokens.append(smiles_tok)
                else:
                    new_smiles_tokens.append(new_smiles_tok)
            smiles = ".".join(new_smiles_tokens)
        return smiles

    @timeout_decorator.timeout(600)
    def standardize_tautomerization(self, smiles: str):
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol2 = self.standardizer.tautomer_parent(mol)
            return Chem.MolToSmiles(mol2)
        except:
            return None
