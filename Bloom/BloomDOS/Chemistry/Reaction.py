from typing import List, Optional, Union

import rdkit
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from Bloom.BloomDOS.Chemistry.ChemUtils import validate_smiles

# ignore rdkit warnings
# these warnings correspond to mapped indexes in SMARTS
rdkit.RDLogger.DisableLog("rdApp.*")


class ReactionSmarts:

    def __init__(self, smarts: str):
        self.smarts = smarts
        self.rxn = rdChemReactions.ReactionFromSmarts(smarts)

    def RunReactant(self, smiles: str) -> List[str]:
        # note reactions are single templates
        products = []
        reactant = Chem.MolFromSmiles(smiles)
        # run reaction
        ps = self.rxn.RunReactants((reactant,))
        for pset in ps:
            smiles = []
            for p in pset:
                # update atoms with map number from reactant
                for a in p.GetAtoms():
                    if a.HasProp("react_atom_idx"):
                        raidx = int(a.GetProp("react_atom_idx"))
                        mno = reactant.GetAtomWithIdx(raidx).GetAtomMapNum()
                        a.SetAtomMapNum(mno)
                smiles.append(Chem.MolToSmiles(p))
            product_smiles = ".".join(smiles)
            if validate_smiles(product_smiles):
                # remove explicit hydrogens
                product_smiles = Chem.MolToSmiles(
                    Chem.MolFromSmiles(product_smiles)
                )
                products.append(product_smiles)
        return list(set(products))

    def RecursiveRunReactant(
        self, smiles: str, return_last: bool = False
    ) -> Union[List[str], Optional[str]]:
        cache = {}
        final_products = set()
        interm_products = set(self.RunReactant(smiles))
        last_iteration = None
        iteration = 1
        while len(interm_products) > 0:
            # remove products already seen
            interm_products = interm_products - final_products
            cache[iteration] = list(interm_products)
            if len(interm_products) > 0:
                last_iteration = int(iteration)
            # cache products
            final_products.update(interm_products)
            # run reactions
            to_add = set()
            for smiles in interm_products:
                to_add.update(self.RunReactant(smiles))
            interm_products.update(to_add)
            iteration += 1
        if return_last:
            if last_iteration == None:
                return None
            else:
                return cache[last_iteration][0]
        else:
            return list(final_products)


####################################################################
# Defined Reactions
####################################################################

aa_tautomerization = ReactionSmarts(
    "[#8:4]/[#6:1](-[*:2])=[#7:3]>>[#8H0:4]=[#6:1](-[*:2])-[#7:3]-[H]"
)
keto_tautomerization_1 = ReactionSmarts(
    "[*:1]-[#6:2](/[#6:4]=[#6:5](/[*:7])-[#8:6])=[#8:3]>>[*:1]-[#6:2](-[#6:4]-[#6:5](-[*:7])=[#8:6])=[#8:3]"
)
keto_tautomerization_2 = ReactionSmarts(
    "[*:1]-[#6:2](-[#8:3])/[#6:4]=[#6:5](/[*:7])-[#8:6]>>[*:1]-[#6:2](-[#8:3])-[#6:4]-[#6:5](-[*:7])=[#8:6]"
)
dethiolation = ReactionSmarts("[*:1]-[#6](-[#16])=[#8]>>[*:1]-[#0]")
anomeric_clevage = ReactionSmarts(
    "[#6z2:1](-[#8D{2-2}z0:2])-[#8H:3]>>[*]-[#6:1]-[#8:2]"
)
