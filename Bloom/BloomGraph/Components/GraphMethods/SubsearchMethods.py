from typing import List, Tuple

from rdkit import Chem


class SubsearchMethods:

    def search_units_by_smarts(self, smarts: str) -> List[Tuple]:
        # retruns units that hit smarts
        smarts_mol = Chem.MolFromSmarts(smarts)
        atom_to_node = self.atom_to_node
        hits = self.mol.GetSubstructMatches(smarts_mol)
        return [set(atom_to_node[a] for a in h) for h in hits]
