from typing import Dict, Set

from rdkit import Chem


class AtomProperties:

    @property
    def total_atoms(self) -> Set[int]:
        return set(a for n in self.nodes for a in self[n]["atoms"])

    @property
    def total_atom_count(self) -> int:
        return len(self.total_atoms)

    @property
    def unknown_atom_count(self) -> int:
        return len(self.unknown_atoms)

    @property
    def atom_coverage(self) -> float:
        unknown_atom_count = self.unknown_atom_count
        total_atom_count = self.total_atom_count
        return round(
            (total_atom_count - unknown_atom_count) / total_atom_count, 2
        )

    @property
    def atom_to_node(self) -> Dict[int, int]:
        return {a.GetIdx(): a.GetAtomMapNum() for a in self.mol.GetAtoms()}

    @property
    def unknown_atoms(self) -> Set[int]:
        return self.get_atoms_from_nodes(self.unknown_nodes)

    @property
    def heterogenous_atoms(self) -> Set[int]:
        # corresponds to O, N, S
        return {
            a.GetIdx()
            for a in self.mol.GetAtoms()
            if a.GetAtomicNum() in [7, 8, 16]
        }

    @property
    def linker_atoms(self) -> Set[int]:
        # linker atoms are defined by isotope numbers 1, 2, 3, 4
        isotopes = [1, 2, 3, 4, 5]
        cache = set()
        for iso_num in isotopes:
            smarts = Chem.MolFromSmarts(f"[{iso_num}*]")
            hits = self.mol.GetSubstructMatches(smarts)
            cache.update({h[0] for h in hits})
        return cache
