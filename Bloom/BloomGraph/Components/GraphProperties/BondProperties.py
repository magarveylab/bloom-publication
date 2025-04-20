from typing import Dict, Set, Tuple


class BondProperties:

    @property
    def bonds(self) -> Set[int]:
        return set(b.GetIdx() for b in self.mol.GetBonds())

    @property
    def bond_count(self) -> int:
        return len(self.bonds)

    @property
    def bond_to_atoms(self) -> Dict[int, Tuple[int, int]]:
        return {
            b.GetIdx(): (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
            for b in self.mol.GetBonds()
        }

    @property
    def atom_to_bonds(self) -> Dict[int, Set[int]]:
        cache = {a.GetIdx(): set() for a in self.mol.GetAtoms()}
        for b, atoms in self.bond_to_atoms.items():
            for a in atoms:
                cache[a].add(b)
        return cache

    @property
    def node_to_bonds(self) -> Dict[int, Set[int]]:
        atom_to_bonds = self.atom_to_bonds
        return {
            n: set(b for a in atoms for b in atom_to_bonds[a])
            for n, atoms in self.node_to_atoms.items()
        }

    @property
    def edge_to_bonds(self) -> Dict[Tuple[int, int], Set[int]]:
        # bond indexes in mols to edges
        cache = {e: set() for e in self.edges}
        for b in self.mol.GetBonds():
            a1 = b.GetBeginAtom().GetAtomMapNum()
            a2 = b.GetEndAtom().GetAtomMapNum()
            if a1 != a2:
                if (a1, a2) in cache:
                    cache[(a1, a2)].add(b.GetIdx())
                elif (a2, a1) in cache:
                    cache[(a2, a1)].add(b.GetIdx())
        return cache

    @property
    def bond_to_edges(self) -> Dict[int, Tuple[int, int]]:
        return {
            b: edge
            for edge, bonds in self.edge_to_bonds.items()
            for b in bonds
        }

    @property
    def interbonds(self) -> Set[int]:
        # molecular bonds between units
        return set(self.bond_to_edges)

    @property
    def intrabonds(self) -> Set[int]:
        # molecular bonds within units
        return self.bonds - self.interbonds

    @property
    def atom_to_interbond_count(self) -> Dict[int, int]:
        interbonds = self.interbonds
        return {a: len(b & interbonds) for a, b in self.atom_to_bonds.items()}

    @property
    def atom_to_intrabond_count(self) -> Dict[int, int]:
        intrabonds = self.intrabonds
        return {a: len(b & intrabonds) for a, b in self.atom_to_bonds.items()}

    @property
    def atom_to_bond_count(self) -> Dict[int, int]:
        return {a: len(b) for a, b in self.atom_to_bonds.items()}

    @property
    def bond_to_unique_atom_maps(self) -> Dict[int, Tuple[int, int]]:
        bond_to_atoms = self.bond_to_atoms
        return {
            b: (atoms[0] + 1, atoms[1] + 1)
            for b, atoms in bond_to_atoms.items()
        }

    @property
    def aromatic_bonds(self) -> Set[int]:
        return {b.GetIdx() for b in self.mol.GetBonds() if b.GetIsAromatic()}

    @property
    def atom_to_aromatic_bonds(self) -> Dict[int, Set[int]]:
        aromatic_bonds = self.aromatic_bonds
        return {
            a: bonds & aromatic_bonds
            for a, bonds in self.atom_to_bonds.items()
        }
