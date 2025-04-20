class AtomMethods:

    def correct_linker_atom_map_numbers(self):
        # the map number of linker atoms should correspond to the adjacent atom
        linker_atoms = self.linker_atoms
        for a in linker_atoms:
            atom = self.mol.GetAtomWithIdx(a)
            map_nums = set(n.GetAtomMapNum() for n in atom.GetNeighbors()) - {
                0
            }
            if len(map_nums) == 0:
                continue
            atom.SetAtomMapNum(list(map_nums)[0])
        # update mol in graph
        self.mol = self.mol

    def update_isotope(self, old: int, new: int):
        for a in self.mol.GetAtoms():
            if a.GetIsotope() == old:
                a.SetIsotope(new)
        self.mol = self.mol
