import itertools as it
from typing import Set

import networkx as nx
from rdkit import Chem


class PolyketideProperties:

    @property
    def macrolide_type(self) -> bool:
        pks_nodes = self.pks_nodes
        module_nodes = self.module_nodes
        terminal_nodes = self.terminal_nodes
        # make macrolides at least 5 modular units
        if len(module_nodes) < 5:
            return None
        # at least 50 percent of the modular nodes should be polyketide nodes
        if len(pks_nodes & module_nodes) / len(module_nodes) < 0.5:
            return None
        # check for ester bond
        ester_smarts = "[CX3](=O)[OX2H0]"
        hits = self.search_units_by_smarts(ester_smarts)
        # filter hits to contain terminal nodes and module nodes
        terminal_hits = [
            h
            for h in hits
            if len(h & terminal_nodes) > 0 and h.issubset(module_nodes)
        ]
        if len(terminal_hits) > 0:
            terminal_nodes = set(n for h in terminal_hits for n in h)
            cycle_lengths = []
            for n in terminal_nodes:
                try:
                    cycle_lengths.append(
                        nx.find_cycle(self.G, source=n, orientation="original")
                    )
                except:
                    continue
            if len(cycle_lengths) > 0:
                max_cycle_length = max(len(c) for c in cycle_lengths)
                if max_cycle_length >= 5:
                    return {"macrolide", f"macrolide ({max_cycle_length})"}
                else:
                    return None
            else:
                return None
        else:
            return None

    @property
    def is_enediyne(self):
        # SMARTS pattern for enediyne
        enedyne_smarts = "[C]=[C]-[C]#[C]"
        smarts_mol = Chem.MolFromSmarts(enedyne_smarts)
        hits = self.mol.GetSubstructMatches(smarts_mol)
        # group the two sides of the enediyne
        g = nx.Graph()
        for h in hits:
            g.add_node(h)
        for a, b in it.combinations(hits, 2):
            if len(set(a) & set(b)) == 2:
                g.add_edge(a, b)
        groups = list(nx.connected_components(g))
        final_hits = [set(a for h in g for a in h) for g in groups]
        # enediyne must have a hit with 6 atoms
        final_hits = [h for h in final_hits if len(h) == 6]
        if len(final_hits) > 0:
            return True
        else:
            return False

    @property
    def is_polyether(self):
        polyether_nodes = self.get_nodes(["polyether"])
        return True if len(polyether_nodes) > 0 else False

    @property
    def is_decalin(self):
        decalin_smarts = [
            "[#6]12-[#6]=[#6]-[#6]-[#6]-[#6]-1-[#6]-[#6]-[#6]-[#6]-2",
            "[#6]12-[#6]-[#6]=[#6]-[#6]-[#6]-1-[#6]-[#6]-[#6]-[#6]-2",
        ]
        for s in decalin_smarts:
            smarts_mol = Chem.MolFromSmarts(s)
            hits = self.mol.GetSubstructMatches(smarts_mol)
            if len(hits) > 0:
                return True
        else:
            return False

    @property
    def is_spiral(self):
        spiral_nodes = self.get_nodes(["spiral"])
        return True if len(spiral_nodes) > 0 else False

    @property
    def is_beta_branch(self):
        beta_branches = self.get_nodes(["pk_beta_branch"])
        return True if len(beta_branches) >= 2 else False

    @property
    def is_aromatic(self):
        aromatic_enums = [
            "C1-KS-R1",
            "C1-KR-I2",
            "C1-KS-A3",
            "C2-KR-A1",
            "C2-KS-R2",
            "C3-KS-A1",
            "C3-KR-I2",
            "C3-KS-R3",
            "C7-KS-A1",
            "C7-TE-R",
        ]
        aromatic_nodes = self.get_nodes(aromatic_enums)
        return True if len(aromatic_nodes) > 0 else False

    @property
    def is_pyran(self):
        pyran_nodes = self.get_nodes(["pyran"])
        return True if len(pyran_nodes) > 0 else False

    @property
    def is_polyene(self):
        pks_nodes = self.pks_nodes
        if len(pks_nodes) == 0:
            return False
        dh_nodes = self.get_nodes(["DH"]) & pks_nodes
        dh_stretches = list(nx.connected_components(self.G.subgraph(dh_nodes)))
        for stretch in dh_stretches:
            if len(stretch) >= 4:
                return True
        else:
            return False

    @property
    def is_saturated(self):
        pks_nodes = self.pks_nodes
        if len(pks_nodes) == 0:
            return False
        er_nodes = self.get_nodes(["ER"]) & pks_nodes
        er_stretches = list(nx.connected_components(self.G.subgraph(er_nodes)))
        for stretch in er_stretches:
            if len(stretch) >= 4:
                return True
        else:
            return False

    @property
    def is_polyol(self):
        pks_nodes = self.pks_nodes
        if len(pks_nodes) == 0:
            return False
        kr_nodes = self.get_nodes(["KR"]) & pks_nodes
        kr_stretches = list(nx.connected_components(self.G.subgraph(kr_nodes)))
        for stretch in kr_stretches:
            if len(stretch) >= 4:
                return True
        else:
            return False

    @property
    def is_linear(self):
        pks_nodes = self.pks_nodes
        if len(pks_nodes) == 0:
            return False
        module_nodes = self.module_nodes
        terminal_nodes = self.terminal_nodes
        # at least 50 percent of the modular nodes should be polyketide nodes
        if len(pks_nodes & module_nodes) / len(module_nodes) < 0.5:
            return False
        # check for carboxylic bond
        carbocylic_smarts = "[CX3](=O)[OX2H1]"
        hits = self.search_units_by_smarts(carbocylic_smarts)
        # filter hits to contain terminal nodes and module nodes
        terminal_hits = [h for h in hits if len(h & terminal_nodes) > 0]
        if len(terminal_hits) > 0:
            return True
        else:
            return False

    @property
    def is_hybrid(self):
        pks_nodes = self.pks_nodes
        if len(pks_nodes) == 0:
            return False
        nrps_nodes = self.nrps_nodes
        module_nodes = self.module_nodes
        pk_percent = len(pks_nodes & module_nodes) / len(module_nodes)
        nrps_percent = len(nrps_nodes & module_nodes) / len(module_nodes)
        if pk_percent >= 0.5 and nrps_percent >= 0.2:
            return True
        else:
            return False

    @property
    def is_polycyclic(self):
        ring_enums = ["C5-I2", "C5-KS-R3", "C4-KR-A1", "C4-MeDH-R2"]
        ring_nodes = self.get_nodes(ring_enums)
        return True if len(ring_nodes) > 0 else False

    @property
    def polyketide_subclasses(self) -> Set[str]:
        subclasses = set()
        macrolide_type = self.macrolide_type
        if macrolide_type != None:
            subclasses.update(macrolide_type)
        if self.is_enediyne:
            subclasses.add("enediyne")
        if self.is_polyether:
            subclasses.add("polyether")
        if self.is_spiral:
            subclasses.add("spiral")
        if self.is_beta_branch:
            subclasses.add("beta_branch")
        if self.is_aromatic:
            subclasses.add("aromatic")
        if self.is_pyran:
            subclasses.add("pyran")
        if self.is_polyene:
            subclasses.add("polyene")
        if self.is_saturated:
            subclasses.add("saturated")
        if self.is_polyol:
            subclasses.add("polyol")
        if self.is_linear:
            subclasses.add("linear")
        if self.is_hybrid:
            subclasses.add("hybrid")
        if self.is_polycyclic:
            subclasses.add("polycyclic")
        return list(subclasses)
