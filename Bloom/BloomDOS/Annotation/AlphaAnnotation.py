import logging
from typing import Set

from rdkit import Chem
from tqdm import tqdm

from Bloom.BloomDOS.Annotation.SmartsLibrary import smarts_library
from Bloom.BloomDOS.Graph.AlphaAnnotationGraph import AlphaAnnotationGraph


class AlphaAnnotation:
    # annotate alpha carbons of polyketides
    def __init__(self, mol: Chem.Mol, silent_output: bool = False):
        self.mol = mol
        self.silent_output = silent_output
        # graph stores annotations of the different alpha chain mappings
        self.G = AlphaAnnotationGraph(self.mol)
        self.non_alpha_carbons = set()
        self.alpha_carbons = set()

    def find_non_alpha_regions(self):
        # this is to speed up alpha search
        # avoid alpha carbon annotation to any atoms annotated by these libraries
        libraries_to_consider = ["sugar", "fattyacid", "aminoacid"]
        ignore_enums = {"hybrid", "aKT"}
        self.non_alpha_carbons = set()
        for name in tqdm(
            libraries_to_consider,
            desc="Searching for non alpha carbon regions",
            disable=self.silent_output,
        ):
            for smarts in smarts_library[name]:
                if len(ignore_enums & smarts.enums) > 0:
                    continue
                hits = smarts.GetSubstructMatches(self.mol)
                for h in hits:
                    self.non_alpha_carbons.update(h["patt"])

    def find_potential_alpha_sites(self):
        # find atoms that are potential alpha flags
        self.potential_C_adj_O = set()
        self.potential_C_alk = set()
        self.potential_C_sat = set()
        self.potential_C_starter = set()
        for smarts in tqdm(
            smarts_library["alphasearch"],
            desc="Searching with alphasearch library",
            disable=self.silent_output,
        ):
            hits = smarts.GetSubstructMatches(self.mol)
            for anno in hits:
                if "C_adj_O" in anno["enums"]:
                    self.potential_C_adj_O.update(anno["patt"])
                elif "C_alk" in anno["enums"]:
                    self.potential_C_alk.update(anno["patt"])
                elif "C_sat" in anno["enums"]:
                    self.potential_C_sat.update(anno["patt"])
                elif "C_starter" in anno["enums"]:
                    self.potential_C_starter.update(anno["patt"])

    def map_alpha_chains(self):
        # use terminal caboxlyic acid to map long alpha chain
        # profile the number of potential alpha sites that align with chain
        for smarts in tqdm(
            smarts_library["alphachain"],
            desc="Searching with alphachain library",
            disable=self.silent_output,
        ):
            hits = smarts.GetSubstructMatches(self.mol)
            for anno in hits:
                if len(anno["patt"] & self.non_alpha_carbons) > 0:
                    continue
                alpha_restrictions = anno["alpha_restrictions"]
                anno["C_adj_O"] = len(
                    self.potential_C_adj_O & alpha_restrictions
                )
                anno["C_alk"] = len(self.potential_C_alk & alpha_restrictions)
                anno["C_sat"] = len(self.potential_C_sat & alpha_restrictions)
                anno["C_start"] = len(
                    self.potential_C_starter & alpha_restrictions
                )
                self.G.add_node(anno)

    def dereplicate(self):
        # find annotations that are subset of each other
        edges = self.G.get_edges(overlap=True, subset=True)
        # remvoe chains based on alpha annotation
        to_remove = set()
        for x, y in edges:
            x_patt = self.G[x]["patt"]
            y_patt = self.G[y]["patt"]
            # check if one alpha chain is subset of another
            # if so remove the one with less alpha site annotations
            if x_patt.issubset(y_patt) or y_patt.issubset(x_patt):
                # remove chaings by carbon adjacent to oxygen
                # this is most confident potential alpha site (KS, KR)
                # if its equal, then use carbons part of starters
                # if its equal, then use carbons part of double bond systems (DH)
                # if its equal, then use carbons part of sat systems (ER)
                keys = ["C_adj_O", "C_start", "C_alk", "C_sat"]
                for k in keys:
                    x_value = self.G[x][k]
                    y_value = self.G[y][k]
                    if x_value > y_value:
                        to_remove.add(y)
                        break
                    elif y_value > x_value:
                        to_remove.add(x)
                        break
                    else:
                        continue
        # remove dereplicated sites
        self.G.remove_nodes_from(to_remove)

    def get_best_annotation(self) -> Set[int]:
        # return atom indexes corresponding to confirmed alpha sites
        # function to priortize group of non-overlapping annotations
        # find non-overlap groups of chains
        logging.info("Finding Alpha Carbons: non-overlapping annotations")
        nodes = self.G.get_nodes()
        if len(nodes) > 0:
            nonoverlap_groups = self.G.get_nonoverlap_groups_from(nodes)
            confirmed_alpha_sites = set()
            for group in nonoverlap_groups:
                for n in group:
                    alpha_restrictions = self.G[n]["alpha_restrictions"]
                    confirmed_alpha_sites.update(alpha_restrictions)
            self.alpha_carbons = confirmed_alpha_sites
