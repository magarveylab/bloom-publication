import itertools as it
import logging
from collections import Counter
from typing import List, Set, Tuple

import networkx as nx

from Bloom.BloomDOS.Graph.AnnotationGraph import AnnotationGraph


def overlap_stretches(stretches: List[Set[int]]):
    # find overlapping nodes between stretches
    # this is used to find confident nodes that can be used
    # as a template for dereplication
    overlap = nx.Graph()
    overlap_nodes = set()
    for idx, s in enumerate(stretches):
        overlap.add_node(idx)
    for s1, s2 in it.combinations(overlap.nodes, 2):
        if len(stretches[s1] & stretches[s2]) > 0:
            overlap.add_edge(s1, s2)
    for group in nx.connected_components(overlap):
        nodes = Counter([n for s in group for n in stretches[s]])
        overlap_nodes.update(
            [n for n, count in nodes.items() if count == len(group)]
        )
    return overlap_nodes


class PolyketideDereplication:

    def __init__(self, G: AnnotationGraph):
        self.G = G

    def remove_by_alpha_carbon(self):
        logging.info(
            "Removing polyketide annotations by mismatch on alpha carbons"
        )
        # subset overlap in polyketide nodes
        ignore_nodes = self.G.get_nodes(enums={"starter", "DH-shift"})
        nodes = self.G.get_nodes(enums={"pk"})
        # n1 and n2 overlap
        # n1 is subset of n2
        # edge can be used for dereplication
        edges_to_analyze = self.G.get_edges(
            overlap=True,
            nodes=nodes - ignore_nodes,
            subset=True,
            avoid_derep=False,
        )
        # does alpha carbon count suggest that n1 should be removed
        to_remove = set()
        for n1, n2 in edges_to_analyze:
            n1_alpha_hit = self.G[n1]["alpha_hit"]
            n2_alpha_hit = self.G[n2]["alpha_hit"]
            # does alpha hit suggest that n1 should be removed
            if n2_alpha_hit > n1_alpha_hit:
                to_remove.add(n1)
        self.G.remove_nodes_from(to_remove)

    def remove_by_dh_shifts(self, max_size: int = 4):
        # removing DH shifts by conflicts
        logging.info("Removing DH shifts by alpha conflicts")
        dh_shift_nodes = self.G.get_nodes(enums={"DH-shift"})
        to_remove = set()
        if len(dh_shift_nodes) > 0:
            to_remove.update(
                set(n for n in dh_shift_nodes if self.G[n]["alpha_hit"] == 0)
            )
        self.G.remove_nodes_from(to_remove)
        # removing DH shifts by stretch
        logging.info("Removing DH shifts by length")
        keep = set()
        to_remove = set()
        for shift_direction in ["DH-shift-forward", "DH-shift-backward"]:
            dh_shift_nodes = self.G.get_nodes(enums={shift_direction})
            stretches = self.G.get_continous_stretches_from(dh_shift_nodes)
            for s in stretches:
                if len(s) > max_size:
                    to_remove.update(s)
                else:
                    keep.update(s)
        to_remove = to_remove - keep
        if len(to_remove) > 0:
            self.G.remove_nodes_from(to_remove)

    def remove_by_unit_type(self):

        def get_pk_prioritzation_score(enums):
            if "pk_keto" in enums:
                return 1
            elif "pk_beta_branch" in enums:
                return 2
            elif "pk_canonical" in enums:
                return 3

        # extract polyketide edges
        logging.info("Removing polyketide annotations by probability")
        # n1 and n2 overlap
        # n1 is subset of n2
        # edge can be used for dereplication
        nodes = self.G.get_nodes(enums={"pk"}) - self.G.get_nodes(
            enums={"starter", "pk_specialized"}
        )
        edges_to_analyze = self.G.get_edges(
            overlap=True, nodes=nodes, subset=True, avoid_derep=False
        )
        # polyketide units fall under 3 categories:
        # beta keto, beta branch, canonical
        # priortization is based on how common those units are in natural products
        pk_node_scores = {}
        for n in nodes:
            enums = self.G[n]["enums"]
            pk_node_scores[n] = get_pk_prioritzation_score(enums)
        # does pk score suggest that n1 should be removed
        # keep nodes with higher score
        # if equal, do not remove any nodes
        to_remove = set()
        for n1, n2 in edges_to_analyze:
            n1_score = pk_node_scores[n1]
            n2_score = pk_node_scores[n2]
            if n1_score != None and n2_score != None:
                if n2_score > n1_score:
                    to_remove.add(n1)
        self.G.remove_nodes_from(to_remove)

    def remove_by_polyether_stretches(self, size: int = 4):
        logging.info("Removing pk annotations by polyether stretch")
        pe_nodes = self.G.get_nodes(enums={"polyether"})
        if len(pe_nodes) > 0:
            polyether_stretches = self.G.get_continous_stretches_from(pe_nodes)
            # apply size filter
            polyether_stretches = [
                s for s in polyether_stretches if len(s) >= size
            ]
            # find common nodes between stretches
            confident_nodes = overlap_stretches(polyether_stretches)
            if len(confident_nodes) > 0:
                # use these common nodes to remove polyketide units
                nodes_to_examine = self.G.get_nodes(enums={"pk"})
                nodes_to_examine = nodes_to_examine - confident_nodes
                confident_atoms = self.G.get_atoms_from_nodes(confident_nodes)
                to_remove = set()
                for n in nodes_to_examine:
                    if self.G[n]["patt"].issubset(confident_atoms):
                        to_remove.add(n)
                self.G.remove_nodes_from(to_remove)

    def remove_by_polyketide_stretches(self, size: int = 4):
        logging.info("Removing annotations by confident modular stretches")
        # calculate modular stretches
        ignore_enums = {"curated_module"}
        pk_nodes = self.G.get_nodes(enums={"pk"})
        if len(pk_nodes) > 0:
            # include aa_nodes for modular stretch determination
            aa_nodes = self.G.get_nodes(enums={"aa"})
            # calculate modular stretches
            modular_nodes = pk_nodes | aa_nodes
            modular_stretches = self.G.get_continous_stretches_from(
                modular_nodes
            )
            # apply size filter
            modular_stretches = [
                s for s in modular_stretches if len(s) >= size
            ]
            # find common nodes between stretches
            confident_nodes = overlap_stretches(modular_stretches)
            if len(confident_nodes) > 0:
                # use these common nodes to remove polyketide units
                nodes_to_examine = self.G.get_nodes(enums={"pk"})
                nodes_to_examine = nodes_to_examine - confident_nodes
                confident_atoms = self.G.get_atoms_from_nodes(confident_nodes)
                to_remove = set()
                for n in nodes_to_examine:
                    if len(self.G[n]["enums"] & ignore_enums) > 0:
                        continue
                    if self.G[n]["patt"].issubset(confident_atoms):
                        to_remove.add(n)
                self.G.remove_nodes_from(to_remove)
