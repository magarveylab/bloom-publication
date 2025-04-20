import logging
from typing import List, Set, Tuple

from rdkit import Chem
from tqdm import tqdm

from Bloom.BloomDOS.Annotation.SmartsLibrary import (
    library_filters,
    smarts_library,
)
from Bloom.BloomDOS.Graph.AnnotationGraph import AnnotationGraph


def get_variable_response(patt: Tuple[int, ...]):
    return {
        "patt": set(patt),
        "description": "variable",
        "notation": "R",
        "patt_len": len(patt),
        "original_patt_len": len(patt),
        "enums": {"variable"},
        "avoid_dereplication": True,
        "can_dereplicate": False,
        "alpha_hit": 0,
        "variable_restrictions": {},
        "unknown_restrictions": set(),
        "priortization": 1,
    }


class UnitAnnotation:

    def __init__(
        self,
        mol: Chem.Mol,
        G: AnnotationGraph,
        alpha_carbons: Set[int],
        skip_enums: List[str],
        only_consider_enums: List[str],
        additional_library: dict,
        silent_output: bool = False,
    ):

        self.mol = mol
        self.G = G
        self.alpha_carbons = alpha_carbons
        self.enums_to_skip = set(skip_enums)
        self.only_consider_enums = set(only_consider_enums)
        self.additional_library = additional_library
        self.silent_output = silent_output
        # Compile unit search library
        self.unit_library = {k: v for k, v in smarts_library.items()}
        self.unit_library.update({k: v for k, v in additional_library.items()})
        # results
        self.solutions = []

    def limit_unit_library(self):
        # certain basic patterns are indicative of what library to use
        # such as the cyclic ring in azole systems
        detected_enum = set()
        all_filtered_enums = set()
        for f in library_filters:
            all_filtered_enums.add(f["enum"])
            if self.mol.HasSubstructMatch(f["smarts"]) == True:
                detected_enum.add(f["enum"])
        self.enums_to_skip.update(all_filtered_enums - detected_enum)

    def find_units(self):
        # avoid atoms
        avoid_atoms = set()
        for atom in self.mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                anno = get_variable_response((atom.GetIdx(),))
                # cache variable annotation
                self.G.add_node(anno)
                # avoid mapping to variable atoms in the future
                avoid_atoms.add(atom.GetIdx())
        # smarts search
        for name, library in self.unit_library.items():
            message = f"Searching with {name} library"
            # skip annnotations from alpha library (already done)
            if name in ["alphasearch", "alphachain"]:
                continue
            for smarts in tqdm(
                library, desc=message, disable=self.silent_output
            ):
                # Don't skip library if it's additional units, even if they have other enums
                if name != "additional":
                    # skip enums based on filter process
                    if len(smarts.enums & self.enums_to_skip) > 0:
                        continue
                    if (
                        len(self.only_consider_enums) > 0
                        and len(smarts.enums & self.only_consider_enums) == 0
                    ):
                        continue
                # annotation
                hits = smarts.GetSubstructMatches(
                    self.mol,
                    avoid_atoms=avoid_atoms,
                    alpha_carbons=self.alpha_carbons,
                )
                for anno in hits:
                    self.G.add_node(anno)

    def get_best_annotated_coverage(self):
        # find solution
        logging.info(
            "Find unit representation of molecule: non-overlapping annotations"
        )
        # all nodes
        nodes = self.G.get_nodes()
        # flexible nodes
        misc_nodes = self.G.get_nodes(enums={"miscellaneous"})
        # nonmodular nodes
        nonmodular_nodes = self.G.get_nodes(enums={"nonmodular"})
        # flexible nodes
        flexible_nodes = self.G.get_nodes(enums={"flexible"})
        # primary metabolite nodes
        primary_nodes = self.G.get_nodes(enums={"primary_metabolite"})
        # find solutions with different subset of nodes
        phases = [
            nodes
            - misc_nodes
            - nonmodular_nodes
            - flexible_nodes
            - primary_nodes,
            nodes - nonmodular_nodes - flexible_nodes - primary_nodes,
            nodes - flexible_nodes - primary_nodes,
            nodes - primary_nodes,
            nodes,
        ]
        # total atom annotation
        total_atom_count = self.G.atom_count
        last_phase = len(phases)
        for idx, nodes_to_search_with in enumerate(phases, 1):
            if len(nodes_to_search_with) == 0:
                continue
            if (
                idx != 1
                and idx != last_phase
                and phases[idx - 1] == phases[idx - 2]
            ):
                continue
            solutions = self.G.get_nonoverlap_groups_from(nodes_to_search_with)
            best_atom_coverage = max(
                [len(self.G.get_atoms_from_nodes(s)) for s in solutions]
            )
            if best_atom_coverage == total_atom_count:
                self.solutions = solutions
                break
            elif idx == last_phase:
                self.solutions = solutions
            else:
                continue

    def apply_unknown_restrictions(self):
        # compile all annotations
        new_solutions = []
        for final_annotations in self.solutions:
            logging.info(
                "Removing unconfident annotations adjacent to unknown regions"
            )
            annotated_atoms = self.G.get_atoms_from_nodes(final_annotations)
            # remove annotations that are adjacent to unknown regions
            # based on unknown restrictions
            to_remove = set()
            for a in final_annotations:
                search = self.G[a]["unknown_restrictions"]
                if (
                    len(search) > 0
                    and search.issubset(annotated_atoms) == False
                ):
                    to_remove.add(a)
            new_solutions.append(final_annotations - to_remove)
        self.solutions = new_solutions

    def dereplicate_solutions(self):
        # dereplicate solutions by notation and annotated atoms
        cache = {}
        for s in self.solutions:
            sid = tuple(
                sorted(
                    [
                        tuple(
                            [self.G[n]["notation"].split(" | ")[0]]
                            + sorted(self.G[n]["patt"])
                        )
                        for n in s
                    ]
                )
            )
            cache[sid] = s
        self.solutions = list(cache.values())
