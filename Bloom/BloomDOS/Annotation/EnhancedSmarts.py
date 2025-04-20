import itertools as it
from typing import Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
from rdkit import Chem

from Bloom.BloomDOS.DataStructs import (
    EnhancedSmartsDict,
    EnhancedSmartsResponse,
)

####################################################################
# Helper Functions
####################################################################


def get_value(
    dictionary: dict, key: str, default: str, cast: Optional[Callable] = None
):
    result = dictionary.get(key, default)
    if result == None:
        return default
    else:
        if cast != None:
            return cast(result)
        else:
            return result


def reformat_int2set(dictionary: dict) -> Dict[int, set]:
    return {int(k): set(v) for k, v in dictionary.items()}


####################################################################
# SMARTS for molecule annotation
####################################################################


class EnhancedSmarts:

    def __init__(
        self,
        smarts: str,
        notation: str,
        description: str,
        smiles: Optional[str],
        ignore_patt_idx: List[int],
        enums: Set[str],
        avoid_dereplication: bool,
        can_dereplicate: bool,
        variable_map: Dict[int, Set[str]],
        variable_to_atomic_num_map: Dict[int, int],
        unknown_restriction_map: Set[int],
        alpha_restriction_map: Set[int],
        priortization: int,
        uniquify: bool,
        hash_id: str,
    ):
        # variables
        self.smarts = Chem.MolFromSmarts(smarts)
        self.notation = notation
        self.description = description
        self.smiles = smiles
        self.ignore_patt_idx = ignore_patt_idx
        self.enums = enums
        self.avoid_dereplication = avoid_dereplication
        self.can_dereplicate = can_dereplicate
        self.variable_map = variable_map
        self.variable_to_atomic_num_map = variable_to_atomic_num_map
        self.unknown_restriction_map = unknown_restriction_map
        self.alpha_restriction_map = alpha_restriction_map
        self.priortization = priortization
        self.uniquify = uniquify
        self.hash_id = hash_id

    def GetSubstructMatches(
        self,
        mol: Chem.Mol,
        avoid_atoms: Set[int] = set(),
        limit_atoms: Set[int] = set(),
        alpha_carbons: Set[int] = set(),
    ) -> List[EnhancedSmartsResponse]:
        # avoid atoms -> substructure matches cannot overlap with these atoms
        # limit atoms -> substructure matches must be a subset of these atoms
        # alpha_carbons -> must match to alpha atoms in smarts query
        patterns = mol.GetSubstructMatches(self.smarts, uniquify=self.uniquify)
        matches = []
        for patt in patterns:
            if self._pass_atom_variable_restriction(mol, patt) == False:
                continue
            # clean patt to remove positions to ignore
            idx_to_ignore = self.ignore_patt_idx
            filtered_patt = set(
                a
                for idx, a in enumerate(patt)
                if idx not in self.ignore_patt_idx
            )
            # only consider patterns without avoid_atoms
            if len(avoid_atoms.intersection(filtered_patt)) > 0:
                continue
            # sometimes you are looking for a specific patt
            if (
                len(limit_atoms) > 0
                and filtered_patt.issubset(limit_atoms) == False
            ):
                continue
            response = self._get_smarts_response(
                patt, filtered_patt, alpha_carbons
            )
            matches.append(response)
        if self.uniquify == False:
            matches = self._dereplicate_responses(matches)
        return matches

    def _pass_atom_variable_restriction(
        self, mol: Chem.Mol, patt: Tuple[int, ...]
    ) -> bool:
        if len(self.variable_to_atomic_num_map) > 0:
            for idx, atom_nums in self.variable_to_atomic_num_map.items():
                atom = mol.GetAtomWithIdx(patt[idx])
                if atom.GetAtomicNum() not in atom_nums:
                    return False
            else:
                return True
        else:
            return True

    def _get_smarts_response(
        self,
        original_patt: Tuple[int, ...],
        filtered_patt: Set[int],
        alpha_carbons: Set[int],
    ) -> EnhancedSmartsDict:
        # determine alpha carbon annotations
        alpha_restrictions = set(
            original_patt[a] for a in self.alpha_restriction_map
        )
        alpha_hit = len(alpha_restrictions & alpha_carbons)
        # determine variable restrictions
        variable_restrictions = {
            original_patt[a]: enums for a, enums in self.variable_map.items()
        }
        # determine unknown_restrictions
        unknown_restrictions = set(
            original_patt[a] for a in self.unknown_restriction_map
        )
        return {
            "description": self.description,
            "notation": self.notation,
            "hash_id": self.hash_id,
            "patt": filtered_patt,
            "patt_len": len(filtered_patt),
            "original_patt_len": len(original_patt),
            "enums": self.enums,
            "avoid_dereplication": self.avoid_dereplication,
            "can_dereplicate": self.can_dereplicate,
            "alpha_restrictions": alpha_restrictions,
            "alpha_hit": alpha_hit,
            "variable_restrictions": variable_restrictions,
            "unknown_restrictions": unknown_restrictions,
            "priortization": self.priortization,
        }

    def _dereplicate_responses(self, responses: List[EnhancedSmartsResponse]):
        overlap = nx.Graph()
        response_indexes = list(range(len(responses)))
        overlap.add_nodes_from(response_indexes)
        for i1, i2 in it.combinations(response_indexes, 2):
            r1 = responses[i1]
            r2 = responses[i2]
            if r1["patt"] == r2["patt"]:
                if r1["enums"] == r2["enums"]:
                    if (
                        r1["variable_restrictions"]
                        == r2["variable_restrictions"]
                    ):
                        if (
                            r1["alpha_restrictions"]
                            == r2["alpha_restrictions"]
                        ):
                            if (
                                r1["unknown_restrictions"]
                                == r2["unknown_restrictions"]
                            ):
                                overlap.add_edge(i1, i2)
        return [
            responses[list(g)[0]] for g in nx.connected_components(overlap)
        ]

    @classmethod
    def _load_from_dict(cls, data: EnhancedSmartsDict) -> "EnhancedSmarts":
        # load variables with default values if not present
        smarts = data["smarts"]
        notation = data["notation"]
        description = data["description"]
        hash_id = data["hash_id"]
        smiles = data.get("smiles")
        ignore_patt_idx = get_value(data, "ignore_patt_idx", [])
        enums = get_value(data, "enums", [], cast=set)
        avoid_dereplication = get_value(data, "avoid_dereplication", False)
        can_dereplicate = get_value(data, "can_dereplicate", True)
        variable_map = get_value(
            data, "variable_map", {}, cast=reformat_int2set
        )
        variable_to_atomic_num_map = get_value(
            data, "variable_to_atomic_num_map", {}, cast=reformat_int2set
        )
        unknown_restriction_map = get_value(
            data, "unknown_restriction_map", set(), cast=set
        )
        alpha_restriction_map = get_value(
            data, "alpha_restriction_map", set(), cast=set
        )
        priortization = get_value(data, "priortization", 0)
        uniquify = get_value(data, "uniquify", True)
        return cls(
            smarts,
            notation,
            description,
            smiles,
            ignore_patt_idx,
            enums,
            avoid_dereplication,
            can_dereplicate,
            variable_map,
            variable_to_atomic_num_map,
            unknown_restriction_map,
            alpha_restriction_map,
            priortization,
            uniquify,
            hash_id,
        )
