from typing import Dict, List, Optional, Set, Tuple


class NodeProperties:

    @property
    def nodes(self) -> Set[int]:
        return set(self.G.nodes)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def single_node(self) -> Optional[int]:
        return list(self.nodes)[0] if self.node_count == 1 else False

    @property
    def node_data(self) -> Tuple[int, dict]:
        return self.G.nodes(data=True)

    @property
    def nodes_in_mol(self) -> Set[int]:
        return {a.GetAtomMapNum() for a in self.mol.GetAtoms()} - {0}

    @property
    def node_index(self) -> Tuple[int]:
        return tuple(sorted(self.nodes))

    def get_nodes(self, enums: List[str]):
        return set().union(*[self.enum_to_nodes.get(e, set()) for e in enums])

    @property
    def pks_nodes(self) -> Set[int]:
        return self.get_nodes(["polyketide_monomer", "pk"])

    @property
    def nrps_nodes(self) -> Set[int]:
        return self.get_nodes(["aminoacid", "aa"])

    @property
    def sugar_nodes(self) -> Set[int]:
        return self.get_nodes(["sugar"])

    @property
    def tailoring_nodes(self) -> Set[int]:
        return self.get_nodes(["tailoring"])

    @property
    def primary_nodes(self) -> Set[int]:
        return self.get_nodes(["primary"])

    @property
    def secondary_nodes(self) -> Set[int]:
        return self.get_nodes(["secondary"])

    @property
    def starter_nodes(self) -> Set[int]:
        return self.get_nodes(["starter"])

    @property
    def terminal_nodes(self) -> Set[int]:
        return self.get_nodes(["TE"])

    @property
    def unknown_nodes(self) -> Set[int]:
        return self.get_nodes(["unknown"])

    @property
    def variable_nodes(self) -> Set[int]:
        return self.get_nodes(["variable"])

    @property
    def module_nodes(self) -> Set[int]:
        return (
            self.pks_nodes
            | self.nrps_nodes
            | self.starter_nodes
            | self.terminal_nodes
        )

    @property
    def nonmodular_nodes(self) -> Set[int]:
        return self.nodes - self.module_nodes

    @property
    def ambiguous_nodes(self) -> Set[int]:
        return self.unknown_nodes | self.variable_nodes

    @property
    def unambiguous_nodes(self) -> Set[int]:
        return self.nodes - self.ambiguous_nodes

    @property
    def annotated_nodes(self) -> Set[int]:
        return self.nodes - self.unknown_nodes

    @property
    def node_to_atoms(self) -> Dict[int, Set[int]]:
        return {n: self[n]["atoms"] for n in self.nodes}
