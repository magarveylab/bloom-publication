import itertools as it
from collections import Counter
from typing import List, Optional, Set, Tuple

import networkx as nx
from rdkit import Chem
from tqdm import tqdm

from Bloom.BloomDOS.DataStructs import EnhancedSmartsResponse
from Bloom.BloomDOS.GeneralUtils import (
    adj_avoid_dereplication,
    avoid_dereplication,
    clique_linkers,
    get_multi_max,
)


class AnnotationGraph:

    def __init__(
        self,
        mol: Chem.Mol,
        calculate_overlap: bool = True,
        calculate_adjacency: bool = True,
        silent_output: bool = True,
    ):
        self.calculate_overlap = calculate_overlap
        self.calculate_adjacency = calculate_adjacency
        self.silent_output = silent_output
        # every annotation is given a unique id
        self.unique_node_id = 1
        # store meta information on nodes
        self.G = nx.Graph()
        # graphs store relationships between overlapping nodes
        # 1. exact -> annotations that map to exact same region
        # 2. subset -> n1 is subset of n2
        # 3. avoid_derep -> can edge be used for dereplication
        # edges are used to remove annotations
        # (if True, do no use edge information for removal)
        self.G_edge = nx.DiGraph()
        # graph stores adjacent non-overlapping annotations
        # necessary to calculate modular stretches or polyether
        self.G_adj = nx.Graph()
        # graph stores adjacent non-overlapping annotations
        # but only if there is a variable restriction
        # node (with variable restrictions) -> node (meets variable restrictions)
        self.G_var = nx.DiGraph()
        # graph stores atomic data - track atoms to nodes
        self.G_atom = nx.Graph()
        for a in mol.GetAtoms():
            self.G_atom.add_node(a.GetIdx(), nodes=set())
        # load atomic bonds for adjacency calculation
        for b in mol.GetBonds():
            self.G_atom.add_edge(
                b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_id=b.GetIdx()
            )

    def __getitem__(self, key):
        # class acts like dictionary
        return self.G.nodes[key]["annotation"]

    @property
    def nodes(self):
        return self.G.nodes

    @property
    def edges(self):
        return self.G.edges

    @property
    def atom_count(self):
        return len(self.G_atom.nodes)

    ####################################################################
    # Debug properties and methods
    ####################################################################

    @property
    def annotation_frequency(self):
        return Counter([self[n]["notation"] for n in self.nodes])

    def get_patts(self, notation: str):
        return [
            self[n]["patt"]
            for n in self.nodes
            if self[n]["notation"] == notation
        ]

    ####################################################################
    # Methods to add data to class
    ####################################################################

    def add_node(self, annotation: EnhancedSmartsResponse) -> int:
        existing_nodes = set(self.nodes)
        self.unique_node_id += 1
        # find intrabonds (bonds within a unit)
        unit_G = self.G_atom.subgraph(annotation["patt"])
        annotation["intrabonds"] = set(
            b["bond_id"] for a1, a2, b in unit_G.edges(data=True)
        )
        # add node
        self.G.add_node(self.unique_node_id, annotation=annotation)
        self.G_adj.add_node(self.unique_node_id, enums=annotation["enums"])
        if len(annotation["variable_restrictions"]) > 0:
            self.G_var.add_node(self.unique_node_id)
        # track enums with regards to atoms
        for a in annotation["patt"]:
            self.G_atom.nodes[a]["nodes"].add(self.unique_node_id)
        # add edge information to previous nodes in graph
        for n in existing_nodes:
            self.add_edge(n, self.unique_node_id)
        return self.unique_node_id

    def add_edge(self, n1: int, n2: int):
        # annotated atoms
        n1_patt = self[n1]["patt"]
        n2_patt = self[n2]["patt"]
        # associated enums
        n1_enums = self[n1]["enums"]
        n2_enums = self[n2]["enums"]
        # cache overlapping annotations
        if self.calculate_overlap == True and len(n1_patt & n2_patt) > 0:
            # cache overlap
            self.G.add_edge(n1, n2)
            # exact indicates an annotation mapping exact same region
            if n1_patt == n2_patt:
                exact = True
            else:
                exact = False
            # is n1 subset of n2?
            n1_to_n2_subset = n1_patt.issubset(n2_patt)
            # is n2 subset of n1?
            n2_to_n1_subset = n2_patt.issubset(n1_patt)
            # n1_to_n2_avoid_derep -> can n2 not be used to remove n1?
            # n2_to_n1_avoid_derep -> can n1 not be used to remove n2?
            # define edges in parameters where dereplication should not be considered
            for pair in it.product(n1_enums, n2_enums):
                if avoid_dereplication.has_edge(*pair):
                    n1_to_n2_avoid_derep = True
                    n2_to_n1_avoid_derep = True
                    break
            else:
                n1_to_n2_avoid_derep = False
                n2_to_n1_avoid_derep = False
            # define annotations where dereplication should not be considered
            # n1 cannot be dereplicated or n2 cannot dereplicate n1
            if (
                self[n1]["avoid_dereplication"] == True
                or self[n2]["can_dereplicate"] == False
            ):
                n1_to_n2_avoid_derep = True
            # n2 cannot be dereplicated or n1 cannot dereplicate n2
            if (
                self[n2]["avoid_dereplication"] == True
                or self[n1]["can_dereplicate"] == False
            ):
                n2_to_n1_avoid_derep = True
            # cache edge meta data
            self.G_edge.add_edge(
                n1,
                n2,
                exact=exact,
                subset=n1_to_n2_subset,
                avoid_derep=n1_to_n2_avoid_derep,
            )
            self.G_edge.add_edge(
                n2,
                n1,
                exact=exact,
                subset=n2_to_n1_subset,
                avoid_derep=n2_to_n1_avoid_derep,
            )
        # cache adjacent annotations
        if self.calculate_adjacency is True and self.is_adjacent(n1, n2):
            # check if adjacent link introduces new units to avoid dereplicate
            for e1, e2 in it.product(n1_enums, n2_enums):
                if adj_avoid_dereplication.has_edge(e1, e2):
                    enums_to_avoid_derep = adj_avoid_dereplication[e1][e2][
                        "enums_to_avoid_derep"
                    ]
                    avoid_derep_nodes = set()
                    if e1 in enums_to_avoid_derep:
                        avoid_derep_nodes.add(n1)
                    if e2 in enums_to_avoid_derep:
                        avoid_derep_nodes.add(n2)
                    break
            else:
                avoid_derep_nodes = set()
            # cache association
            self.G_adj.add_edge(n1, n2, avoid_derep_nodes=avoid_derep_nodes)
            # check if edge agrees with variable restriction
            # some adacent nodes have requirements based on the enums
            # such as a pk unit must be beside another aa unit or pk unit
            # n1 has variable restrictions and n2 meets variable restrictions
            if len(self[n1]["variable_restrictions"]) > 0:
                restricted_atoms = self.get_enum_restricted_atoms(n1, n2)
                if len(restricted_atoms) > 0:
                    self.G_var.add_edge(
                        n1, n2, restricted_atoms=restricted_atoms
                    )
            # n2 has variable restrictions and n1 meets variable restrictions
            # G_var is DiGraph (so captures the directionality of this rel)
            if len(self[n2]["variable_restrictions"]) > 0:
                restricted_atoms = self.get_enum_restricted_atoms(n2, n1)
                if len(restricted_atoms) > 0:
                    self.G_var.add_edge(
                        n2, n1, restricted_atoms=restricted_atoms
                    )

    def remove_nodes_from(self, nodes: Set[int]):
        # some annotations are dependent on each other
        # if dependency is not met, remove conflicting nodes
        conflicting_nodes = self.get_variable_conflicts(self.nodes)
        nodes_to_remove = nodes | conflicting_nodes
        while len(nodes_to_remove) > 0:
            # remove enums from atom graph
            for a in self.G_atom.nodes:
                a_nodes = self.G_atom.nodes[a]["nodes"]
                self.G_atom.nodes[a]["enums"] = a_nodes - nodes_to_remove
            # remove nodes from graphs
            self.G.remove_nodes_from(nodes_to_remove)
            self.G_edge.remove_nodes_from(nodes_to_remove)
            self.G_adj.remove_nodes_from(nodes_to_remove)
            self.G_var.remove_nodes_from(nodes_to_remove)
            # replace nodes to remove with new conflicting nodes
            nodes_to_remove = self.get_variable_conflicts(self.nodes)

    ####################################################################
    # Methods to query data from graph
    ####################################################################

    def get_nodes(self, enums: Optional[Set[str]] = None) -> Set[int]:
        nodes = set()
        for n in self.nodes:
            if enums != None and len(self[n]["enums"] & enums) == 0:
                continue
            nodes.add(n)
        return nodes

    def get_edges(
        self,
        nodes: Optional[Set[int]] = None,
        exact: Optional[bool] = None,
        subset: Optional[bool] = None,
        avoid_derep: Optional[bool] = None,
        overlap: Optional[bool] = None,
        adjacent: Optional[bool] = None,
    ) -> Set[Tuple[int, int]]:
        edges = set()
        if overlap is True:
            # some nodes you avoid dereplication based on adjacent context
            ignore_nodes = set()
            if (
                avoid_derep == False
            ):  # edges where nodes can dereplicate each other
                for n1, n2, e in self.G_adj.edges(data=True):
                    ignore_nodes.update(e["avoid_derep_nodes"])
            # search overlap edges
            for n1, n2, e in self.G_edge.edges(data=True):
                if len({n1, n2}.intersection(ignore_nodes)) > 0:
                    continue
                if nodes != None and {n1, n2}.issubset(nodes) is False:
                    continue
                if exact != None and exact != e["exact"]:
                    continue
                if subset != None and subset != e["subset"]:
                    continue
                if avoid_derep != None and avoid_derep != e["avoid_derep"]:
                    continue
                edges.add((n1, n2))
        # search adjacent edges
        if adjacent is True:
            for n1, n2 in self.G_adj:
                if nodes != None and {n1, n2}.issubset(nodes) is False:
                    continue
                edges.add((n1, n2))
        return edges

    def get_enums_from_nodes(self, nodes: Set[int]) -> Set[str]:
        return set(e for n in nodes for e in self[n]["enums"])

    def get_atoms_from_nodes(self, nodes: Set[int]) -> Set[int]:
        return set(a for n in nodes for a in self[n]["patt"])

    def get_enum_count(self, nodes: Set[int], enum: str) -> int:
        return len([n for n in nodes if enum in self[n]["enums"]])

    def is_adjacent(self, n1: int, n2: int) -> bool:
        n1_patt = self[n1]["patt"]
        n2_patt = self[n2]["patt"]
        if len(n1_patt & n2_patt) == 0:
            for bond in it.product(n1_patt, n2_patt):
                if self.G_atom.has_edge(*bond):
                    return True
            else:
                return False
        else:
            return False

    def get_enum_restricted_atoms(self, n1: int, n2: int) -> Set[int]:
        n1_var = self[n1]["variable_restrictions"]
        n2_patt = self[n2]["patt"]
        n2_enums = self[n2]["enums"]
        # check if atoms in n1 variable restrictions are mapped by n2
        overlap = set(n1_var) & n2_patt
        restricted_atoms = set()
        if len(overlap) > 0:
            for a in overlap:
                # consider variable enum
                a_enums = n1_var[a] | {"variable"}
                if len(a_enums & n2_enums) > 0:
                    restricted_atoms.add(a)
        return restricted_atoms

    def get_variable_conflicts(self, nodes: Set[int]) -> Set[int]:
        # certain atoms of a unit must be annotated with another unit
        # of a defined enum - if not, there is a conflict
        # these atoms are usually variable atoms
        # you want to limit variable conflicts to the node search space
        # function is used in determining if a subset of annotations is valid
        conflicting_nodes = set()
        subspace = self.G_var.subgraph(nodes)
        atoms = self.get_atoms_from_nodes(nodes)
        for n in nodes:
            n_var = set(self[n]["variable_restrictions"])
            if len(n_var) > 0:
                # look for any restricted atoms in query atom space
                enum_restricted_atoms = n_var & atoms
                # if there are atoms, look if any node in search space
                # does not fulfill requirements -> hence there is a conflict
                if len(enum_restricted_atoms) > 0:
                    for a in enum_restricted_atoms:
                        fulfilled = False
                        for _, _, e in subspace.edges(n, data=True):
                            # find first instance of fulfilled requirement
                            if a in e["restricted_atoms"]:
                                fulfilled = True
                                break
                        # variable conflict present for atom
                        # this is conflicting node
                        if fulfilled == False:
                            conflicting_nodes.add(n)
                            break
        return conflicting_nodes

    ####################################################################
    # Adopted graph methods from networkx
    ####################################################################

    def edge_subgraph(
        self, edges=Set[int], adjacent: bool = False
    ) -> nx.Graph:
        if adjacent is True:
            return self.G_adj.edge_subgraph(edges).copy()
        else:
            return self.G_edge.edge_subgraph(edges).copy()

    def neighbors(self, node_id: int) -> Set[int]:
        return set(self.G.neighbors(node_id))

    def predecessors(self, node_id: int) -> Set[int]:
        return set(self.G.predecessors(node_id))

    ####################################################################
    # Specialized graph methods
    ####################################################################

    def get_nonoverlap_groups_from(self, nodes: List[int]) -> List[Set[int]]:
        # G_s -> corresponds to overlap graph (with selected nodes)
        # G_calc -> corresponds to node groups per clique calculation
        # G_adj_s -> corresponds to adjacent graph (with selected nodes)
        G_s = self.G.subgraph(nodes)
        G_calc = G_s.copy()
        G_adj_s = self.G_adj.subgraph(nodes).to_undirected(as_view=True)
        # add edges to connect nodes with linker restrictions
        # only certain linker restrictions are considered according to parameters
        # this is to handle dependency between different clique calculations
        for linker in clique_linkers:
            G_calc.add_edges_from(linker.search(G_adj_s))
        # split nodes into communities - each community will undergo calculation
        # smaller graph - faster compute time
        overlap_groups = list(nx.connected_components(G_calc))
        pool = []
        for nodes in tqdm(
            overlap_groups,
            desc="calculate nonoverlap groups",
            disable=self.silent_output,
        ):
            if len(nodes) == 1:
                pool.append([nodes])
            else:
                atom_space = self.get_atoms_from_nodes(nodes)
                subgraph = G_s.subgraph(nodes)
                nonoverlap_subgraph = nx.Graph()
                nonoverlap_subgraph.add_edges_from(nx.non_edges(subgraph))
                # store groups non-overlap annotations
                cache = []
                one_unit_solution = False
                find_maximal_clique = False
                # check if one unit can explain full coverage
                for n in nodes:
                    cache.append(
                        {"group": [n], "score": self.priortization([n])}
                    )
                    if self[n]["patt"] == atom_space:
                        one_unit_solution = True
                # because we are working with a graph capturing non-overlap edges
                # clique corresponds to groups of non-overlapping annotations
                if one_unit_solution == False:
                    # check if the top 3 units can explain at least 90% of space
                    top_n = sorted(
                        nodes, key=lambda x: len(self[x]["patt"]), reverse=True
                    )[:3]
                    top_n_atom_space = self.get_atoms_from_nodes(top_n)
                    coverage = len(top_n_atom_space & atom_space) / len(
                        atom_space
                    )
                    if coverage < 0.9:
                        find_maximal_clique = True
                    # enumerate all cliques
                    # stop at early solution at early cardinality
                    if find_maximal_clique == False:
                        early_stop = 100
                        # enumerate all cliques starts from lowest cardinality to largest
                        # we want the solution that uses the least units but covers the most atoms
                        for c in nx.enumerate_all_cliques(nonoverlap_subgraph):
                            conflicting_nodes = self.get_variable_conflicts(c)
                            if len(conflicting_nodes) == 0:
                                cardinality = len(c)
                                # check if atom space can be fully explained
                                # if so capture cardinality
                                # we want to continue accepting solutions at that cardinality
                                # maybe there is a better one
                                if cardinality > early_stop:
                                    break
                                c_atom_space = self.get_atoms_from_nodes(c)
                                if c_atom_space == atom_space:
                                    early_stop = int(cardinality)
                                # cache
                                cache.append(
                                    {
                                        "group": c,
                                        "score": self.priortization(c),
                                    }
                                )
                    # enumerate_all_cliques is super slow for large graphs
                    # because its a large graph with very small units
                    # we are making the approximation that the maximal clique
                    # usually results in a larger atom annotation
                    # this is an ok approximation because we removed alot of
                    # subset annotations in the dereplication
                    else:
                        for c in nx.find_cliques(nonoverlap_subgraph):
                            conflicting_nodes = self.get_variable_conflicts(c)
                            if len(conflicting_nodes) == 0:
                                cache.append(
                                    {
                                        "group": c,
                                        "score": self.priortization(c),
                                    }
                                )
                # keep groups with best priortization scores
                keep = get_multi_max(cache, "score")
                pool.append([k["group"] for k in keep])
        # combine groups from pool to calculate overall non-overlap groups
        nonoverlap_groups = [
            {y for x in p for y in x} for p in it.product(*pool)
        ]
        cache = [
            {"group": g, "score": self.priortization(g)}
            for g in nonoverlap_groups
        ]
        keep = get_multi_max(cache, "score")
        return [k["group"] for k in keep]

    def get_continous_stretches_from(self, nodes: Set[int]) -> List[Set[int]]:
        # organize annotations into non-overlapping groups
        nonoverlap_groups = self.get_nonoverlap_groups_from(nodes)
        # profile each stretch
        stretches = {}
        overlap = nx.Graph()
        idx = 1
        for group in nonoverlap_groups:
            subgraph = self.G_adj.subgraph(group)
            for stretch_nodes in nx.connected_components(subgraph):
                atoms = self.get_atoms_from_nodes(stretch_nodes)
                stretches[idx] = {
                    "nodes": stretch_nodes,
                    "atoms": atoms,
                    "atom_count": len(atoms),
                }
                overlap.add_node(idx)
                idx += 1
        # find overlapping stretches
        for s1, s2 in it.combinations(stretches, 2):
            if len(stretches[s1]["atoms"] & stretches[s2]["atoms"]) > 0:
                overlap.add_edge(s1, s2)
        out = []
        for group in nx.connected_components(overlap):
            cache = [stretches[x] for x in group]
            best = get_multi_max(cache, "atom_count")
            out.extend([s["nodes"] for s in best])
        return out

    def get_unknown_units(self, annotations: Set[int]) -> List[Set[int]]:
        annotated_atoms = self.get_atoms_from_nodes(annotations)
        all_atoms = self.G_atom.nodes
        unannotated_atoms = all_atoms - annotated_atoms
        subgraph = self.G_atom.subgraph(unannotated_atoms)
        return list(nx.connected_components(subgraph))

    def priortization(
        self, annotations: Set[int]
    ) -> Tuple[int, int, int, int, bool, int]:
        # default priortization function for nonoverlap groups
        # 1. most annotated atoms
        # 2. least flexible units
        # 3. most sugar units
        # 4. alpha carbon count
        # 5. least pK R variable groups
        # 6. least pk_specialized units
        # 7. use of starter units
        # 8. use of terminal units
        # 9. most curated pk ring units
        # 10. most number of aa units
        # 11. least number of units
        return (
            len(self.get_atoms_from_nodes(annotations)),
            -self.get_enum_count(annotations, "flexible"),
            self.get_enum_count(annotations, "sugar"),
            sum([self[n]["alpha_hit"] for n in annotations]),
            -self.get_enum_count(annotations, "pk-R"),
            -self.get_enum_count(annotations, "pk_specialized"),
            -self.get_enum_count(annotations, "DH-shift-backward"),
            -self.get_enum_count(annotations, "DH-shift-forward"),
            True if self.get_enum_count(annotations, "starter") > 0 else False,
            True if self.get_enum_count(annotations, "TE") > 0 else False,
            self.get_enum_count(annotations, "curated_module"),
            self.get_enum_count(annotations, "aa"),
            -len(annotations),
        )
