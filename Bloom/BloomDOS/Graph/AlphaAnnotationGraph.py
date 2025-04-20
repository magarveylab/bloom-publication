from typing import Set, Tuple

from rdkit import Chem

from Bloom.BloomDOS.Graph.AnnotationGraph import AnnotationGraph


class AlphaAnnotationGraph(AnnotationGraph):

    def __init__(
        self,
        mol: Chem.Mol,
        calculate_overlap: bool = True,
        calculate_adjacency: bool = False,
    ):
        super().__init__(
            mol,
            calculate_overlap=calculate_overlap,
            calculate_adjacency=calculate_adjacency,
        )

    def priortization(
        self, annotations: Set[int]
    ) -> Tuple[int, int, int, int]:
        # 1. most alpha carbons adjacent to oxygen
        # 2. most alpha carbons in double bond systems
        # 3. most alpha carbons in saturated sysyems
        # 4. most atom coverage
        # 5. least patterns
        return (
            sum([self[a]["C_adj_O"] for a in annotations]),
            sum([self[a]["C_start"] for a in annotations]),
            sum([self[a]["C_alk"] for a in annotations]),
            sum([self[a]["C_sat"] for a in annotations]),
            sum([self[a]["patt_len"] for a in annotations]),
            -len(annotations),
        )
