import os
from typing import List

import networkx as nx
import pandas as pd

from Bloom.BloomDOS.Graph.CliqueLinker import CliqueLinker


# common functions
def get_multi_max(l: List[dict], k: str):
    max_value = max(l, key=lambda x: x[k])[k]
    return [x for x in l if x[k] == max_value]


# package location
curdir = os.path.abspath(os.path.dirname(__file__))

# library destinations
# data loaded in BearLite.Annotation.SmartsLibrary
rendered_lib_dir = f"{curdir}/RenderedLibrary"
library_filter_fp = f"{curdir}/Parameters/library_filters.csv"
cactus_db_dir = f"{curdir}/CactusDB"

# load enum pairings to avoid dereplication when annotating molecules
avoid_dereplication_fp = f"{curdir}/Parameters/avoid_dereplication.csv"
avoid_dereplication = nx.Graph()
for rec in pd.read_csv(avoid_dereplication_fp).to_dict("records"):
    avoid_dereplication.add_edge(rec["enum_1"], rec["enum_2"])

# load enum pairings where if they are adjacent,
# avoid dereplication of enum by other unit
adj_avoid_dereplication_fp = (
    f"{curdir}/Parameters/adjacent_avoid_dereplication.csv"
)
adj_avoid_dereplication = nx.Graph()
for rec in pd.read_csv(adj_avoid_dereplication_fp).to_dict("records"):
    enums_to_avoid_derep = set()
    if rec["avoid_enum_1"] == True:
        enums_to_avoid_derep.add(rec["enum_1"])
    if rec["avoid_enum_2"] == True:
        enums_to_avoid_derep.add(rec["enum_2"])
    adj_avoid_dereplication.add_edge(
        rec["enum_1"], rec["enum_2"], enums_to_avoid_derep=enums_to_avoid_derep
    )

# load patterns to connect groups during clique calculation
clique_linker_fp = f"{curdir}/Parameters/clique_linkers.csv"
clique_linkers = []
for row in pd.read_csv(clique_linker_fp).to_dict("records"):
    clique_linkers.append(CliqueLinker._load_from_row(row))

# load special variable atoms
# this is used in creating smarts patterns in systems such as polyketides
sva_fp = f"{curdir}/Chemistry/Tables/special_variable_atoms.csv"
other_variable_atomic_nums = {
    r["atomic_num"]: r for r in pd.read_csv(sva_fp).to_dict("records")
}
