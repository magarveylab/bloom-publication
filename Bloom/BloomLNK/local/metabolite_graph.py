import json
from typing import List

import pandas as pd

from Bloom.BloomLNK.local.common import create_graph_from_dags
from Bloom.BloomLNK.utils import curdir

########################################################################
# Graph Schema
########################################################################

col_name_to_node_type = {
    "metabolite_id": "Metabolite",
    "breakdown_id": "MetaboliteBreakdown",
    "unit_region_id": "MetaboliteMolecularRegion",
    "adj_unit_region_id": "MetaboliteMolecularRegion",
}
col_name_to_node_properties = {
    "unit_region_id": ["unit"],
}
col_name_to_edge_type = {
    ("metabolite_id", "breakdown_id"): "metabolite_to_breakdown",
    ("breakdown_id", "unit_region_id"): "metab_breakdown_to_mol_region",
    ("unit_region_id", "adj_unit_region_id"): "mol_region_adj",
}
col_name_to_edge_properties = {}

########################################################################
# Load libraries
########################################################################


def load_library():
    library_dir = f"{curdir}/local/tables"
    library = {
        "smarts_to_unit": {},
    }
    # smarts to units
    df = pd.read_csv(f"{library_dir}/smarts_to_unit.csv")
    library["smarts_to_unit"] = dict(zip(df.smarts_hash_id, df.unit))
    return library


library = load_library()
smarts_to_unit = library["smarts_to_unit"]

########################################################################
# DAGs to construct metabolite graph
########################################################################


def get_unit_dag(metabolite_id: int, breakdown_data: dict):
    out = []
    for breakdown_id, data in breakdown_data.items():
        for n in data["nodes"]:
            local_node_id = n["node_id"]
            node_id = f"{breakdown_id}_{local_node_id}"
            smarts_hash_id = n["meta"].get("hash_id")
            unit = smarts_to_unit.get(smarts_hash_id)
            out.append(
                {
                    "metabolite_id": metabolite_id,
                    "breakdown_id": breakdown_id,
                    "unit_region_id": node_id,
                    "unit": unit,
                }
            )
    return out


def get_unit_adj_dag(metabolite_id: int, breakdown_data: dict):
    out = []
    for breakdown_id, data in breakdown_data.items():
        for e in data["edges"]:
            n1, n2 = e["n1"], e["n2"]
            n1 = f"{breakdown_id}_{n1}"
            n2 = f"{breakdown_id}_{n2}"
            out.append(
                {
                    "metabolite_id": metabolite_id,
                    "breakdown_id": breakdown_id,
                    "unit_region_id": n1,
                    "adj_unit_region_id": n2,
                }
            )
    return out


########################################################################
# Main Functions
########################################################################


def get_metabolite_graphs(metabolite_id: int, breakdown_fps: List[str]):
    # load bear breakdowns
    breakdown_data = {
        idx: json.load(open(fp)) for idx, fp in enumerate(breakdown_fps, 1)
    }
    # prepare dags
    unit_dags = get_unit_dag(
        metabolite_id=metabolite_id,
        breakdown_data=breakdown_data,
    )
    unit_adj_dags = get_unit_adj_dag(
        metabolite_id=metabolite_id, breakdown_data=breakdown_data
    )
    # prepare graph
    G = create_graph_from_dags(
        dags=unit_dags + unit_adj_dags,
        col_name_to_node_type=col_name_to_node_type,
        col_name_to_node_properties=col_name_to_node_properties,
        col_name_to_edge_type=col_name_to_edge_type,
        col_name_to_edge_properties=col_name_to_edge_properties,
    )
    return G
