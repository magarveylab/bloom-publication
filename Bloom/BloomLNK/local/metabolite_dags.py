import json
from typing import List

import pandas as pd

from Bloom.BloomLNK.utils import curdir

########################################################################
# Load libraries
########################################################################


def load_library():
    library_dir = f"{curdir}/local/tables"
    library = {}
    # smarts to units
    df = pd.read_csv(f"{library_dir}/smarts_to_unit.csv")
    library["smarts_to_unit"] = dict(zip(df.smarts_hash_id, df.unit_id))
    # unit to module tag
    df = pd.read_csv(f"{library_dir}/unit_to_module_tag.csv")
    library["unit_to_module_tag"] = dict(zip(df.unit_id, df.module_tag))
    # unit to pks reaction tag
    df = pd.read_csv(f"{library_dir}/unit_to_pks_reaction_tag.csv")
    library["unit_to_pks_reaction_tag"] = dict(
        zip(df.unit_id, df.pks_reaction_tag)
    )
    # unit to substrate
    df = pd.read_csv(f"{library_dir}/unit_to_substrate.csv")
    library["unit_to_substrate"] = dict(zip(df.unit_id, df.substrate))
    # unit to substrate family
    df = pd.read_csv(f"{library_dir}/unit_to_substrate_family.csv")
    library["unit_to_substrate_family"] = dict(
        zip(df.unit_id, df.substrate_family)
    )
    # sugar tag to ec4
    sugar_tag_to_ec4 = {}
    for r in pd.read_csv(f"{library_dir}/sugar_tag_to_ec4.csv").to_dict(
        "records"
    ):
        sugar_tag = r["sugar_tag"]
        if sugar_tag not in sugar_tag_to_ec4:
            sugar_tag_to_ec4[sugar_tag] = set()
        sugar_tag_to_ec4[sugar_tag].add(r["ec4"])
    # unit to sugar tag
    unit_to_sugar_tag = {}
    for r in pd.read_csv(f"{library_dir}/unit_to_sugar_tag.csv").to_dict(
        "records"
    ):
        unit_id = r["unit_id"]
        rule_id = r["rule_id"]
        sugar_tag = r["sugar_tag"]
        if sugar_tag not in sugar_tag_to_ec4:
            continue
        if unit_id not in unit_to_sugar_tag:
            unit_to_sugar_tag[unit_id] = []
        for ec4 in sugar_tag_to_ec4[sugar_tag]:
            unit_to_sugar_tag[unit_id].append(
                {
                    "rule_id": rule_id,
                    "sugar_reaction_tag": sugar_tag,
                    "ec4": ec4,
                }
            )
    library["unit_to_sugar_tag"] = unit_to_sugar_tag
    # unit to tailoring tag
    unit_to_tailoring_tag = {}
    for r in pd.read_csv(f"{library_dir}/unit_to_tailoring_tag.csv").to_dict(
        "records"
    ):
        unit_to_tailoring_tag[r["unit_id"]] = {
            "rule_id": r["rule_id"],
            "tailoring_reaction_tag": r["tailoring_tag"],
            "ec3": r["ec3"],
        }
    library["unit_to_tailoring_tag"] = unit_to_tailoring_tag
    # reaction to ec4
    reaction_to_ec4 = {}
    for r in pd.read_csv(f"{library_dir}/reaction_to_ec4.csv").to_dict(
        "records"
    ):
        reaction_id = r["reaction_id"]
        ec4 = r["ec4"]
        if reaction_id not in reaction_to_ec4:
            reaction_to_ec4[reaction_id] = set()
        reaction_to_ec4[reaction_id].add(ec4)
    # reaction to protein family
    reaction_to_pf = {}
    for r in pd.read_csv(
        f"{library_dir}/reaction_to_protein_family_tag.csv"
    ).to_dict("records"):
        reaction_id = r["reaction_id"]
        protein_family_tag = r["protein_family_tag"]
        if reaction_id not in reaction_to_pf:
            reaction_to_pf[reaction_id] = set()
        reaction_to_pf[reaction_id].add(protein_family_tag)
    # unit to reaction
    unit_to_reaction = {}
    for r in pd.read_csv(f"{library_dir}/unit_to_reaction.csv").to_dict(
        "records"
    ):
        unit_id = r["unit_id"]
        rule_id = r["rule_id"]
        reaction_id = r["reaction_id"]
        if unit_id not in unit_to_reaction:
            unit_to_reaction[unit_id] = []
        for ec4 in reaction_to_ec4.get(reaction_id, []):
            unit_to_reaction[unit_id].append(
                {
                    "rule_id": rule_id,
                    "reaction_id": reaction_id,
                    "ec4": ec4,
                    "dag": "reaction_ec4",
                }
            )
        for pf in reaction_to_pf.get(reaction_id, []):
            unit_to_reaction[unit_id].append(
                {
                    "rule_id": rule_id,
                    "reaction_id": reaction_id,
                    "protein_family_tag": pf,
                    "dag": "reaction_gene",
                }
            )
    library["unit_to_reaction"] = unit_to_reaction
    return library


library = load_library()
smarts_to_unit = library["smarts_to_unit"]
unit_to_module_tag = library["unit_to_module_tag"]
unit_to_pks_reaction_tag = library["unit_to_pks_reaction_tag"]
unit_to_substrate = library["unit_to_substrate"]
unit_to_substrate_family = library["unit_to_substrate_family"]
unit_to_sugar_tag = library["unit_to_sugar_tag"]
unit_to_tailoring_tag = library["unit_to_tailoring_tag"]
unit_to_reaction = library["unit_to_reaction"]

########################################################################
# Helper Functions
########################################################################


def get_unit_to_regions(breakdown_data: dict):
    unit_to_regions = {}
    for breakdown_id, data in breakdown_data.items():
        for n in data["nodes"]:
            local_node_id = n["node_id"]
            node_id = f"{breakdown_id}_{local_node_id}"
            smarts_hash_id = n["meta"].get("hash_id")
            if smarts_hash_id in smarts_to_unit:
                unit_id = smarts_to_unit.get(smarts_hash_id)
                if unit_id not in unit_to_regions:
                    unit_to_regions[unit_id] = set()
                unit_to_regions[unit_id].add(node_id)
    return unit_to_regions


########################################################################
# DAGs to construct metabolite graph
########################################################################


def get_module_tag_dag(unit_to_regions):
    return [
        {
            "unit_region_id": r,
            "module_tag": unit_to_module_tag[unit_id],
            "dag": "module_tag",
        }
        for unit_id, regions in unit_to_regions.items()
        if unit_id in unit_to_module_tag
        for r in regions
    ]


def get_pks_reaction_tag_dag(unit_to_regions):
    return [
        {
            "unit_region_id": r,
            "pks_reaction_tag": unit_to_pks_reaction_tag[unit_id],
            "dag": "pks_reaction_tag",
        }
        for unit_id, regions in unit_to_regions.items()
        if unit_id in unit_to_pks_reaction_tag
        for r in regions
    ]


def get_substrate_dag(unit_to_regions):
    return [
        {
            "unit_region_id": r,
            "substrate": unit_to_substrate[unit_id],
            "dag": "substrate",
        }
        for unit_id, regions in unit_to_regions.items()
        if unit_id in unit_to_substrate
        for r in regions
    ]


def get_substrate_family_tag_dag(unit_to_regions):
    return [
        {
            "unit_region_id": r,
            "substrate_family_tag": unit_to_substrate_family[unit_id],
            "dag": "substrate_family_tag",
        }
        for unit_id, regions in unit_to_regions.items()
        if unit_id in unit_to_substrate_family
        for r in regions
    ]


def get_sugar_rule_ec4_dag(unit_to_regions):
    return [
        {
            "unit_region_id": r,
            "rule_id": rule["rule_id"],
            "sugar_reaction_tag": rule["sugar_reaction_tag"],
            "ec4": rule["ec4"],
            "dag": "sugar_rule_ec4",
        }
        for unit_id, regions in unit_to_regions.items()
        if unit_id in unit_to_sugar_tag
        for rule in unit_to_sugar_tag[unit_id]
        for r in regions
    ]


def get_tailoring_rule_ec3_dag(unit_to_regions):
    return [
        {
            "unit_region_id": r,
            "rule_id": unit_to_tailoring_tag[unit_id]["rule_id"],
            "tailoring_reaction_tag": unit_to_tailoring_tag[unit_id][
                "tailoring_reaction_tag"
            ],
            "ec3": unit_to_tailoring_tag[unit_id]["ec3"],
            "dag": "tailoring_rule_ec3",
        }
        for unit_id, regions in unit_to_regions.items()
        if unit_id in unit_to_tailoring_tag
        for r in regions
    ]


def get_reaction_dag(unit_to_regions):
    dags = []
    for unit_id, regions in unit_to_regions.items():
        if unit_id in unit_to_reaction:
            for r in regions:
                for d in unit_to_reaction[unit_id]:
                    d = d.copy()
                    d["unit_region_id"] = r
                    dags.append(d)
    return dags


def get_metabolite_dags(breakdown_fps: List[str]):
    # load bear breakdowns
    breakdown_data = {
        idx: json.load(open(fp)) for idx, fp in enumerate(breakdown_fps, 1)
    }
    unit_to_regions = get_unit_to_regions(breakdown_data=breakdown_data)
    dags = (
        get_module_tag_dag(unit_to_regions=unit_to_regions)
        + get_pks_reaction_tag_dag(unit_to_regions=unit_to_regions)
        + get_substrate_dag(unit_to_regions=unit_to_regions)
        + get_substrate_family_tag_dag(unit_to_regions=unit_to_regions)
        + get_sugar_rule_ec4_dag(unit_to_regions=unit_to_regions)
        + get_tailoring_rule_ec3_dag(unit_to_regions=unit_to_regions)
        + get_reaction_dag(unit_to_regions=unit_to_regions)
    )
    return dags
