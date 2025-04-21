import itertools as it
import time
from typing import List

import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smt
from scipy.stats import chi2_contingency
from tqdm import tqdm


def run_umap(
    matrix: np.array,
    n_components: int = 20,
    n_neighbors: int = 15,
    n_epochs: int = 500,
    min_dist=0.1,
):
    import cuml

    print("Running Dimension Reduction ...")
    start = time.time()
    reducer = cuml.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        n_epochs=n_epochs,
        min_dist=min_dist,
        random_state=12,
    )
    embedding = reducer.fit_transform(matrix)
    end = time.time()
    timing = round(end - start, 2)
    print(f"Took {timing} seconds")
    return embedding


def run_hdbscan(
    reduced_matrix: np.array,
    matrix_keys: List[dict],
    min_cluster_size: int = 5,
    metric: str = "euclidean",
):
    import cuml

    print("Running Soft Clustering ...")
    start = time.time()
    clusterer = cuml.cluster.hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=metric,
        prediction_data=True,
    )
    clusterer.fit(reduced_matrix)
    labels = clusterer.labels_
    end = time.time()
    timing = round(end - start, 2)
    print(f"Took {timing} seconds")
    output = []
    for idx, l in enumerate(labels):
        row = dict(matrix_keys[idx])
        row["family_id"] = int(l)
        output.append(row)
    return output


def compute_clustering(
    matrix: np.array,
    matrix_keys: List[dict],
    n_components: int = 20,
    n_neighbors: int = 15,
    n_epochs: int = 500,
    min_dist=0.1,
    min_cluster_size: int = 5,
    metric: str = "euclidean",
):

    # run umap
    reduced_matrix, _ = run_umap(
        matrix=matrix,
        n_components=n_components,
        n_neighbors=n_neighbors,
        n_epochs=n_epochs,
        min_dist=min_dist,
    )
    # run soft clustering with hdbscan
    return run_hdbscan(
        reduced_matrix=reduced_matrix,
        matrix_keys=matrix_keys,
        min_cluster_size=min_cluster_size,
        metric=metric,
    )


def get_unit_orf_associations(units_hdbscan_fp: str, orfs_hdbscan_fp: str):
    unit_lookup = {}
    orf_lookup = {}
    unit_group_to_mibig = {}
    orf_group_to_mibig = {}
    unit_to_group = {}
    orf_to_group = {}
    mibig_to_units = {}
    mibig_to_orfs = {}
    # reorganize unit data
    unit_data = pd.read_csv(units_hdbscan_fp).to_dict("records")
    for row in unit_data:
        family_id = row["family_id"]
        mibig_id = row["mibig_id"]
        unit_id = (mibig_id, row["unit_id"])
        unit_lookup[unit_id] = row["metabolite_id"]
        # ignore units that do not form any groups
        if family_id == -1:
            continue
        if family_id not in unit_group_to_mibig:
            unit_group_to_mibig[family_id] = set()
        unit_group_to_mibig[family_id].add(mibig_id)
        if mibig_id not in mibig_to_units:
            mibig_to_units[mibig_id] = set()
        mibig_to_units[mibig_id].add(unit_id)
        unit_to_group[unit_id] = family_id
    # reorganize orf data
    orf_data = pd.read_csv(orfs_hdbscan_fp).to_dict("records")
    for row in orf_data:
        family_id = row["family_id"]
        mibig_id = row["mibig_id"]
        orf_id = (mibig_id, row["orf_id"])
        orf_lookup[orf_id] = row["cluster_id"]
        # ignore orfs that do not form any groups
        if family_id == -1:
            continue
        if family_id not in orf_group_to_mibig:
            orf_group_to_mibig[family_id] = set()
        orf_group_to_mibig[family_id].add(mibig_id)
        if mibig_id not in mibig_to_orfs:
            mibig_to_orfs[mibig_id] = set()
        mibig_to_orfs[mibig_id].add(orf_id)
        orf_to_group[orf_id] = family_id
    # run analysis on each mibig
    total_mibig = set(mibig_to_units) & set(mibig_to_orfs)
    all_associations = []
    for mibig_id in tqdm(total_mibig):
        out = []
        combinations = list(
            it.product(mibig_to_units[mibig_id], mibig_to_orfs[mibig_id])
        )
        for unit_id, orf_id in tqdm(combinations, leave=False):
            unit_group_id = unit_to_group[unit_id]
            orf_group_id = orf_to_group[orf_id]
            overlap = (
                unit_group_to_mibig[unit_group_id]
                & orf_group_to_mibig[orf_group_id]
            )
            unit_only = unit_group_to_mibig[unit_group_id] - overlap
            orf_only = orf_group_to_mibig[orf_group_id] - overlap
            unison = overlap | unit_only | orf_only
            absent = total_mibig - unison
            contigency_matrix = [
                [len(overlap), len(unit_only)],
                [len(orf_only), len(absent)],
            ]
            chi2, p_value, dof, expected = chi2_contingency(contigency_matrix)
            jaccard_score = round(len(overlap) / len(unison), 2)
            out.append(
                {
                    "mibig_id": mibig_id,
                    "metabolite_id": unit_lookup[unit_id],
                    "cluster_id": orf_lookup[orf_id],
                    "unit_id": unit_id[1],
                    "orf_id": orf_id[1],
                    "overlap": len(overlap),
                    "unit_only": len(unit_only),
                    "orf_only": len(orf_only),
                    "absent": len(absent),
                    "jaccard_score": jaccard_score,
                    "chi2": round(chi2, 5),
                    "p_value": round(p_value, 5),
                    "dof": dof,
                }
            )
        # normalize p-values
        p_values = [x["p_value"] for x in out]
        bh_corr_p = smt.multipletests(p_values, method="fdr_bh")[1]
        bon_corr_p = smt.multipletests(p_values, method="bonferroni")[1]
        for idx, row in enumerate(out):
            row["bh_corr_p"] = round(bh_corr_p[idx], 5)
            row["bon_corr_p"] = round(bon_corr_p[idx], 5)
        all_associations.extend(out)
    return all_associations


def get_unit_module_associations(
    units_hdbscan_fp: str, modules_hdbscan_fp: str
):
    unit_lookup = {}
    module_lookup = {}
    unit_group_to_mibig = {}
    module_group_to_mibig = {}
    unit_to_group = {}
    module_to_group = {}
    mibig_to_units = {}
    mibig_to_modules = {}
    # reorganize unit data
    unit_data = pd.read_csv(units_hdbscan_fp).to_dict("records")
    for row in unit_data:
        family_id = row["family_id"]
        mibig_id = row["mibig_id"]
        unit_id = (mibig_id, row["unit_id"])
        unit_lookup[unit_id] = row["metabolite_id"]
        # ignore units that do not form any groups
        if family_id == -1:
            continue
        if family_id not in unit_group_to_mibig:
            unit_group_to_mibig[family_id] = set()
        unit_group_to_mibig[family_id].add(mibig_id)
        if mibig_id not in mibig_to_units:
            mibig_to_units[mibig_id] = set()
        mibig_to_units[mibig_id].add(unit_id)
        unit_to_group[unit_id] = family_id
    # reorganize orf data
    module_data = pd.read_csv(modules_hdbscan_fp).to_dict("records")
    for row in module_data:
        family_id = row["family_id"]
        mibig_id = row["mibig_id"]
        module_id = (mibig_id, row["module_id"])
        module_lookup[module_id] = row["cluster_id"]
        # ignore orfs that do not form any groups
        if family_id == -1:
            continue
        if family_id not in module_group_to_mibig:
            module_group_to_mibig[family_id] = set()
        module_group_to_mibig[family_id].add(mibig_id)
        if mibig_id not in mibig_to_modules:
            mibig_to_modules[mibig_id] = set()
        mibig_to_modules[mibig_id].add(module_id)
        module_to_group[module_id] = family_id
    # run analysis on each mibig
    total_mibig = set(mibig_to_units) & set(mibig_to_modules)
    all_associations = []
    for mibig_id in tqdm(total_mibig):
        out = []
        combinations = list(
            it.product(mibig_to_units[mibig_id], mibig_to_modules[mibig_id])
        )
        for unit_id, module_id in tqdm(combinations, leave=False):
            unit_group_id = unit_to_group[unit_id]
            module_group_id = module_to_group[module_id]
            overlap = (
                unit_group_to_mibig[unit_group_id]
                & module_group_to_mibig[module_group_id]
            )
            unit_only = unit_group_to_mibig[unit_group_id] - overlap
            module_only = module_group_to_mibig[module_group_id] - overlap
            unison = overlap | unit_only | module_only
            absent = total_mibig - unison
            contigency_matrix = [
                [len(overlap), len(unit_only)],
                [len(module_only), len(absent)],
            ]
            chi2, p_value, dof, expected = chi2_contingency(contigency_matrix)
            jaccard_score = round(len(overlap) / len(unison), 2)
            out.append(
                {
                    "mibig_id": mibig_id,
                    "metabolite_id": unit_lookup[unit_id],
                    "cluster_id": module_lookup[module_id],
                    "unit_id": unit_id[1],
                    "module_id": module_id[1],
                    "overlap": len(overlap),
                    "unit_only": len(unit_only),
                    "module_only": len(module_only),
                    "absent": len(absent),
                    "jaccard_score": jaccard_score,
                    "chi2": round(chi2, 5),
                    "p_value": round(p_value, 5),
                    "dof": dof,
                }
            )
        # normalize p-values
        p_values = [x["p_value"] for x in out]
        bh_corr_p = smt.multipletests(p_values, method="fdr_bh")[1]
        bon_corr_p = smt.multipletests(p_values, method="bonferroni")[1]
        for idx, row in enumerate(out):
            row["bh_corr_p"] = round(bh_corr_p[idx], 5)
            row["bon_corr_p"] = round(bon_corr_p[idx], 5)
        all_associations.extend(out)
    return all_associations
