from typing import List

import numpy as np
import pandas as pd

from Bloom import dataset_dir
from Bloom.BloomLNK.chemotypes import (
    normalize_bgc_chemotypes,
    sorted_metabolites,
)
from Bloom.BloomLNK.inference.Pipeline import LNKPipeline
from Bloom.BloomLNK.jaccard import run_jaccard_on_ibis_result
from Bloom.BloomLNK.local.bgc_graph import (
    get_embeddings_for_bgc_graph,
    get_orf_to_dags,
    quality_control_bgc_filtering,
)
from Bloom.BloomLNK.utils import curdir


# load thresholds
def load_lnk_thresholds():
    thresholds_fp = f"{curdir}/inference/thresholds.csv"
    data = pd.read_csv(thresholds_fp).to_dict("records")
    return {r["chemotype"]: r for r in data}


lnk_thresholds = load_lnk_thresholds()


def get_bgc_mol_associations(
    ibis_dir: str,
    output_fp: str,
    min_orf_count: int = 4,
    min_module_count: int = 4,
    sm_dag_dir: str = f"{dataset_dir}/sm_dags",
    sm_graph_dir: str = f"{dataset_dir}/sm_graphs",
    top_n_jaccard: int = 1000,
    report_top_n: int = 10,
    only_consider_metabolite_ids: List[int] = None,
):
    filtered_bgcs = quality_control_bgc_filtering(
        ibis_dir=ibis_dir,
        min_orf_count=min_orf_count,
        min_module_count=min_module_count,
    )
    if len(filtered_bgcs) == 0:
        raise ValueError(
            "No BGCs passed the filtering criteria. Please check your parameters."
        )
    # find metabolites to run
    normalized_filtered_bgcs = normalize_bgc_chemotypes(filtered_bgcs)
    bgc_to_metabolites_to_run = {}
    for bgc in normalized_filtered_bgcs:
        cluster_id = bgc["cluster_id"]
        bgc_to_metabolites_to_run[cluster_id] = set()
        for chemotype in bgc["chemotypes"]:
            bgc_to_metabolites_to_run[cluster_id].update(
                sorted_metabolites.get(chemotype, [])
            )
    if only_consider_metabolite_ids != None:
        to_remove = set()
        for bgc in bgc_to_metabolites_to_run:
            bgc_to_metabolites_to_run[bgc] = set(
                bgc_to_metabolites_to_run[bgc]
            ) & set(only_consider_metabolite_ids)
            if len(bgc_to_metabolites_to_run[bgc]) == 0:
                to_remove.add(bgc)
        for bgc in to_remove:
            del bgc_to_metabolites_to_run[bgc]
    # run jaccard similarity scores
    clusters_to_run = list(bgc_to_metabolites_to_run.keys())
    metabolites_to_run = set(
        m
        for metabolite_ids in bgc_to_metabolites_to_run.values()
        for m in metabolite_ids
    )
    jaccard_matrix = run_jaccard_on_ibis_result(
        ibis_dir=ibis_dir,
        clusters_to_run=clusters_to_run,
        metabolites_to_run=metabolites_to_run,
    )
    # filter metabolites to run with Graphormer
    for cluster_id, metabolites_to_run in bgc_to_metabolites_to_run.items():
        ranked_metabolites = sorted(
            metabolites_to_run,
            key=lambda x: jaccard_matrix[cluster_id].get(x, 0),
            reverse=True,
        )
        bgc_to_metabolites_to_run[cluster_id] = ranked_metabolites[
            :top_n_jaccard
        ]
    # run Graphormer
    orf_to_dags = get_orf_to_dags(ibis_dir=ibis_dir)
    embeddings = get_embeddings_for_bgc_graph(ibis_dir=ibis_dir)
    pipe = LNKPipeline(gpu_id=0)
    graphormer_matrix = {}
    for cluster_id, metabolites_to_run in bgc_to_metabolites_to_run.items():
        if len(metabolites_to_run) == 0:
            continue
        graphormer_matrix[cluster_id] = {}
        result = pipe.run_on_ibis_result(
            ibis_dir=ibis_dir,
            metabolites_to_run=metabolites_to_run,
            clusters_to_run=[cluster_id],
            orf_to_dags=orf_to_dags,
            embeddings=embeddings,
            sm_dag_dir=sm_dag_dir,
            sm_graph_dir=sm_graph_dir,
        )
        for i in result:
            metabolite_id = i["metabolite_id"]
            j = jaccard_matrix[cluster_id][metabolite_id]
            s1 = i["s1"]
            s2 = i["s2"]
            s3 = i["s3"]
            s4 = i["s4"]
            s5 = i["s5"]
            graphormer_matrix[cluster_id][metabolite_id] = {
                "jaccard": j,
                "s1": s1,
                "s2": s2,
                "s2-5-j": np.mean([np.mean([s2, s5]), j]),
                "s3": s3,
                "s3-5": np.mean([s3, s5]),
                "s4": s4,
                "s4-5": np.mean([s4, s5]),
                "s4-5-j": np.mean([np.mean([s4, s5]), j]),
                "s4-j": np.mean([s4, j]),
                "s5": s5,
                "s5-j": np.mean([s5, j]),
            }
    # find hits that pass threshold
    to_export = []
    for bgc in filtered_bgcs:
        hits = []
        cluster_id = bgc["cluster_id"]
        contig_id, contig_start, contig_stop = cluster_id.split("_")
        contig_id = int(contig_id)
        contig_start = int(contig_start)
        contig_stop = int(contig_stop)
        if cluster_id not in graphormer_matrix:
            continue
        chemotypes = bgc["chemotypes"]
        threshold_key = None
        if (
            "NonRibosomalPeptide" in chemotypes
            and "TypeIPolyketide" in chemotypes
        ):
            threshold_key = "Hybrid"
        elif "NonRibosomalPeptide" in chemotypes:
            threshold_key = "NonRibosomalPeptide"
        elif "TypeIPolyketide" in chemotypes:
            threshold_key = "TypeIPolyketide"
        else:
            threshold_key = chemotypes[0]
        if threshold_key not in lnk_thresholds:
            continue
        metric = lnk_thresholds[threshold_key]["metric"]
        threshold = lnk_thresholds[threshold_key]["threshold"]
        for metabolite_id in graphormer_matrix[cluster_id]:
            j = graphormer_matrix[cluster_id][metabolite_id]["jaccard"]
            score = graphormer_matrix[cluster_id][metabolite_id][metric]
            if score > threshold:
                hits.append(
                    {
                        "contig_id": contig_id,
                        "contig_start": contig_start,
                        "contig_stop": contig_stop,
                        "chemotype": threshold_key,
                        "metabolite_id": metabolite_id,
                        "jaccard_score": j,
                        "graphormer_metric": metric,
                        "graphormer_score": round(score, 5),
                        "rank": None,
                    }
                )
        # sort by score and update rank
        hits = sorted(hits, key=lambda x: x["graphormer_score"], reverse=True)
        for i, r in enumerate(hits, 1):
            r["rank"] = i
        to_export.extend(hits[:report_top_n])
    # export
    pd.DataFrame(to_export).to_csv(output_fp, index=False)
