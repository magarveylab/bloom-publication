import pickle

import numpy as np
import pandas as pd


def generate_molecular_embeddings(
    bloom_graph_fp: str,
    output_fp: str,
    gpu_id: int = 0,
):
    from Bloom.BloomEmbedder.inference.MolPipeline import MolPipeline

    pipe = MolPipeline(gpu_id=gpu_id)
    out = pipe.embed_molecule_from(
        metabolite_id=1,  # this is a dummy id
        bloom_graph_fp=bloom_graph_fp,
    )
    # save the output
    pickle.dump(out, open(output_fp, "wb"))


def generate_gene_embeddings(
    ibis_dir: str,
    output_fp: str,
    gpu_id: int = 0,
):
    from Bloom.BloomEmbedder.inference.BGCPipeline import BGCPipeline

    pipe = BGCPipeline(gpu_id=gpu_id)
    out = pipe.embed_clusters_from(ibis_dir=ibis_dir)
    # save the output
    pickle.dump(out, open(output_fp, "wb"))


def compute_clustering_from_matrix(
    matrix_fp: str,
    matrix_keys_fp: str,
    output_fp: str,
):
    from Bloom.BloomEmbedder.clustering import compute_clustering

    matrix = np.load(matrix_fp)
    matrix_keys = pd.read_csv(matrix_keys_fp).to_dict("records")
    out = compute_clustering(matrix=matrix, matrix_keys=matrix_keys)
    pd.DataFrame(out).to_csv(output_fp, index=False)
