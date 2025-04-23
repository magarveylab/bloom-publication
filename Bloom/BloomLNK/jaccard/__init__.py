import os
import pickle
from glob import glob
from typing import List

from tqdm import tqdm

from Bloom import dataset_dir
from Bloom.BloomLNK.jaccard.jaccard_signature import (
    get_final_signature_from_molecule,
    get_final_signatures_from_genome,
)
from Bloom.BloomLNK.jaccard.utils import jaccard_similarity


def build_molecular_jacccard_signature_library(
    bloom_dos_pred_dir: str, output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    filenames = glob(f"{bloom_dos_pred_dir}/*")
    out = {}
    for fp in filenames:
        metabolite_id = int(fp.split("/")[-1].split(".")[0])
        out[metabolite_id] = get_final_signature_from_molecule(fp)
    output_fp = f"{output_dir}/molecular_jaccard_signature_library.pkl"
    pickle.dump(out, open(output_fp, "wb"))


def load_molecular_jaccard_signature_library():
    signature_fp = f"{dataset_dir}/molecular_jaccard_signature_library.pkl"
    if os.path.exists(signature_fp) == False:
        raise ValueError(
            f"Signature file {signature_fp} does not exist. Please run build_molecular_jacccard_signature_library() first."
        )
    return pickle.load(open(signature_fp, "rb"))


molecular_jaccard_signature_library = (
    load_molecular_jaccard_signature_library()
)


def run_jaccard_on_ibis_result(
    ibis_dir: str,
    clusters_to_run: List[str],
    metabolites_to_run: List[int],
):
    cluster_signatures = get_final_signatures_from_genome(
        ibis_dir=ibis_dir,
        cluster_ids=clusters_to_run,
    )
    out = {}
    for cluster_id, bgc_signature in tqdm(
        cluster_signatures.items(), desc="Running Jaccard"
    ):
        hits = {}
        for m in tqdm(metabolites_to_run, leave=False):
            if m not in molecular_jaccard_signature_library:
                continue
            metabolite_signature = molecular_jaccard_signature_library[m]
            jaccard = jaccard_similarity(bgc_signature, metabolite_signature)
            if jaccard > 0:
                hits[m] = jaccard
        out[cluster_id] = hits
    return out
