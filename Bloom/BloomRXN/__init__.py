import pickle

import pandas as pd

from Bloom.BloomRXN.inference.Pipeline import ReactionInferencePipeline


def generate_embeddings_from_csv(csv_fp: str, output_fp: str, gpu_id: int = 0):
    pipe = ReactionInferencePipeline(gpu_id=gpu_id)
    data = pd.read_csv(csv_fp).to_dict("records")
    out = []
    for r in data:
        query_id = r["query_id"]
        smiles = r["smiles"]
        embedding = pipe(smiles)
        out.append({"query_id": query_id, "embedding": embedding})
    pickle.dump(out, open(output_fp, "wb"))


def generate_ec_from_csv(csv_fp: str, output_fp: str, gpu_id: int = 0):
    from Bloom.BloomRXN.Qdrant.Classification import rxn_ec_classification

    pipe = ReactionInferencePipeline(gpu_id=gpu_id)
    data = pd.read_csv(csv_fp).to_dict("records")
    query = []
    for r in data:
        query_id = r["query_id"]
        smiles = r["smiles"]
        embedding = pipe(smiles)
        query.append({"query_id": query_id, "embedding": embedding})
    out = rxn_ec_classification(query)
    # export
    to_export = []
    for r in out:
        for p in r["predictions"]:
            to_export.append(
                {
                    "query_id": r["query_id"],
                    "label": p["label"],
                    "homology": p["homology"],
                    "distance": p["distance"],
                    "rank": p["rank"],
                }
            )
    pd.DataFrame(to_export).to_csv(output_fp, index=False)
