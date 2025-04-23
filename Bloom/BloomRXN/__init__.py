import pickle

import pandas as pd

from Bloom.BloomRXN.inference.Pipeline import ReactionInferencePipeline


def generate_embeddings_from_csv(csv_fp: str, output_fp: str, gpu_id: int = 0):
    pipe = ReactionInferencePipeline(gpu_id=gpu_id)
    data = pd.read_csv(csv_fp).to_dict("records")
    out = []
    for r in data:
        reaction_id = r["reaction_id"]
        smiles = r["smiles"]
        embedding = pipe(smiles)
        out.append({"reaction_id": reaction_id, "embedding": embedding})
    pickle.dump(out, open(output_fp, "wb"))
