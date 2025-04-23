import pickle


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
