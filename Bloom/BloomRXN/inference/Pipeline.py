from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from Bloom.BloomRXN.inference import Preprocess
from Bloom.BloomRXN.utils import model_dir, vocab_dir


def get_vocab(fp: str) -> Dict[str, int]:
    df = pd.read_csv(fp)
    return dict(zip(df.word, df.index))


def get_atom_vocab() -> Dict[str, int]:
    return get_vocab(f"{vocab_dir}/atom_vocab.csv")


def get_bond_vocab() -> Dict[str, int]:
    return get_vocab(f"{vocab_dir}/bond_vocab.csv")


class ReactionInferencePipeline:

    def __init__(
        self,
        node_encoder_fp: str = f"{model_dir}/node_encoder.pt",
        edge_encoder_fp: str = f"{model_dir}/edge_encoder.pt",
        gnn_fp: str = f"{model_dir}/gnn.pt",
        transformer_fp: str = f"{model_dir}/transformer.pt",
        graph_pooler_fp: str = f"{model_dir}/graph_pooler.pt",
        gpu_id: Optional[int] = None,
    ):
        # load vocab
        self.atom_vocab = get_atom_vocab()
        self.bond_vocab = get_bond_vocab()
        # load models (torchscript format)
        self.node_encoder = torch.jit.load(node_encoder_fp)
        self.edge_encoder = torch.jit.load(edge_encoder_fp)
        self.gnn = torch.jit.load(gnn_fp)
        self.transformer = torch.jit.load(transformer_fp)
        self.graph_pooler = torch.jit.load(graph_pooler_fp)
        # move models to gpu (if device defined)
        self.gpu_id = gpu_id
        if isinstance(self.gpu_id, int):
            self.node_encoder.to(f"cuda:{self.gpu_id}")
            self.edge_encoder.to(f"cuda:{self.gpu_id}")
            self.gnn.to(f"cuda:{self.gpu_id}")
            self.transformer.to(f"cuda:{self.gpu_id}")
            self.graph_pooler.to(f"cuda:{self.gpu_id}")
        super().__init__()

    def __call__(self, smiles: str) -> np.array:
        data = self.preprocess(smiles)
        return self._forward(data)

    def preprocess(self, smiles: str) -> Data:
        graph = Preprocess.get_rxn_graph_from(smiles=smiles)
        return Preprocess.get_graph_tensor_from(
            G=graph, atom_vocab=self.atom_vocab, bond_vocab=self.bond_vocab
        )

    def _forward(self, data: Data, **forward_kwargs) -> np.array:
        data = Batch.from_data_list([data])
        if isinstance(self.gpu_id, int):
            data = data.to(f"cuda:{self.gpu_id}")
        # preprocess node and edge encoding
        data.x = self.node_encoder(data.x, data.extra_x)
        data.edge_attr = self.edge_encoder(
            data.edge_attr, data.extra_edge_attr
        )
        # message passing
        data.x = self.gnn(data.x, data.edge_index, data.edge_attr)
        # transformer (global attention accross nodes)
        data.x = self.transformer(data.x, data.batch)
        # graph readout
        pooled_output = self.graph_pooler(data.x, data.batch)
        return np.array(pooled_output.detach().cpu())[0]
