from glob import glob
from typing import Dict, Optional

import torch
from torch_geometric.data import Batch

from Bloom.BloomEmbedder.graphs.BGCGraph import BGCGraph, get_node_vocab
from Bloom.BloomEmbedder.utils import curdir
from Bloom.CommonUtils.HeteroGraph import (
    batch_to_homogeneous,
    get_lookup_from_hetero,
)


class BGCPipeline:

    def __init__(self, gpu_id: Optional[int] = None):
        self.node_types_with_embedding = ["Orf", "Domain"]
        # load vocab
        self.node_vocab = get_node_vocab()
        # load models
        model_dir = f"{curdir}/models/bgc"
        # load node encoders
        self.node_encoders = {}
        for model_fp in glob(f"{model_dir}/node_encoders/*"):
            node_type = model_fp.split("/")[-1].split(".")[0]
            self.node_encoders[node_type] = torch.jit.load(model_fp)
        # load edge type encoder
        self.edge_type_encoder = torch.jit.load(
            f"{model_dir}/edge_type_encoder.pt"
        )
        # load message passing nn
        self.gnn = torch.jit.load(f"{model_dir}/gnn.pt")
        # load transformer
        self.transformer = torch.jit.load(f"{model_dir}/transformer.pt")
        # move models to gpu (if devide defined)
        self.gpu_id = gpu_id
        if isinstance(self.gpu_id, int):
            for node_type in self.node_encoders:
                self.node_encoders[node_type].to(f"cuda:{self.gpu_id}")
            self.edge_type_encoder.to(f"cuda:{self.gpu_id}")
            self.gnn.to(f"cuda:{self.gpu_id}")
            self.transformer.to(f"cuda:{self.gpu_id}")

    def __call__(
        self,
        G: BGCGraph,
        orf_traceback: Dict[int, str],  # graph_node_id -> orf_id
        module_traceback: Dict[int, str],  # graph_node_id -> module_id
    ):
        cluster_id = G.graph_id
        data = self.preprocess(G=G)
        result = self._forward(data=data)
        result_reformatted = {
            "cluster_id": cluster_id,
            "BGC": result["BGC"],
            "Orf": {},
            "Module": {},
        }
        for node_id, orf_id in orf_traceback.items():
            if node_id in result["Orf"]:
                result_reformatted["Orf"][orf_id] = result["Orf"][node_id]
        for node_id, module_id in module_traceback.items():
            if node_id in result["Module"]:
                result_reformatted["Module"][module_id] = result["Module"][
                    node_id
                ]
        return result_reformatted

    def embed_clusters_from(self, ibis_dir: str):
        bgcs = BGCGraph.build_from_ibis_output(ibis_dir=ibis_dir)
        out = []
        for r in bgcs:
            out.append(
                self(
                    G=r["graph"],
                    orf_traceback=r["orf_traceback"],
                    module_traceback=r["module_traceback"],
                )
            )
        return out

    def preprocess(self, G: BGCGraph) -> Batch:
        # prepare tensor
        G.label_schema = None
        data = G.get_tensor_data(
            node_vocab=self.node_vocab,
            edge_vocab={},
            node_types_to_consider=None,  # default consider all
            edge_types_to_consider=None,  # default consider all
            apply_edge_attr=False,
        )
        data = Batch.from_data_list([data])
        return data

    def _forward(self, data: Batch):
        if isinstance(self.gpu_id, int):
            data.to(f"cuda:{self.gpu_id}")
        # preprocess node encoding (all types should be converted to same dimensionality)
        for node_type, node_encoder in self.node_encoders.items():
            if node_type in self.node_types_with_embedding:
                data[node_type]["x"] = node_encoder(data[node_type]["x"])
            else:  # label
                data[node_type]["x"] = node_encoder(data[node_type]["x"], None)
        # convert heterogenous to homogenous
        lookup = get_lookup_from_hetero(data)
        data = batch_to_homogeneous(data)
        # edge encode by edge type
        data.edge_attr = self.edge_type_encoder(data.edge_type, None)
        # message passing
        data.x = self.gnn(data.x, data.edge_index, data.edge_attr)
        # transformer (global attention accross nodes)
        data.x = self.transformer(data.x, data.batch)
        # convert homogenous to heterogenous
        data = data.to_heterogeneous()
        # get bgc embedding
        bgc_embedding = data[lookup["BGC"]].x.cpu().detach().numpy()[1]
        # get orf embeddings
        orf_x = data[lookup["Orf"]].x.cpu().detach().numpy()
        orf_node_ids = data[lookup["Orf"]].node_ids.cpu().detach().numpy()
        orf_contex_emb = dict(zip(list(orf_node_ids), orf_x))
        # get module embeddings
        module_x = data[lookup["Module"]].x.cpu().detach().numpy()
        module_node_ids = (
            data[lookup["Module"]].node_ids.cpu().detach().numpy()
        )
        module_contex_emb = dict(zip(list(module_node_ids), module_x))
        return {
            "BGC": bgc_embedding,
            "Orf": orf_contex_emb,
            "Module": module_contex_emb,
        }
