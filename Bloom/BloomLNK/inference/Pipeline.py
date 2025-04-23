import json
import pickle
from glob import glob
from typing import List, Literal, Optional

import numpy as np
import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from Bloom import dataset_dir
from Bloom.BloomLNK import curdir
from Bloom.BloomLNK.graph import MetaboloGraph, get_vocab
from Bloom.BloomLNK.local.bgc_graph import (
    get_bgc_graphs,
    get_embeddings_for_bgc_graph,
)
from Bloom.BloomLNK.local.unison import unite_graphs
from Bloom.CommonUtils.HeteroGraph import (
    batch_to_homogeneous,
    get_lookup_from_hetero,
)

VersionOptions = Literal[
    "final", "kfold-0", "kfold-1", "kfold-2", "kfold-3", "kfold-4"
]


class BearLinkerPipeline:

    def __init__(
        self, version: VersionOptions = "final", gpu_id: Optional[int] = None
    ):
        super().__init__()
        self.version = version
        self.node_types_with_embedding = ["Orf", "Domain", "Reaction"]
        # load vocab
        self.node_vocab = get_vocab()
        # load models
        model_dir = f"{curdir}/models/{self.version}"
        # load node encoders
        self.node_encoders = {}
        for model_fp in glob(f"{model_dir}/node_encoders/*"):
            node_type = model_fp.split("/")[-1].split("_node_encoder")[0]
            self.node_encoders[node_type] = torch.jit.load(model_fp)
        # load edge type encoder
        self.edge_type_encoder = torch.jit.load(
            f"{model_dir}/edge_type_encoder.pt"
        )
        # load message passing nn
        self.gnn = torch.jit.load(f"{model_dir}/gnn.pt")
        # load transformer
        self.transformer = torch.jit.load(f"{model_dir}/transformer.pt")
        # load graph pooler
        self.graph_pooler = torch.jit.load(f"{model_dir}/graph_pooler.pt")
        # load heads
        self.heads = {}
        for model_fp in glob(f"{model_dir}/heads/*"):
            head_name = model_fp.split("/")[-1].split("_head")[0]
            self.heads[head_name] = torch.jit.load(model_fp)
        # move models to gpu (if device defined)
        self.gpu_id = gpu_id
        if isinstance(self.gpu_id, int):
            for node_type in self.node_encoders:
                self.node_encoders[node_type].to(f"cuda:{self.gpu_id}")
            self.edge_type_encoder.to(f"cuda:{self.gpu_id}")
            self.gnn.to(f"cuda:{self.gpu_id}")
            self.transformer.to(f"cuda:{self.gpu_id}")
            self.graph_pooler.to(f"cuda:{self.gpu_id}")
            for head_name in self.heads:
                self.heads[head_name].to(f"cuda:{self.gpu_id}")

    def __call__(self, G: MetaboloGraph):
        data = self.preprocess(G)
        output = self._forward(data)
        metabolite_id, cluster_id = G.graph_id.split("-")
        try:
            output["metabolite_id"] = int(metabolite_id)
        except:
            output["metabolite_id"] = metabolite_id
        output["cluster_id"] = cluster_id
        return output

    def preprocess(self, G: MetaboloGraph) -> Batch:
        # prepare tensor
        data = G.get_tensor_data(
            node_vocab=self.node_vocab,
            edge_vocab={},
            node_types_to_consider=None,  # default consider all
            edge_types_to_consider=None,  # default consider all
            apply_edge_attr=False,
        )
        data = Batch.from_data_list([data])
        return data

    def _forward(self, data: Batch) -> np.array:
        if isinstance(self.gpu_id, int):
            data = data.to(f"cuda:{self.gpu_id}")
        # preprocess node encoding (all types should be converted to same dimensionality)
        for node_type, node_encoder in self.node_encoders.items():
            if node_type in self.node_types_with_embedding:
                data[node_type]["x"] = node_encoder(data[node_type]["x"])
            else:  # label
                data[node_type]["x"] = node_encoder(
                    data[node_type]["x"], data[node_type].get("extra_x", None)
                )
        # convert heterogenous to homogenous
        lookup = get_lookup_from_hetero(data)
        data = batch_to_homogeneous(data)
        # edge encode by edge type
        data.edge_attr = self.edge_type_encoder(
            data.edge_type, getattr(data, "edge_attr", None)
        )
        # message passing
        data.x = self.gnn(data.x, data.edge_index, data.edge_attr)
        # transformer (global attention accross nodes)
        data.x = self.transformer(data.x, data.batch)
        # convert homogenous to heterogenous
        data = data.to_heterogeneous()
        # get pooled output
        pooled_output = self.graph_pooler(
            data[lookup[self.graph_pooler.node_type]]["x"],
            data[lookup[self.graph_pooler.node_type]]["batch"],
        )
        # get heads
        output = {}
        for head_name, head in self.heads.items():
            logits = head(pooled_output)
            logits = logits.cpu().detach().numpy()[0]
            logits = self.softmax(logits)
            output[head_name] = round(float(logits[-1]), 3)
        return output

    def run_from_local_filenames(
        self,
        ibis_dir: str,
        genome: str,
        metabolite_ids: List[int],
        clusters_to_run: Optional[List[str]] = None,
        sm_dag_dir: str = f"{dataset_dir}/sm_dags",
        sm_graph_dir: str = f"{dataset_dir}/sm_graphs",
        min_orf_count: int = 4,
        min_module_count: int = 4,
        run_all: bool = False,
    ):
        # prepare bgc graphs
        ibis_genome_dir = f"{ibis_dir}/{genome}"
        bgc_graphs = get_bgc_graphs(
            ibis_dir=ibis_genome_dir,
            min_orf_count=min_orf_count,
            min_module_count=min_module_count,
            run_all=run_all,
        )
        bgc_graphs = {i["cluster_id"]: i["graph"] for i in bgc_graphs}
        if len(bgc_graphs) == 0:
            return []
        # prepare bgc embeddings
        embeddings = get_embeddings_for_bgc_graph(ibis_dir=ibis_genome_dir)
        # run analysis
        output = []
        if clusters_to_run == None:
            clusters_to_run = list(bgc_graphs.keys())
        for m in tqdm(metabolite_ids, desc="Inference"):
            # load metabolite graph
            mol_G = pickle.load(open(f"{sm_graph_dir}/{m}.pkl", "rb"))
            mol_dags = json.load(open(f"{sm_dag_dir}/{m}.json", "r"))
            for c in tqdm(clusters_to_run, leave=False):
                bgc_G = bgc_graphs[c]
                orf_embedding = embeddings[c]["orfs"]
                domain_embedding = embeddings[c]["domains"]
                unified_G = unite_graphs(
                    mol_G=mol_G,
                    bgc_G=bgc_G,
                    mol_dags=mol_dags,
                    min_sugar_rxn_freq=0,
                    min_rxn_anno_freq=0,
                    min_rule_rxn_freq=0,
                )
                G = MetaboloGraph.build_graph(
                    graph_id=f"{m}-{c}",
                    graph=unified_G,
                    orf_embedding=orf_embedding,
                    domain_embedding=domain_embedding,
                )
                output.append(self(G))
        return output

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.exp(x).sum(-1, keepdims=True)
