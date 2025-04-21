import json
from glob import glob
from typing import Dict, Optional

import torch
from torch_geometric.data import Batch

from Bloom.BloomEmbedder import curdir
from Bloom.BloomEmbedder.graphs.MoleculeGraph import (
    MoleculeGraph,
    get_edge_vocab,
    get_node_vocab,
)
from Bloom.CommonUtils.HeteroGraph import (
    batch_to_homogeneous,
    get_lookup_from_hetero,
)


class MolPipeline:

    def __init__(self, gpu_id: Optional[int] = None):
        edge_types = [
            ("Atom", "bond", "Atom"),
            ("Atom", "atom_to_unit", "MetaboliteMolecularRegion"),
            (
                "MetaboliteMolecularRegion",
                "mol_region_adj",
                "MetaboliteMolecularRegion",
            ),
        ]
        self.edge_type_lookup = {e[1]: e for e in edge_types}
        # load vocab
        self.node_vocab = get_node_vocab()
        self.edge_vocab = get_edge_vocab()
        # load models
        model_dir = f"{curdir}/models/molecule"
        # load node encoders
        self.node_encoders = {}
        for model_fp in glob(f"{model_dir}/node_encoders/*"):
            node_type = model_fp.split("/")[-1].split(".")[0]
            self.node_encoders[node_type] = torch.jit.load(model_fp)
        # load edge encoders
        self.edge_encoders = {}
        for model_fp in glob(f"{model_dir}/edge_encoders/*"):
            edge_type = model_fp.split("/")[-1].split(".")[0]
            self.edge_encoders[edge_type] = torch.jit.load(model_fp)
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
            for edge_type in self.edge_encoders:
                self.edge_encoders[edge_type].to(f"cuda:{self.gpu_id}")
            self.edge_type_encoder.to(f"cuda:{self.gpu_id}")
            self.gnn.to(f"cuda:{self.gpu_id}")
            self.transformer.to(f"cuda:{self.gpu_id}")

    def __call__(
        self,
        G: MoleculeGraph,
        unit_traceback: Dict[int, str],  # graph_node_id -> unit_node_id
    ):
        metabolite_id = G.graph_id
        data = self.preprocess(G=G)
        result = self._forward(data=data)
        result_reformatted = {
            "metabolite_id": metabolite_id,
            "Molecule": result["Molecule"],
            "Unit": {},
        }
        for node_id, unit_id in unit_traceback.items():
            if node_id in result["Unit"]:
                result_reformatted["Unit"][unit_id] = result["Unit"][node_id]
        return result_reformatted

    def embed_molecule_from(self, metabolite_id: int, bear_fp: str):
        r = MoleculeGraph.build_from_bear_output(
            graph_id=metabolite_id, bear_fp=bear_fp
        )
        out = self(
            G=r["graph"],
            unit_traceback=r["unit_traceback"],
        )
        # add smiles (useful meta data)
        out["smiles"] = json.load(open(bear_fp))["smiles"]
        return out

    def preprocess(self, G: MoleculeGraph) -> Batch:
        # prepare tensor
        data = G.get_tensor_data(
            node_vocab=self.node_vocab,
            edge_vocab=self.edge_vocab,
            node_types_to_consider=None,  # default consider all
            edge_types_to_consider=None,  # default consider all
            apply_edge_attr=True,
            apply_multigraph_wrapper=False,
        )
        data = Batch.from_data_list([data])
        return data

    def _forward(self, data: Batch):
        if isinstance(self.gpu_id, int):
            data.to(f"cuda:{self.gpu_id}")
        # preprocess node encoding (all types should be converted to same dimensionality)
        for node_type, node_encoder in self.node_encoders.items():
            data[node_type]["x"] = node_encoder(data[node_type]["x"], None)
        # preprocess edge encoding (all types should be converted to same dimensionality)
        for edge_name, edge_encoder in self.edge_encoders.items():
            edge_type = self.edge_type_lookup[edge_name]
            data[edge_type]["edge_attr"] = edge_encoder(
                data[edge_type]["edge_attr"],
                None,
            )
        # convert heterogenous to homogenous
        lookup = get_lookup_from_hetero(data)
        data = batch_to_homogeneous(data)
        # edge encode by edge type
        data.edge_attr = self.edge_type_encoder(data.edge_type, data.edge_attr)
        # message passing
        data.x = self.gnn(data.x, data.edge_index, data.edge_attr)
        # transformer (global attention accross nodes)
        data.x = self.transformer(data.x, data.batch)
        # convert homogenous to heterogenous
        data = data.to_heterogeneous()
        # get bgc embedding
        mol_embedding = data[lookup["Molecule"]].x.cpu().detach().numpy()[1]
        # get unit embeddings
        unit_key = lookup["MetaboliteMolecularRegion"]
        unit_x = data[unit_key].x.cpu().detach().numpy()
        unit_node_ids = data[unit_key].node_ids.cpu().detach().numpy()
        unit_contex_emb = dict(zip(list(unit_node_ids), unit_x))
        return {"Molecule": mol_embedding, "Unit": unit_contex_emb}
