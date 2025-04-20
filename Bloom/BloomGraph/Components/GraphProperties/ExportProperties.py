from typing import Dict, List

from Bloom.BloomGraph import DataStructs


class ExportProperties:

    @property
    def node_exports(self) -> List[DataStructs.BloomNodeDict]:
        nodes = []
        for n in self.nodes:
            if "top_label" not in self[n]:
                continue
            enums = list(self[n].get("enums", []))
            labels = [
                {"label": l, "score": s}
                for l, s in self[n].get("labels", {}).items()
            ]
            meta = self[n].get("meta", {})
            count = self[n].get("count", 1)
            nodes.append(
                {
                    "node_id": n,
                    "top_label": self[n]["top_label"],
                    "labels": labels,
                    "count": count,
                    "enums": enums,
                    "meta": meta,
                }
            )
        return nodes

    @property
    def node_exports_dict(self) -> Dict[int, DataStructs.BloomNodeDict]:
        return {n["node_id"]: n for n in self.node_exports}

    @property
    def edge_exports(self) -> List[DataStructs.BloomEdgeDict]:
        edges = []
        for n1, n2, e in self.G.edges(data=True):
            edges.append({"n1": n1, "n2": n2, "meta": e.get("meta", {})})
        return edges

    @property
    def graph_dict(self) -> DataStructs.BloomGraphDict:
        out = {
            "smiles": self.smiles,
            "nodes": self.node_exports,
            "edges": self.edge_exports,
            "meta": self.meta,
        }
        return out
