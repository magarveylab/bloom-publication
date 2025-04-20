from typing import Dict, List, Optional, TypedDict


class NodeLabelDict(TypedDict):
    label: str
    score: float


class BloomNodeDict(TypedDict):
    node_id: int
    labels: List[NodeLabelDict]
    enums: List[str]
    count: int
    meta: dict


class BloomEdgeDict(TypedDict):
    n1: int
    n2: int
    meta: dict


class BloomGraphDict(TypedDict):
    nodes: List[BloomNodeDict]
    edges: List[BloomEdgeDict]
    smiles: str  # mapped smiles -> atoms correspond to nodes


class ImageDataDict(TypedDict):
    svg: str
    colour_map: Optional[Dict[str, str]]
