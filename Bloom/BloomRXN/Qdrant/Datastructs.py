from typing import List, TypedDict

import numpy as np


class DataQuery(TypedDict):
    query_id: int  # identifier
    embedding: np.array


class DistHitResponse(TypedDict):
    subject_id: int
    distance: float
    label: str
    data: dict


class KnnOutput(TypedDict):
    hash_id: str
    label: str
    distance: float
    homology: float
    reference_id: int
