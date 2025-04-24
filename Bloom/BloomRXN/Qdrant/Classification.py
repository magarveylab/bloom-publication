from collections import Counter
from functools import partial
from typing import List

from Bloom.BloomRXN.Qdrant import Databases
from Bloom.BloomRXN.Qdrant.Datastructs import (
    DataQuery,
    DistHitResponse,
    KnnOutput,
)
from Bloom.BloomRXN.Qdrant.QdrantBase import QdrantBase


def ontology_neighborhood_classification(
    hits: List[DistHitResponse],
    top_n: int = 10,
    return_n: int = 5,
):
    # only consider top n hits
    hits = sorted(hits, key=lambda x: x["distance"])[:top_n]
    # some reference might have multiple labels - split these cases
    all_labels = []
    all_observed = []
    neighborhood = []
    for h in hits:
        observed = set()
        labels = [h["label"]] if isinstance(h["label"], str) else h["label"]
        all_labels.extend(labels)
        for label in labels:
            toks = label.split(".")
            top_level = len(toks)
            breakdown = [".".join(toks[:l]) for l in range(1, top_level + 1)]
            neighborhood.append(
                {
                    "reference_id": h["subject_id"],
                    "label": label,
                    "distance": h["distance"],
                    "breakdown": breakdown,
                }
            )
            observed.update(breakdown)
        all_observed.extend(breakdown)
    # if no hits present at distance cutoff
    if len(neighborhood) == 0:
        return []
    # calculate frequency of observed label
    label_freq = Counter(all_observed)
    # annotate each hit by observed frequency in neighborhood
    for n in neighborhood:
        n["scores"] = [label_freq[e] for e in n["breakdown"]]
    # reorganize into labels
    lookup = {}
    distance_lookup = {}
    for n in neighborhood:
        label = n["label"]
        distance = n["distance"]
        if label not in lookup:
            lookup[label] = {"scores": n["scores"], "distance": []}
            distance_lookup[label] = {}
        lookup[label]["distance"].append(distance)
        distance_lookup[label][distance] = n["reference_id"]
    # choose best observed by frequency and then by distance
    return_labels = sorted(
        lookup,
        key=lambda x: (lookup[x]["scores"], -min(lookup[x]["distance"])),
        reverse=True,
    )[:return_n]
    # response
    response = []
    rank = 1
    for label in return_labels:
        c = all_labels.count(label)
        homology_score = round(c / top_n, 2)
        distance = min(lookup[label]["distance"])
        # output
        reference_id = distance_lookup[label][distance]
        output = {
            "label": label,
            "reference_id": reference_id,
            "homology": homology_score,
            "rank": rank,
        }
        output["distance"] = distance
        response.append(output)
        rank += 1
    return response


def KNNClassification(
    query_list: List[DataQuery],
    qdrant_db: QdrantBase = None,
    top_n: int = 5,
    dist_cutoff: float = 500.0,  # arbitrarily large.
    batch_size: int = 100,
    return_n: int = 1,
    ignore_self_matches: bool = False,
) -> List[KnnOutput]:
    # Initialize Qdrant Database
    db = qdrant_db()
    # run KNN
    predictions = db.batch_search(
        queries=query_list,
        batch_size=batch_size,
        max_results=top_n,
        return_embeds=False,
        return_data=True,
        distance_cutoff=dist_cutoff,
        ignore_self_matches=ignore_self_matches,
    )
    # classification
    response = []
    for p in predictions:
        query, hits = p["query_id"], p["hits"]
        if len(hits) > 0:
            cls_result = ontology_neighborhood_classification(
                hits,
                top_n=top_n,
                return_n=return_n,
            )
        else:
            cls_result = []
        response.append({"query_id": query, "predictions": cls_result})
    # terminate connection
    del db
    return response


# Define KNN functions
rxn_ec_classification = partial(
    KNNClassification, qdrant_db=Databases.RXNReference
)
