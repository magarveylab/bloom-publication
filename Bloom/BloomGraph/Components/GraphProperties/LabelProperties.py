from collections import Counter
from typing import Dict, Set, Tuple


class LabelProperties:

    @property
    def top_labels(self) -> Set[str]:
        return {self[n]["top_label"] for n in self.nodes}

    @property
    def enumerated_top_labels(self) -> Set[Tuple[str, int]]:
        labels = [self[n]["top_label"] for n in self.nodes]
        return {(x, i) for x, y in Counter(labels).items() for i in range(y)}

    @property
    def all_labels(self) -> Set[str]:
        return {l for n in self.nodes for l in self[n]["labels"]}

    @property
    def all_enumerated_labels(self) -> Set[Tuple[str, int]]:
        labels = [l for n in self.nodes for l in self[n]["labels"]]
        return {(x, i) for x, y in Counter(labels).items() for i in range(y)}

    @property
    def label_to_nodes(self) -> Dict[str, Set[int]]:
        cache = {l: set() for l in self.top_labels}
        for n in self.nodes:
            cache[self[n]["top_label"]].add(n)
        return cache
