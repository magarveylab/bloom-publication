from collections import Counter
from typing import Set

import networkx as nx


class ChainMethods:

    def get_chains(self, max_length: int = 5) -> Set[int]:
        chains = []
        kmers = {i: set() for i in range(1, max_length + 1)}
        nodes = self.nodes
        unknown_nodes = self.unknown_nodes
        annotated_nodes = nodes - self.unknown_nodes
        for start in annotated_nodes:
            for end in annotated_nodes:
                if start == end:
                    kmers[1].add((start,))
                else:
                    paths = list(
                        nx.all_simple_paths(
                            self.G,
                            source=start,
                            target=end,
                        )
                    )
                    for p in paths:
                        if set(p) & unknown_nodes:
                            continue
                        length = len(p)
                        if length > max_length:
                            continue
                        kmers[length].add(tuple(p))
        # translate kmers to notations
        for k, paths in kmers.items():
            translated_paths = []
            for p in paths:
                translated_paths.append(
                    tuple([self[n]["top_label"] for n in p])
                )
            counts = Counter(translated_paths)
            for p, c in counts.items():
                chains.append({"kmer": k, "path": list(p), "frequency": c})
        return chains
