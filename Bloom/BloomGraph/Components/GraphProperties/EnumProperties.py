from typing import Dict, Set


class EnumProperties:

    @property
    def all_enums(self) -> Set[str]:
        return {e for n in self.nodes for e in self[n].get("enums", [])}

    @property
    def _enum_to_nodes(self) -> Dict[str, Set[int]]:
        cache = {e: set() for e in self.all_enums}
        for n in self.nodes:
            for e in self[n].get("enums", []):
                cache[e].add(n)
        return cache
