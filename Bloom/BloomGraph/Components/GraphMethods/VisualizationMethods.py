from typing import List

import pandas as pd

from Bloom.BloomGraph.Visualization.Colour import get_colour_palette
from Bloom.BloomGraph.Visualization.Highlight import highlight_mol
from Bloom.BloomGraph.Visualization.Image import get_image


class VisualizationMethods:

    def visualize_by_units(self, tmp_dir: str = "image_cache"):
        # map colouring to nodes
        units = set(self[n]["top_label"] for n in self.nodes)
        colour_tups = get_colour_palette(len(units))
        units_to_colour = {u: c for u, c in zip(units, colour_tups)}
        nodes_to_colour = {
            n: units_to_colour[self[n]["top_label"]] for n in self.nodes
        }
        # add colout to atoms
        atoms_to_colour = {}
        for n, atoms in self.node_to_atoms.items():
            for a in atoms:
                atoms_to_colour[a] = units_to_colour[self[n]["top_label"]]
        # get image data
        image_data = highlight_mol(
            self.smiles,
            atoms_to_colour=atoms_to_colour,
            node_to_atoms=self.node_to_atoms,
        )
        # colour map is used to contruct the legend
        image_data["colour_map"] = units_to_colour
        image = get_image(image_data, tmp_dir, self.hash_id, add_legend=True)
        return image

    def tabulate_nodes(self, meta_columns: List[str] = []) -> pd.DataFrame:
        table = []
        for n in self.nodes:
            row = {
                "node_id": n,
                "label": self[n]["top_label"],
                "count": self[n]["count"],
                "enums": list(self[n]["enums"]),
            }
            meta = self[n]["meta"]
            for c in meta_columns:
                if c in ["orfs"]:
                    default = []
                elif c in [
                    "anno_score",
                    "module_orf",
                    "module_start",
                    "module_end",
                ]:
                    default = -1
                else:
                    default = None
                row[c] = meta.get(c, default)
            table.append(row)
        return pd.DataFrame(table)

    def tabulate_edges(self, meta_columns: List[str] = []) -> pd.DataFrame:
        table = []
        for n1, n2, data in self.G.edges(data=True):
            meta = data["meta"]
            row = {"n1": n1, "n2": n2}
            for c in meta_columns:
                if c in ["orfs"]:
                    default = []
                elif c in ["module_orf", "module_start", "module_end"]:
                    default = -1
                else:
                    default = None
                row[c] = meta.get(c, default)
            table.append(row)
        return pd.DataFrame(table)
