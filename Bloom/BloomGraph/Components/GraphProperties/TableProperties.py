import pandas as pd


class TableProperties:

    @property
    def node_table(self) -> pd.DataFrame:
        meta_columns = ["description", "atom_indexes"]
        return self.tabulate_nodes(meta_columns)
