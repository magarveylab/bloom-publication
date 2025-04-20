import json


class ExportMethods:

    def export(self, filepath: str):
        json.dump(self.graph_dict, open(filepath, "w"))
