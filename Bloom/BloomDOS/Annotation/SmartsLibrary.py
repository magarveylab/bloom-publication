import json
import os

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from Bloom.BloomDOS.Annotation.EnhancedSmarts import EnhancedSmarts
from Bloom.BloomDOS.GeneralUtils import library_filter_fp, rendered_lib_dir

####################################################################
# Annotation Library
####################################################################


def load_smarts_library(key: str):
    fp = f"{rendered_lib_dir}/{key}.json"
    if os.path.exists(fp):
        message = "Loading {} library".format(key)
        return [
            EnhancedSmarts._load_from_dict(row)
            for row in tqdm(json.load(open(fp)), desc=message)
        ]
    else:
        return []


smarts_library = {
    "aminoacid": load_smarts_library("aminoacid"),
    "azole": load_smarts_library("azole"),
    "fattyacid": load_smarts_library("fattyacid"),
    "miscellaneous": load_smarts_library("miscellaneous"),
    "primary_metabolite": load_smarts_library("primary_metabolite"),
    "secondary_metabolite": load_smarts_library("secondary_metabolite"),
    "polyketide": load_smarts_library("polyketide"),
    "nucleoside": load_smarts_library("nucleoside"),
    "sugar": load_smarts_library("sugar"),
    "tailoring": load_smarts_library("tailoring"),
    "alphasearch": load_smarts_library("alphasearch"),
    "alphachain": load_smarts_library("alphachain"),
    "terpene": load_smarts_library("terpene"),
    "curated_module": load_smarts_library("curated_module"),
    "type2pk": load_smarts_library("type2pk"),
}

####################################################################
# Library filters
####################################################################

library_filters = pd.read_csv(library_filter_fp).to_dict("records")
for f in library_filters:
    f["smarts"] = Chem.MolFromSmiles(f["patt"])
