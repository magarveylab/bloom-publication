import argparse
import json
import os
from typing import List

from Bloom.BloomDOS.DataStructs import EnhancedSmartsDict
from Bloom.BloomDOS.GeneralUtils import rendered_lib_dir
from Bloom.BloomDOS.Library.Curated.Miscellaneous import Miscellaneous
from Bloom.BloomDOS.Library.Curated.ModuleUnit import ModuleUnit
from Bloom.BloomDOS.Library.FattyAcid.FattyAcid import FattyAcid
from Bloom.BloomDOS.Library.Metabolite.Metabolite import (
    PrimaryMetabolite,
    SecondaryMetabolite,
)
from Bloom.BloomDOS.Library.Nucleoside.Nucleoside import Nucleoside
from Bloom.BloomDOS.Library.Peptide.AminoAcid import AminoAcid
from Bloom.BloomDOS.Library.Peptide.Azole import Azole
from Bloom.BloomDOS.Library.Polyketide.AlphaChains import AlphaChains
from Bloom.BloomDOS.Library.Polyketide.AlphaSearch import AlphaSearch
from Bloom.BloomDOS.Library.Polyketide.Polyketide import Polyketide
from Bloom.BloomDOS.Library.Sugar.Sugar import Sugar
from Bloom.BloomDOS.Library.Tailoring.Tailoring import Tailoring
from Bloom.BloomDOS.Library.Terpene.Terpene import Terpene
from Bloom.BloomDOS.Library.Type2PK.Type2PK import Type2PK


def add_enums_to_library(library):
    # Add variable_restriction enum
    for unit in library:
        if "variable_map" in unit:
            if len(list(unit["variable_map"].keys())) > 0:
                unit["enums"] = unit["enums"] + ["variable_restriction"]
    return library


def export_library(library: List[EnhancedSmartsDict], name: str):
    library = add_enums_to_library(library)
    output_fp = "{}/{}.json".format(rendered_lib_dir, name)
    json.dump(library, open(output_fp, "w"))


def render_library(
    render_curated: bool,
    render_aminoacid: bool,
    render_azole: bool,
    render_fattyacid: bool,
    render_polyketide: bool,
    render_sugar: bool,
    render_nucleoside: bool,
    render_tailoring: bool,
    render_primary_metabolite: bool,
    render_secondary_metabolite: bool,
    render_terpene: bool,
    render_type2pk: bool,
    render_all: bool,
):

    # create directory for rendered library
    os.makedirs(rendered_lib_dir, exist_ok=True)
    if render_curated or render_all:
        export_library(Miscellaneous().library, "miscellaneous")
        export_library(ModuleUnit().library, "curated_module")
    if render_aminoacid or render_all:
        export_library(AminoAcid().library, "aminoacid")
    if render_azole or render_all:
        export_library(Azole().library, "azole")
    if render_fattyacid or render_all:
        export_library(FattyAcid().library, "fattyacid")
    if render_polyketide or render_all:
        export_library(AlphaChains().library, "alphachain")
        export_library(AlphaSearch().library, "alphasearch")
        export_library(Polyketide().library, "polyketide")
    if render_sugar or render_all:
        export_library(Sugar().library, "sugar")
    if render_nucleoside or render_all:
        export_library(Nucleoside().library, "nucleoside")
    if render_tailoring or render_all:
        export_library(Tailoring().library, "tailoring")
    if render_terpene or render_all:
        export_library(Terpene().library, "terpene")
    if render_type2pk or render_all:
        export_library(Type2PK().library, "type2pk")
    if render_secondary_metabolite or render_all:
        export_library(SecondaryMetabolite().library, "secondary_metabolite")
    if render_primary_metabolite or render_all:
        export_library(PrimaryMetabolite().library, "primary_metabolite")


# script information
parser = argparse.ArgumentParser(
    description="Render SMARTS library to annotate molecules"
)
# variables to update libraries
parser.add_argument("-render_curated", action="store_true")
parser.add_argument("-render_aminoacid", action="store_true")
parser.add_argument("-render_azole", action="store_true")
parser.add_argument("-render_fattyacid", action="store_true")
parser.add_argument("-render_polyketide", action="store_true")
parser.add_argument("-render_sugar", action="store_true")
parser.add_argument("-render_nucleoside", action="store_true")
parser.add_argument("-render_tailoring", action="store_true")
parser.add_argument("-render_terpene", action="store_true")
parser.add_argument("-render_type2pk", action="store_true")
parser.add_argument("-render_primary_metabolite", action="store_true")
parser.add_argument("-render_secondary_metabolite", action="store_true")
parser.add_argument("-render_all", action="store_true")

# render library
render_library(**vars(parser.parse_args()))
