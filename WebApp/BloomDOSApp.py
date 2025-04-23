import logging
import os

import streamlit as st
from rdkit import Chem

from Bloom.BloomDOS.Molecule import Molecule

logging.getLogger().setLevel(logging.WARNING)


class BloomDOSApp:

    def __init__(self, tmp_dir="./tmp"):
        self.tmp_dir = tmp_dir
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        self.launch_app()

    def launch_app(self):

        self.header = st.header("Biosynthetic Breakdown with BloomDOS")
        self.smiles_input = st.text_input("Input SMILES:", "")
        self.clear_cache = st.sidebar.button("clear cache")

        self.skip_enums = set()
        # Initialize checkboxes for library enums
        self.enums_header = st.sidebar.subheader(
            "Unit types to avoid in analysis:"
        )
        enums = {
            "Amino Acids": "aa",
            "Azole": "azole",
            "Polyketide": "pk",
            "Fatty Acids": "fattyacid",
            "Nucleosides": "nucleoside",
            "Sugars": "sugar",
            "Terpene": "terpene",
            "Curated": "miscellaneous",
            "Tailoring": "tailoring",
            "Primary": "primary_metabolite",
            "Secondary": "secondary_metabolite",
            "Flexible": "flexible",
        }
        for enum in enums:
            self.check = st.sidebar.checkbox(enum)
            if self.check:
                self.skip_enums.add(enums[enum])

        if validate_smiles(self.smiles_input):

            print(self.smiles_input)

            retro = Molecule(self.smiles_input)
            retro.standardize()
            retro.predict_biosynthesis(
                skip_enums=self.skip_enums, silent_output=True
            )

            img = retro.visualize_biosynthesis(index=0, tmp_dir=self.tmp_dir)

            self.img_display = st.image(img, use_container_width="auto")


def validate_smiles(smiles_input):
    if smiles_input in ["", None]:
        return False
    elif Chem.MolFromSmiles(smiles_input) == None:
        return False
    else:
        return True


app = BloomDOSApp()
