import ast
import os

import pandas as pd
from rdkit import Chem

from Bloom.BloomDOS.Chemistry.Reaction import ReactionSmarts
from Bloom.BloomDOS.GeneralUtils import curdir

# define dir where substrates/reactions can be found
pk_dir = os.path.join(curdir, "Library/Polyketide/Tables/")

################################################################################
# Class to create polyketide modules
################################################################################


class PolyketideModule:

    def __init__(self):
        # polyketide modules with uniquify argument
        # these are instances of symmetrical motif
        self.symmetrical_motifs = ["Mal-DH"]

    def load_starters(self):
        # load starter units for biosynthesis
        starters_fp = os.path.join(pk_dir, "starters.csv")
        self.starters = pd.read_csv(starters_fp).to_dict("records")
        for row in self.starters:
            row["enums"] = {"pk", "starter"}

    def load_substrates(self):
        # load extender units for biosynthesis
        substrate_fp = os.path.join(pk_dir, "substrates.csv")
        self.substrates = {}
        for row in pd.read_csv(substrate_fp).to_dict("records"):
            # cast enums
            row["enums"] = set(ast.literal_eval(row["enums"]))
            self.substrates[row["notation"]] = row

    def load_module_reactions(self):
        # load module reactions for biosynthesis
        mod_rxn_fp = os.path.join(pk_dir, "module_reactions.csv")
        all_units = set(self.substrates)
        self.mod_rxns = []
        for row in pd.read_csv(mod_rxn_fp).to_dict("records"):
            # cast enums
            row["enums"] = set(ast.literal_eval(row["enums"]))
            # determine substrates for reactions
            row["ignore_units"] = set(ast.literal_eval(row["ignore_units"]))
            row["only_consider_units"] = set(
                ast.literal_eval(row["only_consider_units"])
            )
            if len(row["only_consider_units"]) > 0:
                row["consider_substrates"] = row["only_consider_units"]
            else:
                row["consider_substrates"] = all_units - row["ignore_units"]
            # load reaction
            row["reaction_smarts"] = ReactionSmarts(row["reaction_smarts"])
            self.mod_rxns.append(row)

    def compute_module_monomers(self):
        # find all combinations between module reactions and substratese
        self.module_monomers = []
        for rxn in self.mod_rxns:
            for s in rxn["consider_substrates"]:
                # notation
                description = "{}-{}".format(
                    self.substrates[s]["description"], rxn["reaction_name"]
                )
                notation = "{}-{}".format(s, rxn["reaction_name"])
                # do i avoid dereplication of this unit
                avoid_dereplication = self.substrates[s]["avoid_dereplication"]
                # depriortize units such as OHMeMal
                priortization = self.substrates[s]["priortization"]
                # compile enums
                enums = rxn["enums"] | self.substrates[s]["enums"]
                # compute new smiles
                rxn_smarts = rxn["reaction_smarts"]
                smiles = self.substrates[s]["smiles"]
                product = rxn_smarts.RunReactant(smiles)[0]
                # map all the atoms to the same module
                mol = Chem.MolFromSmiles(product)
                [a.SetAtomMapNum(1) for a in mol.GetAtoms()]
                product = Chem.MolToSmiles(mol)
                # uniquify
                if notation in self.symmetrical_motifs:
                    uniquify = False
                elif rxn["uniquify"] == False:
                    uniquify = False
                elif self.substrates[s]["uniquify"] == False:
                    uniquify = False
                else:
                    uniquify = True
                # cache
                self.module_monomers.append(
                    {
                        "description": description,
                        "reaction_id": rxn["reaction_id"],
                        "substrate": s,
                        "notation": notation,
                        "smiles": product,
                        "avoid_dereplication": avoid_dereplication,
                        "priortization": priortization,
                        "substrate_id": self.substrates[s]["substrate_id"],
                        "substrate_family": self.substrates[s][
                            "substrate_family"
                        ],
                        "domain_id": rxn["domain_id"],
                        "domain_subclass_id": rxn["domain_subclass_id"],
                        "enums": enums,
                        "uniquify": uniquify,
                    }
                )

    def load_terminal_reactions(self):
        # load terminal reactions for biosynthesis
        term_rxn_fp = os.path.join(pk_dir, "terminal_reactions.csv")
        self.terminal_rxns = []
        for row in pd.read_csv(term_rxn_fp).to_dict("records"):
            row["reaction_smarts"] = ReactionSmarts(row["reaction_smarts"])
            self.terminal_rxns.append(row)

    def load_specialized_motifs(self):
        self.specialized_motifs = {}
        # reactions to create specialized motifs
        spec_rxn_fp = "{}/specialized_reactions.csv".format(pk_dir)
        for row in pd.read_csv(spec_rxn_fp).to_dict("records"):
            rid = row["reaction_id"]
            row["enums"] = set(ast.literal_eval(row["enums"])) | {
                "pk_specialized"
            }
            row["reaction_smarts"] = ReactionSmarts(row["reaction_smarts"])
            row["predecessor_reaction_ids"] = ast.literal_eval(
                row["predecessor_reaction_ids"]
            )
            row["predecessor_substrate_ids"] = ast.literal_eval(
                row["predecessor_substrate_ids"]
            )
            row["ignore_substrate_ids"] = ast.literal_eval(
                row["ignore_substrate_ids"]
            )
            row["linkers"] = []
            row["bonds"] = []
            row["atoms_to_keep"] = None
            row["new_alpha_atoms"] = None
            self.specialized_motifs[rid] = row
        # load linker descriptors - determines the adjacent units when mapping
        spec_linker_fp = "{}/specialized_motif_linker_descriptors.csv".format(
            pk_dir
        )
        for row in pd.read_csv(spec_linker_fp).to_dict("records"):
            rid = row["reaction_id"]
            row["linker_smarts"] = Chem.MolFromSmarts(row["linker_smarts"])
            row["linker_atoms"] = [
                int(a) for a in ast.literal_eval(row["linker_atoms"])
            ]
            row["linker_enums"] = ast.literal_eval(row["linker_enums"])
            self.specialized_motifs[rid]["linkers"].append(row)
            # sometimes there is a version two of the reaction
            # for hydrogens in double bond reaction
            other_versions = ast.literal_eval(row["other_versions"])
            for v in other_versions:
                self.specialized_motifs[v]["linkers"].append(row)
        # load bond descriptors - identify aromatic bonds for SMARTS
        spec_bond_fp = "{}/specialized_motif_bond_descriptors.csv".format(
            pk_dir
        )
        for row in pd.read_csv(spec_bond_fp).to_dict("records"):
            rid = row["reaction_id"]
            row["smarts"] = Chem.MolFromSmarts(row["smarts"])
            row["bonds"] = [
                [int(x) for x in b.split("-")]
                for b in ast.literal_eval(row["bonds"])
            ]
            self.specialized_motifs[rid]["bonds"].append(row)
            other_versions = ast.literal_eval(row["other_versions"])
            for v in other_versions:
                self.specialized_motifs[v]["bonds"].append(row)
        # cast as list
        self.specialized_motifs = list(self.specialized_motifs.values())

    def load_standalone_motifs(self):
        self.standalone_motifs = []
        # curated motifs
        stand_fp = "{}/standalone_units.csv".format(pk_dir)
        for row in pd.read_csv(stand_fp).to_dict("records"):
            row["enums"] = ast.literal_eval(row["enums"])
            row["linker_atoms"] = ast.literal_eval(row["linker_atoms"])
            row["linker_atom_enums"] = [
                ast.literal_eval(i)
                for i in row["linker_atom_enums"].split(" | ")
            ]
            self.standalone_motifs.append(row)
