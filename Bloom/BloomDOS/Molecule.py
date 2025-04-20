import os
from functools import partial
from multiprocessing import Pool
from typing import List, Optional, TypedDict, Union

from rdkit import Chem
from tqdm import tqdm

from Bloom.BloomDOS.Annotation.AlphaAnnotation import AlphaAnnotation
from Bloom.BloomDOS.Annotation.EnhancedSmarts import EnhancedSmarts
from Bloom.BloomDOS.Annotation.UnitAnnotation import UnitAnnotation
from Bloom.BloomDOS.Chemistry.Standardization import MoleculeStandardizer
from Bloom.BloomDOS.Dereplication.NonModularDereplication import (
    NonModularDereplication,
)
from Bloom.BloomDOS.Dereplication.PeptideDereplication import (
    PeptideDereplication,
)
from Bloom.BloomDOS.Dereplication.PolyketideDereplication import (
    PolyketideDereplication,
)
from Bloom.BloomDOS.Dereplication.StandardDereplication import (
    StandardDereplication,
)
from Bloom.BloomDOS.Dereplication.StereochemistryDereplication import (
    StereochemistryDereplication,
)
from Bloom.BloomDOS.Graph.AnnotationGraph import AnnotationGraph
from Bloom.BloomDOS.Graph.MolecularGraph import MolecularGraph


class Molecule:

    def __init__(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        [x.SetAtomMapNum(0) for x in mol.GetAtoms()]
        self.smiles = Chem.MolToSmiles(mol)
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.graphs = []

    @property
    def solution_count(self):
        return len(self.graphs)

    def standardize(
        self,
        fast_standardize: bool = True,
        standardize_tautomers: bool = False,
    ):
        # standardize small molecule
        standardizer = MoleculeStandardizer(
            fast_standardize=fast_standardize,
            standardize_tautomers=standardize_tautomers,
        )
        self.smiles = standardizer.standardize_smiles(smiles=self.smiles)
        self.mol = Chem.MolFromSmiles(self.smiles)

    def predict_biosynthesis(
        self,
        skip_enums: List[str] = [],
        only_consider_enums: List[str] = [],
        additional_units: List[dict] = [],
        silent_output: bool = False,
        log_dereplication: bool = False,
    ):
        self.silent_output = silent_output
        self.log_dereplication = log_dereplication
        # Format additional units
        additional_units = [
            EnhancedSmarts._load_from_dict(row) for row in additional_units
        ]
        [x.enums.add("additional") for x in additional_units]
        additional_library = {"additional": additional_units}
        # graph to store all the annotations
        self.G = AnnotationGraph(self.mol, silent_output=self.silent_output)
        # alpha carbon detection - need it for directionality in mapping polyketide units
        alpha_annotation = AlphaAnnotation(
            self.mol, silent_output=silent_output
        )
        alpha_annotation.find_non_alpha_regions()
        alpha_annotation.find_potential_alpha_sites()
        alpha_annotation.map_alpha_chains()
        alpha_annotation.dereplicate()
        alpha_annotation.get_best_annotation()
        self.alpha_carbons = alpha_annotation.alpha_carbons
        # unit detection
        unit_annotation = UnitAnnotation(
            self.mol,
            self.G,
            self.alpha_carbons,
            skip_enums,
            only_consider_enums,
            additional_library,
            silent_output=silent_output,
        )
        unit_annotation.limit_unit_library()
        unit_annotation.find_units()
        self.logger("Annotated with Units")
        # dereplication pipelines
        standard_dereplication = StandardDereplication(self.G)
        polyketide_dereplication = PolyketideDereplication(self.G)
        peptide_dereplication = PeptideDereplication(self.G)
        nonmodular_dereplication = NonModularDereplication(self.G)
        stereo_dereplication = StereochemistryDereplication(self.G)
        # to minimze nonoverlap graph calculation use confident annotations
        # to remove less confident annotations (such as pk)
        stereo_dereplication.remove_sugars()
        self.logger("Removed Sugars")
        standard_dereplication.remove_by_confident_annotations()
        self.logger("Removed by confident annotations")
        # remove flexible annotations if more specific one hits
        standard_dereplication.remove_by_flexibility()
        self.logger("Removed by flexibility")
        # polyketide dereplication
        polyketide_dereplication.remove_by_alpha_carbon()
        self.logger("Removed PK units by alpha carbon")
        polyketide_dereplication.remove_by_dh_shifts()
        self.logger("Removed invalid DH shifts")
        polyketide_dereplication.remove_by_unit_type()
        self.logger("Removed PK units by unit type")
        # peptide dereplication
        peptide_dereplication.remove_by_azoles()
        self.logger("Removed by azole rules")
        # some annotations are priortized over other
        standard_dereplication.remove_by_priortization()
        self.logger("Removed by unit priortization")
        # non modular dereplication
        nonmodular_dereplication.remove_nonmodular_by_exact_coverage()
        self.logger("Removed non modular units by exact coverage")
        nonmodular_dereplication.remove_nonmodular_by_combinatoric_coverage()
        self.logger("Removed non modular units by combinatoric coverage")
        # remove redundant annotations
        standard_dereplication.remove_exact_annotations_by_rings()
        standard_dereplication.remove_exact_annotations()
        self.logger("Removed non modular units by exact coverage")
        # keep annotations that annotate more atoms
        standard_dereplication.remove_by_coverage()
        self.logger("Removed units by coverage")
        # find large continuous stretches to use for unit dereplication
        polyketide_dereplication.remove_by_polyether_stretches()
        self.logger("Removed units by polyether stretches")
        polyketide_dereplication.remove_by_polyketide_stretches()
        self.logger("Removed units by polyketide stretches")
        # find optimal solution based on atom coverage
        unit_annotation.get_best_annotated_coverage()
        # certain fragments such as tailoring cannot be adjacent to unknown regions
        unit_annotation.apply_unknown_restrictions()
        # dereplicate solutions
        unit_annotation.dereplicate_solutions()
        for final_annotations in unit_annotation.solutions:
            molG = MolecularGraph.load_graph_from_annotations(
                self.mol, self.G, final_annotations
            )
            self.graphs.append(molG)

    def logger(self, message):
        if self.log_dereplication:
            print(message)
            print(self.G.annotation_frequency)
            print("-------")

    def visualize_biosynthesis(
        self, index: int = 0, tmp_dir: str = "image_cache"
    ):
        if index <= self.solution_count - 1:
            return self.graphs[index].visualize_by_units(tmp_dir)

    def export_graph(self, filepath: str, index: int = 0):
        if index <= self.solution_count - 1:
            self.graphs[index].export(filepath)

    def tabulate_annotations(self, index: int = 0):
        if index <= self.solution_count - 1:
            return self.graphs[index].node_table


####################################################################
# Multiprocessing
####################################################################


class SmilesInput(TypedDict):
    smiles_id: Union[int, str]
    smiles: str


def single_submission_from_smiles(
    smiles: SmilesInput,
    standardize: bool = False,
    working_dir: Optional[str] = None,
):
    smiles_id = smiles["smiles_id"]
    output_fp = f"{working_dir}/{smiles_id}.json"
    if os.path.exists(output_fp) == False:
        m = Molecule(smiles["smiles"])
        if standardize:
            m.standardize()
        m.predict_biosynthesis()
        if working_dir != None:
            m.export_graph(output_fp)


def multiprocess_subission_from_smiles(
    smiles_list: List[SmilesInput],
    working_dir: str,
    cpus: int = 3,
    standardize: bool = False,
):
    os.makedirs(working_dir, exist_ok=True)
    funct = partial(
        single_submission_from_smiles,
        standardize=standardize,
        working_dir=working_dir,
    )
    pool = Pool(cpus)
    process = pool.imap_unordered(funct, smiles_list)
    [p for p in tqdm(process, total=len(smiles_list))]
