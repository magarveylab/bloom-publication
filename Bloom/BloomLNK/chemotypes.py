from typing import List

import pandas as pd

from Bloom import dataset_dir

chemotype_conversion = {
    "BetaLactone": ["other"],
    "NRPS-IndependentSiderophore": ["NRPS-IndependentSiderophore"],
    "NonRibosomalPeptide": ["NonRibosomalPeptide"],
    "HomoserineLactone": ["other"],
    "Hybrid": ["NonRibosomalPeptide", "TypeIPolyketide"],
    "ArylPolyene": ["other"],
    "TypeIPolyketide": ["TypeIPolyketide"],
    "Cyclodipeptide": ["other"],
    "Ectoine": ["other"],
    "Aminoglycoside": ["Aminoglycoside"],
    "Nucleoside": ["Nucleoside"],
    "Butyrolactone": ["other"],
    "TypeIIPolyketide": ["TypeIIPolyketide"],
    "Phosphonate": ["other"],
    "Hapalindole": ["other"],
    "Terpene": ["Terpene"],
    "Polysaccharide": ["other"],
    "BetaLactam": ["BetaLactam"],
    "CyclicLactoneAutoinducer": ["other"],
    "RevResponseElementContaining": ["other"],
    "Resorcinol": ["other"],
    "Melanin": ["other"],
    "Phenazine": ["other"],
    "Antimetabolite": ["other"],
    "Ladderane": ["other"],
    "Glycolipid": ["other"],
    "RedoxCofactor": ["other"],
    "Bisindole": ["Alkaloid"],
    "NonAlphaPolyAminoAcid": ["other"],
    "NAcetylGlutaminylGlutamineAmide": ["other"],
    "Indole": ["other"],
    "AcylAminoAcids": ["other"],
    "Furan": ["other"],
    "Guanidinotides": ["other"],
    "FattyAcid": ["other"],
    "Alkaloid": ["Alkaloid"],
    "unassigned": ["unassigned"],
    "Phenyl": ["other"],
}


def sort_metabolites_by_chemotypes():
    metab_fp = f"{dataset_dir}/metabolites.csv"
    data = pd.read_csv(metab_fp).to_dict("records")
    sorted_metabolites = {}
    for r in data:
        metabolite_id = r["metabolite_id"]
        chemotype = r["chemotype"]
        for c in chemotype_conversion.get(chemotype, []):
            if c not in sorted_metabolites:
                sorted_metabolites[c] = []
            sorted_metabolites[c].append(metabolite_id)
    return sorted_metabolites


sorted_metabolites = sort_metabolites_by_chemotypes()


def normalize_bgc_chemotypes(bgcs: List[dict]):
    out = []
    for b in bgcs:
        cluster_id = b["cluster_id"]
        normalized_chemotypes = set()
        for c in b["chemotypes"]:
            normalized_chemotypes.update(chemotype_conversion.get(c, []))
        out.append(
            {
                "cluster_id": cluster_id,
                "chemotypes": list(normalized_chemotypes),
            }
        )
    return out
