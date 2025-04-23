# bloom-publication
Biosynthetic Learning from Ontological Organizations of Metabolism (BLOOM)

## Installation

### Inference-Only Installation
1. Install the Package via Pip Symlinks:
    - Create and activate the Conda environment, then install the package in editable mode:
```
    conda env create -f bloom-environment.yml
    conda activate bloom
    pip install -e .
```
2. Prep DOS library for molecular substructure matching.
```
    python Installation.py -render_all
```

## Inference

### Biosynthetic Breakdown Prediction with BLOOM-DOS
BLOOM utilizes a custom SMARTS-based motif library and and post-filtering prioritization to infer combinations of biosynthetic building blocks that plausibly explain the biosynthesis of a given metabolite. Run the following command to generate a `.json` file that maps atom indices in the SMILES structure to candidate biosynthetic units. Set `standardization=True` to resolve SMILES inconsistencies such as tautomerization.

**Single Submission**
```python
from Bloom import BloomDOS

submission = {
    "submission_id": 1,
    "smiles": "CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC3OC(C)CC(C3O)N(C)C)C(C)(O)CC(C)C(=O)C(C)C(O)C1(C)O ",
}

BloomDOS.single_submission(
    submission=submission,
    standardize=True,
    output_dir="sample_output/bloom_dos_predictions",
)

```
**Multi-Parallel Submission (Distributed Across Multiple CPUs)**
```python
from Bloom import BloomDOS

submission_list = [
    {
        "submission_id": 1,
        "smiles": "CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC3OC(C)CC(C3O)N(C)C)C(C)(O)CC(C)C(=O)C(C)C(O)C1(C)O ",
    },
    {
        "submission_id": 2,
        "smiles": "CCC(CCCC/C=C/C=C/C(=O)NC(C(C)O)C(=O)NC1CC(CCNC(=O)/C=C\C(NC1=O)C)O)O",
    },
]

BloomDOS.multiprocess_subission(
    submission_list=submission_list,
    standardize=True,
    cpus=10,
    output_dir="sample_output/bloom_dos_predictions",
)
```
BLOOM-DOS includes a Streamlit-based web application that provides a visual interface for submitting molecules and exploring biosynthetic breakdown results. Run the following commands to deploy the app on a server:
```
cd WebApp
streamlit run BloomDOSApp.py
```

### Biosynthetic Molecular Embeddings with Bloom-Mol
BLOOM generates AI-based embeddings from BLOOM molecular graphs, where nodes represent atoms and mapped biosynthetic substructures. Embeddings are learned through self-supervised masked language modeling and enable rapid molecular comparison that reflects underlying biochemical ontologies. Embeddings are generated for both the complete molecular structure and individual biosynthetic units. Unit-level embeddings can be used to group chemically similar substructures by context and support downstream correlative analyses with gene groupings across paired genomic–molecular datasets, enabling the discovery of new biosynthetic rules.
```python
from Bloom import BloomEmbedder

BloomEmbedder.generate_molecular_embeddings(
    bloom_graph_fp="sample_output/bloom_dos_predictions/1.json",
    output_fp="sample_output/bloom_mol_embeddings/1.pkl",
    gpu_id=0,
)
```