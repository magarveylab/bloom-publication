# bloom-publication
Biosynthetic Learning from Ontological Organizations of Metabolism (BLOOM)

## Installation

### Inference-Only Installation
1. Install the package via pip Symlinks:
    - Create and activate the Conda environment, then install the package in editable mode. **Note** env is currently only compatible with cuda:11 for embedding calculation.
```
    conda env create -f envs/bloom.yml
    conda activate bloom
    pip install -e .
```
2. Prep Bloom-DOS library for molecular substructure matching.
```
    python prepare_dos_library.py -render_all
```
3. Enabling BLOOM-LNK predictions: 
    - Download and extract the pre-generated data files (`sm_dags.zip`, `molecular_jaccard_signature_library.pkl` and `sm_graphs.zip`) from the accompanying [Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.15226521). Place the extracted contents in this [directory](https://github.com/magarveylab/bloom-publication/tree/main/Bloom/datasets).
    - Alternatively, you can generate a custom database by processing a directory of BLOOM-DOS outputs. Ensure that the [reference table](https://github.com/magarveylab/bloom-publication/blob/main/Bloom/datasets/metabolites.csv) is updated so that `metabolite_id` values align with those in your dataset.
```python
from Bloom import BloomLNK

BloomLNK.build_sm_dags(
    bloom_dos_pred_dir="sample_output/bloom_dos_predictions/",
    output_dir="sample_output/custom_database/",
)

BloomLNK.build_sm_graphs(
    bloom_dos_pred_dir="sample_output/bloom_dos_predictions/",
    output_dir="sample_output/custom_database/",
)

BloomLNK.build_molecular_jacccard_signature_library(
    bloom_dos_pred_dir="sample_output/bloom_dos_predictions/",
    output_dir="sample_output/custom_database/",
)
```

4. Set Up Qdrant
    - Install Qdrant and restore the Qdrant reference databases from the provided snapshots. Look under **Qdrant Setup** for more details.

## Supplementary Package Installation for Genomic and Molecular Analysis

The following packages were used to support various analysis including strain prioritization, computation of the polyketide molecular universe, and large-scale analysis of encoded metabolite landscapes. Outputs are integrated with metabolite-level analyses for comprehensive multi-omic interpretation.

| Package | Description | Publication Link
|---|---| ---|
| [IBIS](https://github.com/magarveylab/ibis-publication/tree/main)          | Integrated Biosynthetic Inference Suite (IBIS) - AI-based platform for high-throughput identification and comparison of bacterial metabolism from genomic data   | [Here](https://www.pnas.org/doi/10.1073/pnas.2425048122) | 

## Qdrant Setup 
BLOOM-RXN utilize [Qdrant](https://qdrant.tech/) embedding databases for approximate nearest neighbor (ANN) lookups. Since system configurations may vary, we recommend setting up Qdrant locally using a Docker container, following the [official documentation](https://qdrant.tech/documentation/quickstart/).

The required Qdrant databases for inference are provided as QdrantSnapshots.zip in the accompanying [Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.15226521).

### Restoring Qdrant Databases
To restore the Qdrant databases, ensure that the snapshot files (unzipped) are placed [here](https://github.com/magarveylab/bloom-publication/tree/main/Bloom/QdrantSnapshots) and run the following command.
```
conda activate bloom
python restore_qdrant.py
```

## Graphormer Training
Training scripts for all models can be found [here](https://github.com/magarveylab/bloom-graphormer-training).


## Inference

### Biosynthetic Breakdown Prediction with BLOOM-DOS
BLOOM utilizes a custom SMARTS-based motif library and and post-filtering prioritization to infer combinations of biosynthetic building blocks that plausibly explain the biosynthesis of a given metabolite. Run the following command to generate a `.json` file that maps atom indices in the SMILES structure to candidate biosynthetic units. Set `standardization=True` to resolve SMILES inconsistencies such as tautomerization.

**Single Submission**
```python
from Bloom import BloomDOS

submission = {
    "metabolite_id": 1,
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
        "metabolite_id": 1,
        "smiles": "CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC3OC(C)CC(C3O)N(C)C)C(C)(O)CC(C)C(=O)C(C)C(O)C1(C)O ",
    },
    {
        "metabolite_id": 2,
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
Using BLOOM-DOS Annotations to Predict Polyketide Classes
```python
from Bloom.BloomGraph.BloomGraph import BloomGraph

G = BloomGraph.load_from_json(
    json_fp="sample_output/bloom_dos_predictions/1.json"
)
print(G.polyketide_subclasses)
```

### Biosynthetic Molecular Embeddings with BLOOM-Mol
BLOOM generates AI-based embeddings from BLOOM molecular graphs, where nodes represent atoms and mapped biosynthetic substructures. Embeddings are learned through self-supervised masked language modeling and enable rapid molecular comparison that reflects underlying biochemical ontologies. Embeddings are generated for both the complete molecular structure and individual biosynthetic units. Unit-level embeddings can be used to group chemically similar substructures by context and support downstream correlative analyses with gene groupings across paired genomic–molecular datasets, enabling the discovery of new biosynthetic rules.
```python
from Bloom import BloomEmbedder

BloomEmbedder.generate_molecular_embeddings(
    bloom_graph_fp="sample_output/bloom_dos_predictions/1.json",
    output_fp="sample_output/bloom_mol_embeddings/1.pkl",
    gpu_id=0,
)
```

### Reaction Embeddings with BLOOM-RXN
BLOOM generates AI-based embeddings for chemical reaction that capture features reflective of the Enzyme Commission (EC) hierarchy. These embeddings are integrated into the BLOOM knowledge graph and serve as a core input to BLOOM-LNK for BGC–metabolite association.

```python
from Bloom import BloomRXN

BloomRXN.generate_embeddings_from_csv(
    csv_fp="sample_data/sample_rxns.csv",
    output_fp="sample_output/example_rxn_embeddings.pkl",
)
```
ANN Classification to infer enzyme classes for reactions lacking annotation
```python
from Bloom import BloomRXN

BloomRXN.generate_ec_from_csv(
    csv_fp="sample_data/sample_rxns.csv",
    output_fp="sample_output/example_rxn_ec_classification.csv",
)
```

### Predicting BGC-Molecule Associations with BLOOM-LNK
BLOOM associates BGCs and metabolites using AI-powered knowledge graph reasoning. The following function compares BGCs from an IBIS result to a molecular database, performs chemotype filtering, and returns the top n metabolites that meet the defined cutoff. 

For scalability:
1. Use `only_consider_metabolite_ids` to restrict the search to a specific subset of metabolites.
2. Use `top_n_jaccard` to pre-filter candidates based on Jaccard similarity before applying Graphormer inference.
3. Apply quality filters using `min_orf_count` for non-modular chemotypes and `min_module_count` for modular chemotypes (e.g., Type I PKS, NRPS) to exclude incomplete BGCs.

```python
from Bloom import BloomLNK 

BloomLNK.get_bgc_mol_associations(
    ibis_dir="sample_data/example_ibis_output/",
    output_fp="sample_output/example_bgc_mol_associations.csv",
    min_orf_count=4,
    min_module_count=4,
    top_n_jaccard=1000,
    report_top_n=10,
    only_consider_metabolite_ids=None, # default all metabolites
)
```
### Predicting BGC-Molecule Associations with BLOOM-MOL and BLOOM-BGC
BLOOM predicts gene–unit associations by identifying correlations between embedding-derived groupings.

BLOOM-BGC generates embeddings for open reading frames (ORFs) based on domain architecture and sequence context. These embeddings are learned through self-supervised masked language modeling. 
```python
from Bloom import BloomEmbedder

BloomEmbedder.generate_gene_embeddings(
    ibis_dir="sample_data/example_ibis_output/",
    output_fp="sample_output/example_bgc_embeddings.pkl",
    gpu_id=0,
)
```
#### Density-based clustering of ORF and Unit embeddings.

**Note**: This step uses a separate conda environment (bloom-cluster-cuda11.yml) due to package incompatibilities with the main **CUDA** environment.
```python
from Bloom import BloomEmbedder

# compute orf groupings
BloomEmbedder.compute_clustering_from_matrix(
    matrix_fp="sample_data/example_orf_unit_matrix/orf_embeddings.npy",
    matrix_keys_fp="sample_data/example_orf_unit_matrix/orf_embeddings_keys.csv",
    output_fp="sample_output/unit_orf_groups/orfs.csv",
)

# compute unit groupings
BloomEmbedder.compute_clustering_from_matrix(
    matrix_fp="sample_data/example_orf_unit_matrix/unit_embeddings.npy",
    matrix_keys_fp="sample_data/example_orf_unit_matrix/unit_embeddings_keys.csv",
    output_fp="sample_output/unit_orf_groups/units.csv",
)
```
Statistical association is evaluated using the chi-squared (χ²) test.
```python 
from Bloom import BloomEmbedder

BloomEmbedder.compute_unit_orf_associations(
    units_cluster_fp="sample_output/unit_orf_groups/units.csv",
    orfs_clustering_fp="sample_output/unit_orf_groups/orfs.csv",
    output_fp="sample_output/unit_orf_associations.csv",
)
```