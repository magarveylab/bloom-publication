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
2. Prep DOS library for molecular motif searching.
```
    python Installation.py -render_all
```