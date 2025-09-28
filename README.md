<h1 align="center">Sparkz & ScrewzFix</h1>


This repository contains two tools for predicting protein-ligand complexes faster with cofolding and utilising this faster co-folding for structure-based hallucation.

- **ScrewzFix** – a guidance-based method to speed up the inference of Boltz-1x that can run in different modes depending on how protein and ligand flexibility are treated. 
- **Sparkz** – a structure-prediction hallucination method for generating ligand-pocket sequences using ScrewzFix.

---

## Installation

Simply clone this repo and install using `uv` and `pip` inside the `sparkz` directory.

```bash
uv pip install -e .
```

## ScrewzFix

ScrewzFix provides three modes for inference:

- **`rigiddock`** – rigid docking: protein is treated as a rigid body (no flexibility) for docking.
- **`flexdock`** – flexible docking: both protein sidechains of the poket and ligand are treated with conformational flexibility.  
- **`flexscpack`** – flexible sidechain packing: explores protein sidechain conformations while keeping the backbone relatively fixed.  

To use LigandMPNN's SC packer as a prior for prediction for **`flexscpack`**, set `use_ligandmpnn_scpack` to `true`.


### Example usage

```bash
screwzfix \
    protein_pdb_path="$PROTEIN_PATH" \
    ligand_sdf_path="$LIGAND_PATH" \
    pid="$PDB" \
    ccd="$CCD" \
    out_dir="$OUTPUT_DIR/$PDB" \
    mode=flexdock
```

The `out_dir` follows the same structure as Boltz-1x, with the final `.cif` structure in the `predictions/input` folder.

## Sparkz

Utilises ScrewzFix to hallucinate ligand pocket sequences. Should be run using **`flexscpack`** mode. To increase speed of generation, `sampling_steps` and `recycling_steps` can be decreased.

```bash
sparkz \
    protein_pdb_path="$PROTEIN_PATH" \
    ligand_sdf_path="$LIGAND_PATH" \
    pid="$PDB" \
    ccd="$CCD" \
    num_iterations=50 \
    out_dir="${OUTPUT_DIR}/${PDB}" \
    recycling_steps=0 \
    sampling_steps=25 \
    mode=flexscpack \
```

The `out_dir` saves the final predicted and generated structure in the `final/predictions` folder with the processed data for the hallucation such as MSAs found in the `out_dir`.