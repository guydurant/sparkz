import pandas as pd
import torch
from boltz.main import BoltzDiffusionParams, BoltzSteeringParams
from pathlib import Path
from dataclasses import asdict
import click
import os
from boltz.model.model import Boltz1
from inpainting.screwz import Screwz1
from tqdm import tqdm
from boltz.main import process_inputs  # , BoltzProcessedInput, download
from boltz.data.types import Manifest
from boltz.data.module.inference import BoltzInferenceDataModule
from processing.complex import Complex
from utils.writer import write_complex
from utils.batch_generate import make_batch_from_sequence
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

predict_args = {
    "recycling_steps": 3,
    "sampling_steps": 200,
    "diffusion_samples": 10,
}

def process_csv_file(csv_file):
    """
    Process the CSV file to extract ligand, protein, PID, and CCD information.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        Tuple[List[str], List[str], List[str], List[str]]: Lists of ligands, proteins, PIDs, and CCDs.
    """
    df = pd.read_csv(csv_file)
    ligands = df["ligand"].tolist()
    proteins = df["protein"].tolist()
    pids = df["pid"].tolist()
    ccds = df["ccd"].tolist()
    return ligands, proteins, pids, ccds

def infer(
    model,
    sequence,
    interacting_residues,
    fixed_atom_positions,
    fixed_indices,
    fixed_with_ligands_indices,
    other_hetatms,
    modified_residues,
    ligand_info=None,
    out_dir=None,
    msa_dir=None,
    no_msa=False,
    use_constraints=False,
    cache_dir=None,
    predict_args={
        "recycling_steps": 3,
        "sampling_steps": 200,
        "diffusion_samples": 10,
    }
):
    """
    Infer the structure of a protein-ligand complex.

    Args:
        model (LightningModule): The model to use for inference.
        sequence (str): The protein sequence (one-hot encoded).
        interacting_residues (List[int]): The residues that interact with the ligand.
        fixed_atom_positions (np.ndarray): The fixed atom positions of the protein.
        ligand_info (str): Ligand information from the YAML file.
        num_iterations (int): The number of iterations to run.
        lr (float): The learning rate.
        datetime_ (str): The datetime string for the output file.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch
        .Tensor]]: The batch and the predictions.
    """

    batch = make_batch_from_sequence(
        sequence,
        ligand_info,
        other_hetatms,
        modified_residues,
        out_dir=out_dir,
        msa_dir=msa_dir,
        positions=interacting_residues,
        no_msa=no_msa,
        use_constraints=use_constraints,
        cache_dir=cache_dir,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if fixed_atom_positions is not None and fixed_indices is not None:
        batch["ref_pos"][..., fixed_with_ligands_indices, :] = torch.tensor(
            fixed_atom_positions,
            dtype=batch["ref_pos"].dtype,
            device=batch["ref_pos"].device,
        )[
            ..., fixed_with_ligands_indices, :
        ]  # make ref pos only residues guiding to
        inpainting_mask = torch.zeros(
            batch["ref_pos"].shape[:2], dtype=torch.long, device=device
        )
        inpainting_mask[:, fixed_indices] = 1
        batch["inpainting_mask"] = inpainting_mask.unsqueeze(-1)
        inpainting_mask_with_ligands = torch.zeros(
            batch["ref_pos"].shape[:2], dtype=torch.long, device=device
        )
        inpainting_mask_with_ligands[:, fixed_with_ligands_indices] = 1
        batch["inpainting_mask_with_ligands"] = inpainting_mask_with_ligands.unsqueeze(
            -1
        )
    with torch.no_grad():
        out = model(
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            },
            recycling_steps=predict_args.get("recycling_steps", 0),
            num_sampling_steps=predict_args.get("sampling_steps", 0),
            diffusion_samples=predict_args.get("diffusion_samples", 1),
            run_confidence_sequentially=predict_args.get(
                "run_confidence_sequentially", True
            ),
        )
        out["confidence_score"] = (
            4 * out["complex_plddt"]
            + (
                out["iptm"]
                if not torch.allclose(out["iptm"], torch.zeros_like(out["iptm"]))
                else out["ptm"]
            )
        ) / 5
    return out, batch


@click.command()
@click.option("--protein_pdb_path", default=None, help="Path to input protein PDB file")
@click.option("--ligand_sdf_path", default=None, help="Path to input ligand SDF file")
@click.option(
    "--pid",
    default=None,
    help="Unique Protein ID (e.g. 1a2b_1 or 1a2b_2) for the protein input",
)
@click.option(
    "--ccd", default=None, help="Unique CCD code for the ligand input (e.g. ZRY or DMS)"
)
@click.option("--out_dir", type=str, help="Output directory", required=True)
@click.option(
    "--mode",
    help="Mode to run the model in",
    type=click.Choice(["inpainting", "default"]),
    required=True,
)
@click.option(
    "--guidance_rate",
    type=float,
    help="Guidance rate for the guided mode. Normally run as 1.0 so it is effectively inpainting",
    default=1.0,
)
@click.option("--no_msa", type=bool, help="Disable MSA", default=False)
@click.option(
    "--use_constraints", type=bool, help="Use constraints for docking", default=False
)
@click.option("--overwrite", type=bool, help="Overwrite existing files", default=False)
@click.option("--cache", type=str, help="Cache directory", default="/homes/durant/")
def main(protein_pdb_path, ligand_sdf_path, pid, ccd, out_dir, mode, guidance_rate, no_msa, use_constraints, overwrite, cache):
    model_names = {
        "default": Boltz1,
        "inpainting": Screwz1,
    }
    
    print("Loading model for mode", mode)
    steering_args = BoltzSteeringParams()
    cache_dir = Path(cache).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _torch_model = model_names[mode].load_from_checkpoint(
        Path(f"{cache_dir}/boltz1_conf.ckpt"),
        strict=True,
        map_location="cpu",
        predict_args=predict_args,
        diffusion_process_args=asdict(BoltzDiffusionParams()),
        steering_args=asdict(steering_args),
        ema=False,
    )
    if mode == "inpainting":
        _torch_model.structure_module.set_guidance(guidance_rate, True) 
        mode = f"{mode}_{guidance_rate}"
    _torch_model = _torch_model.to("cuda" if torch.cuda.is_available() else "cpu")
    _torch_model.training = False
    _torch_model.structure_prediction_training = False
    torch.set_grad_enabled(True)
    torch.set_float32_matmul_precision("highest")
    _torch_model.eval()
    # if (
    #     os.path.exists(f"{out_dir}/{pid}/predictions/input_model_0.pdb")
    #     and not overwrite
    # ):
    #     print(f"Skipping {pid} as it already exists")
    #     return
    print(f"Predicting for {pid}")
    if not os.path.exists(f"{out_dir}/{pid}"):
        os.makedirs(f"{out_dir}/{pid}")
    starttime = time.time()
    # complex_obj = Complex(protein, ligand, pid.split("_")[1])
    complex_obj = Complex(protein_pdb_path, ligand_sdf_path, ccd)
    processed_pdb = complex_obj.process_for_guidance(sequence_buffer=10)
    with open(f"{out_dir}/{pid}/pdbblock_ref.pdb", "w") as f:
        f.write(processed_pdb["pdbblock_ref"])
    _torch_model.eval()
    predictions, batch = infer(
        _torch_model,
        processed_pdb["sequence"],
        processed_pdb["pocket_constraint_residue_indices"],
        processed_pdb["pocket_coords"],
        processed_pdb["whole_pocket_atom_indices"],
        processed_pdb["whole_pocket_and_ligand_atom_indices"],
        processed_pdb["other_hetatms"],
        processed_pdb["modified_residues"],
        ligand_info=ccd,
        out_dir=f"{out_dir}/{pid}",
        cache_dir=cache_dir,
        msa_dir=None,
        no_msa=no_msa,
        use_constraints=use_constraints,
    )
    time_taken = time.time() - starttime
    predictions["time_taken"] = time_taken
    write_complex(
        predictions,
        Path(f"{out_dir}/{pid}/processed/structures"),
        Path(f"{out_dir}/{pid}/predictions"),
        batch,
        output_format="pdb",
    )


if __name__ == "__main__":
    main()
