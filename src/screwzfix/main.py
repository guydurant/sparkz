import torch
from boltz.main import BoltzDiffusionParams, BoltzSteeringParams, download_boltz1, download_boltz2
from pathlib import Path
from dataclasses import asdict
import os
from boltz.data.write.writer import BoltzWriter
from screwzfix.screwzfix import ScrewzFix
from processing.complex import Complex
from utils.batch_generate import make_batch_from_sequence
import time
import hydra
from omegaconf import DictConfig

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:   
    device = torch.device("cpu")

predict_args = {
    "recycling_steps": 3,
    "sampling_steps": 200,
    "mx_parallel_samples": 1,
    "diffusion_samples": 1,
}

def infer(
    model,
    sequence,
    interacting_residues,
    fixed_atom_positions,
    fixed_indices,
    fixed_ligands_indices,
    fixed_with_ligands_indices,
    other_hetatms,
    modified_residues,
    side_chain_atom_mask_indices,
    ligand_info=None,
    out_dir=None,
    msa_dir=None,
    no_msa=False,
    use_constraints=False,
    docking_sphere_centre=None,
    cache_dir=None,
    use_scpot=False,
    guidance_type="flex",
    predict_args={
        "recycling_steps": 3,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "max_parallel_samples": 1,
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
        dock= False,  # docking_sphere_centre is not None,
    )
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:   
        device = torch.device("cpu")
    side_chain_atom_mask = torch.zeros(
        batch["ref_pos"].shape[:2], dtype=torch.long, device=batch["ref_pos"].device
    )
    side_chain_atom_mask[:, side_chain_atom_mask_indices] = 1
    batch["sidechain_atom_mask"] = side_chain_atom_mask
    batch["ligand_atom_mask"] = torch.zeros(
        batch["ref_pos"].shape[:2], dtype=torch.bool, device=batch["ref_pos"].device
    )
    batch["ligand_atom_mask"][:, fixed_ligands_indices] = True
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
    
    if docking_sphere_centre is not None:
        docking_sphere_centre = torch.tensor(
            docking_sphere_centre, dtype=batch["ref_pos"].dtype, device=device
        ).unsqueeze(0).unsqueeze(0)
        print("Setting docking sphere centre to", docking_sphere_centre, "shape", docking_sphere_centre.shape)
    
    with torch.no_grad():
        out = model(
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            },
            recycling_steps=predict_args.get("recycling_steps", 0),
            num_sampling_steps=predict_args.get("sampling_steps", 0),
            diffusion_samples=predict_args.get("diffusion_samples", 1),
            max_parallel_samples=predict_args.get("diffusion_samples", 1),
            run_confidence_sequentially=predict_args.get(
                "run_confidence_sequentially", True
            ),
            docking_sphere_centre= docking_sphere_centre,
            guidance_type=guidance_type,
        )
        out["confidence_score"] = (
            4 * out["complex_plddt"]
            + (
                out["iptm"]
                if not torch.allclose(out["iptm"], torch.zeros_like(out["iptm"]))
                else out["ptm"]
            )
        ) / 5
        out["exception"] = False
        out["masks"] = batch["atom_pad_mask"]
        out["coords"] = out["sample_atom_coords"]
    return out, batch


@hydra.main(config_path=os.path.join(os.path.dirname(__file__), "../../config/screwfix"), config_name="config")
def main(cfg: DictConfig):
    """Main entry point for the ScrewzFix application.

    Args:
        cfg (DictConfig): Configuration object containing all settings.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    steering_args = BoltzSteeringParams() 
    cache_dir = Path(cfg.cache).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_download = {
        "boltz1": download_boltz1,
        "boltz2": download_boltz2,
    }
    model_download[cfg.model](cache_dir)
    assert Path.exists(
        cache_dir / "boltz1_conf.ckpt"
    ), f"Model checkpoint not found in {cache_dir}"
    _torch_model = ScrewzFix.load_from_checkpoint(
        Path(f"{cache_dir}/boltz1_conf.ckpt"),
        strict=True,
        map_location="cpu",
        predict_args=predict_args,
        diffusion_process_args=asdict(BoltzDiffusionParams()),
        steering_args=asdict(steering_args),
        ema=False,
    )
    _torch_model = _torch_model.to(device)
    _torch_model.training = False
    _torch_model.structure_prediction_training = False
    torch.set_grad_enabled(True)
    torch.set_float32_matmul_precision("highest")
    _torch_model.eval()
    print(f"Predicting for {cfg.pid}")
    starttime = time.time()
    if not cfg.mode.use_ligandmpnn_scpack:
        complex_obj = Complex(cfg.protein_pdb_path, cfg.ligand_sdf_path, cfg.ccd, cache_dir=cache_dir)
        processed_pdb = complex_obj.process_for_guidance(sequence_buffer=10, guide_ligand= not cfg.mode.ligand_dock, guide_sc=cfg.mode.side_chain_flex)
    else:
        complex_obj = Complex.from_ligandmpnn_scpack(
            cfg.protein_pdb_path,
            cfg.ligand_sdf_path,
            cfg.ccd,
            cache_dir=cache_dir,
        )
        processed_pdb = complex_obj.process_for_guidance(
            sequence_buffer=10,
            guide_ligand=not cfg.mode.ligand_dock,
            guide_sc=True, 
        )
    
    with open(f"pdbblock_ref.pdb", "w") as f:
        f.write(processed_pdb["pdbblock_ref"])
    _torch_model.eval()
    predictions, batch = infer(
        _torch_model,
        processed_pdb["sequence"],
        processed_pdb["pocket_constraint_residue_indices"],
        processed_pdb["pocket_coords"],
        processed_pdb["whole_pocket_atom_indices"],
        processed_pdb["ligand_atom_indices"],
        processed_pdb["whole_pocket_and_ligand_atom_indices"],
        processed_pdb["other_hetatms"],
        processed_pdb["modified_residues"],
        processed_pdb["sidechain_atom_mask"],
        ligand_info=processed_pdb["smiles"],
        out_dir=".",
        cache_dir=cache_dir,
        msa_dir=None,
        no_msa=cfg.no_msa,
        use_constraints=cfg.use_constraints,
        docking_sphere_centre=processed_pdb["docking_sphere_centre"] if cfg.mode.ligand_dock else None,
        # docking_sphere_centre=None, # turn off docking sphere as a bit buggy
        guidance_type=cfg.mode.guidance_type,
    )
    time_taken = time.time() - starttime
    predictions["time_taken"] = time_taken
    boltz_writer = BoltzWriter(
        f"processed/structures",
        f"predictions",
        output_format="mmcif",
        boltz2=False,
    )
    boltz_writer.write_on_batch_end(
        None,
        None,
        predictions,
        0,
        batch,
        0, 
        0,
    )


if __name__ == "__main__":
    main()
