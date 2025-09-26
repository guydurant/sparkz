# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: /vols/opig/projects/guy-affinity/POCKET_DESIGN/sparkz/src/cli.py
# Bytecode version: 3.12.0rc2 (3531)
# Source timestamp: 2025-04-30 11:11:46 UTC (1746011506)

# import wandb
import torch
from boltz.main import BoltzDiffusionParams, BoltzSteeringParams, download_boltz1, download_boltz2
from pathlib import Path
from screwfix.screwz import Screwz1
from dataclasses import asdict
from hallucination.optimise import optimise_pocket_sequence_bindcraft
from screwfix.main import infer
from utils.writer import write_complex
import json
import click
import time
import os
from processing.complex import Complex
from utils.batch_generate import make_batch_from_sequence
import hydra
from boltz.data.write.writer import BoltzWriter
from omegaconf import DictConfig

predict_args = {
    "recycling_steps": 3,
    "sampling_steps": 200,
    "mx_parallel_samples": 1,
    "diffusion_samples": 1,
}


@hydra.main(config_path=os.path.join(os.path.dirname(__file__), "../config/sparkz"), config_name="config")
def main(cfg: DictConfig):
    print('Loading model')
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
    _torch_model = Screwz1.load_from_checkpoint(
        Path(f"{cache_dir}/boltz1_conf.ckpt"),
        strict=True,
        map_location="cpu",
        predict_args=predict_args,
        diffusion_process_args=asdict(BoltzDiffusionParams()),
        steering_args=asdict(steering_args),
        ema=False,
    )
    print('Loaded model into cpu')
    _torch_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    _torch_model.training = True
    _torch_model.structure_prediction_training = True
    torch.set_grad_enabled(True)
    torch.set_float32_matmul_precision('highest')
    _torch_model.eval()
    print('Model loaded')
    # wandb.init(project='sparkz', name=f'{pid}_lr={lr}')
    complex_obj = Complex(cfg.protein_pdb_path, cfg.ligand_sdf_path, cfg.ccd)
    complex_obj.alanine_mutate_inner_pocket()
    processed_pdb = complex_obj.process_for_guidance(sequence_buffer=10)
    #     batch = make_batch_from_sequence(
    #     sequence,
    #     ligand_info,
    #     other_hetatms,
    #     modified_residues,
    #     out_dir=out_dir,
    #     msa_dir=msa_dir,
    #     positions=interacting_residues,
    #     no_msa=no_msa,
    #     use_constraints=use_constraints,
    #     cache_dir=cache_dir,
    #     dock= False,  # docking_sphere_centre is not None,
    # )
    
    batch = make_batch_from_sequence(
        processed_pdb['sequence'],
        processed_pdb["smiles"],
        processed_pdb['other_hetatms'],
        processed_pdb['modified_residues'],
        out_dir='',
        cache_dir=cache_dir,
        positions=processed_pdb['pocket_constraint_residue_indices'],
        no_msa=cfg.no_msa,
        use_constraints=cfg.use_constraints,
        dock = True if cfg.mode.ligand_dock else False,
        )
    side_chain_atom_mask = torch.zeros(
        batch["ref_pos"].shape[:2], dtype=torch.long, device=batch["ref_pos"].device
    )
    side_chain_atom_mask[:, processed_pdb["sidechain_atom_mask"]] = 1
    batch["sidechain_atom_mask"] = side_chain_atom_mask
    # batch["docking_sphere_centre"] = docking_sphere_centre
    batch["ligand_atom_mask"] = torch.zeros(
        batch["ref_pos"].shape[:2], dtype=torch.bool, device=batch["ref_pos"].device
    )
    batch["ligand_atom_mask"][:, processed_pdb["whole_pocket_atom_indices"]] = True
    if processed_pdb["pocket_coords"] is not None and processed_pdb["whole_pocket_atom_indices"] is not None:
        batch["ref_pos"][..., processed_pdb["whole_pocket_and_ligand_atom_indices"], :] = torch.tensor(
            processed_pdb["pocket_coords"],
            dtype=batch["ref_pos"].dtype,
            device=batch["ref_pos"].device,
        )[
            ..., processed_pdb["whole_pocket_and_ligand_atom_indices"], :
        ]  # make ref pos only residues guiding to
        inpainting_mask = torch.zeros(
            batch["ref_pos"].shape[:2], dtype=torch.long, device=device
        )
        inpainting_mask[:, processed_pdb["whole_pocket_atom_indices"]] = 1
        batch["inpainting_mask"] = inpainting_mask.unsqueeze(-1)
        inpainting_mask_with_ligands = torch.zeros(
            batch["ref_pos"].shape[:2], dtype=torch.long, device=device
        )
        inpainting_mask_with_ligands[:, processed_pdb["whole_pocket_and_ligand_atom_indices"]] = 1
        batch["inpainting_mask_with_ligands"] = inpainting_mask_with_ligands.unsqueeze(
            -1
        )
    full_sequence = ''.join([processed_pdb['sequence'][i] for i in processed_pdb['sequence']])
    print(f'Beginning sequence optimization for length {len(full_sequence)}')
    print(torch.tensor(processed_pdb['pocket_constraint_residue_indices']))
    designed_sequence = optimise_pocket_sequence_bindcraft(_torch_model, full_sequence, processed_pdb['pocket_constraint_residue_indices'], batch, num_iterations=cfg.num_iterations, lr=cfg.lr, predict_args={'recycling_steps': cfg.recycling_steps, 'sampling_steps': cfg.sampling_steps, 'diffusion_samples': cfg.num_samples})
    complex_obj = Complex(cfg.protein_pdb_path, cfg.ligand_sdf_path, cfg.ccd)
    final_processed_data = complex_obj.alter_sequence(designed_sequence)
    if os.path.exists('final'):
        print('Removing existing final directory')
        os.system('rm -rf final')
    predictions, batch = infer(
        _torch_model,
        final_processed_data['sequence'],
        final_processed_data['pocket_constraint_residue_indices'],
        final_processed_data['pocket_coords'],
        final_processed_data['whole_pocket_atom_indices'],
        final_processed_data['ligand_atom_indices'],
        final_processed_data['whole_pocket_and_ligand_atom_indices'],
        final_processed_data['other_hetatms'],
        final_processed_data['modified_residues'], 
        final_processed_data['sidechain_atom_mask'],
        ligand_info=final_processed_data["smiles"],
        out_dir='final',
        msa_dir=None,
        no_msa=cfg.no_msa,
        use_constraints=cfg.use_constraints,
        cache_dir=cache_dir,
        docking_sphere_centre=final_processed_data["docking_sphere_centre"] if cfg.mode.ligand_dock else None,
        predict_args=predict_args,
        guidance_type=cfg.mode.guidance_type,
    )
    # predictions, batch = infer(
    #     _torch_model,
    #     processed_pdb["sequence"],
    #     processed_pdb["pocket_constraint_residue_indices"],
    #     processed_pdb["pocket_coords"],
    #     processed_pdb["whole_pocket_atom_indices"],
    #     processed_pdb["ligand_atom_indices"],
    #     processed_pdb["whole_pocket_and_ligand_atom_indices"],
    #     processed_pdb["other_hetatms"],
    #     processed_pdb["modified_residues"],
    #     processed_pdb["sidechain_atom_mask"],
    #     ligand_info=processed_pdb["smiles"],
    #     out_dir=".",
    #     cache_dir=cache_dir,
    #     msa_dir=None,
    #     no_msa=cfg.no_msa,
    #     use_constraints=cfg.use_constraints,
    #     docking_sphere_centre=processed_pdb["docking_sphere_centre"] if cfg.mode.ligand_dock else None,
    #     guidance_type=cfg.mode.guidance_type,
    # )
    # write_complex(predictions, Path(f'{out_dir}/{pid}_final/processed/structures'), Path(f'{out_dir}/{pid}_final/predictions'), batch, output_format='cif')
    print(predictions)
    boltz_writer = BoltzWriter(
        f"final/processed/structures",
        f"final/predictions",
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
    print('Sequence optimized')
    print('Initial sequence:', full_sequence)
    print('Designed sequence:', designed_sequence)
    print('Predicted final sequence:', final_processed_data['sequence'])
    # wandb.finish()
if __name__ == '__main__':
    main()