# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: /vols/opig/projects/guy-affinity/POCKET_DESIGN/sparkz/src/cli.py
# Bytecode version: 3.12.0rc2 (3531)
# Source timestamp: 2025-04-30 11:11:46 UTC (1746011506)

# import wandb
import torch
from boltz.main import BoltzDiffusionParams, download, BoltzSteeringParams
from pathlib import Path
from inpainting.screwz import Screwz1
from dataclasses import asdict
from hallucination.optimise import optimise_pocket_sequence_bindcraft
from inpainting.main import infer
from utils.writer import write_complex
import json
import click
import time
import os
from processing.complex import Complex
from utils.batch_generate import make_batch_from_sequence

@click.command()
@click.option('--protein_pdb_path', default=None, help='Path to input protein PDB file')
@click.option('--ligand_sdf_path', default=None, help='Path to input ligand SDF file')
@click.option('--pid', default=None, help='Unique Protein ID (e.g. 1a2b_1 or 1a2b_2) for the protein input')
@click.option('--ccd', default=None, help='Unique CCD code for the ligand input (e.g. ZRY or DMS)')
@click.option('--out_dir', default='test/', help='Output directory')
@click.option('--mode', help='Mode to run the model in', type=click.Choice(['inpainting', 'default']), default='inpainting')
@click.option('--guidance_rate', type=float, help='Guidance rate for the guided mode. Normally run as 1.0 so it is effectively inpainting', default=1.0)
@click.option('--no_msa', type=bool, help='Disable MSA', default=True)
@click.option('--use_constraints', type=bool, help='Use constraints for docking', default=False)
@click.option('--overwrite', type=bool, help='Overwrite existing files', default=False)
@click.option('--num_iterations', type=int, help='Number of iterations', default=100)
@click.option('--lr', type=float, help='Learning rate', default=0.1)
@click.option('--num_samples', type=int, help='Number of samples', default=1)
@click.option('--recycling_steps', type=int, help='Recycling steps', default=3)
@click.option('--sampling_steps', type=int, help='Sampling steps', default=200)
@click.option('--cache', type=str, help='Cache directory', default='/homes/durant/')
@click.option('--guide_ligand', type=bool, help='Guide ligand', default=True)
def main(protein_pdb_path, ligand_sdf_path, pid, ccd, out_dir, mode, guidance_rate, no_msa, use_constraints, overwrite, num_iterations, lr, num_samples, recycling_steps, sampling_steps, cache, guide_ligand):
    print('Loading model')
    steering_args = BoltzSteeringParams()
    cache_dir = Path(cache).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    download(cache_dir)
    _torch_model = Screwz1.load_from_checkpoint(Path(f'{cache_dir}/boltz1_conf.ckpt'), strict=True, map_location='cpu', predict_args={'recycling_steps': recycling_steps, 'sampling_steps': sampling_steps, 'diffusion_samples': num_samples}, diffusion_process_args=asdict(BoltzDiffusionParams()), steering_args=asdict(steering_args), ema=False)
    print('Loaded model into cpu')
    _torch_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    _torch_model.training = True
    _torch_model.structure_prediction_training = True
    torch.set_grad_enabled(True)
    torch.set_float32_matmul_precision('highest')
    _torch_model.eval()
    print('Model loaded')
    # wandb.init(project='sparkz', name=f'{pid}_lr={lr}')
    complex_obj = Complex(protein_pdb_path, ligand_sdf_path, ccd)
    complex_obj.alanine_mutate_inner_pocket()
    processed_pdb = complex_obj.process_for_guidance(sequence_buffer=10)
    batch = make_batch_from_sequence(processed_pdb['sequence'], processed_pdb["smiles"], processed_pdb['other_hetatms'], processed_pdb['modified_residues'], out_dir=f'{out_dir}/{pid}', cache_dir=cache_dir, msa_dir=f'/vols/opig/projects/guy-affinity/POCKET_DESIGN/boltz_hallucination/msas_chopped/{pid}', positions=processed_pdb['pocket_constraint_residue_indices'], no_msa=no_msa, use_constraints=use_constraints)
    print('Batch created')
    if mode == 'inpainting':
        _torch_model.structure_module.set_guidance(guidance_rate, guide_ligand)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if processed_pdb['pocket_coords'] is not None and processed_pdb['whole_pocket_atom_indices'] is not None:
            batch['ref_pos'][..., processed_pdb['whole_pocket_atom_indices'], :] = torch.tensor(processed_pdb['pocket_coords'], dtype=batch['ref_pos'].dtype, device=batch['ref_pos'].device)[..., processed_pdb['whole_pocket_atom_indices'], :]
            inpainting_mask = torch.zeros(batch['ref_pos'].shape[:2], dtype=torch.long, device=device)
            inpainting_mask[:, processed_pdb['whole_pocket_atom_indices']] = 1
            batch['inpainting_mask'] = inpainting_mask.unsqueeze(-1)
            inpainting_mask_with_ligands = torch.zeros(batch['ref_pos'].shape[:2], dtype=torch.long, device=device)
            inpainting_mask_with_ligands[:, processed_pdb['whole_pocket_and_ligand_atom_indices']] = 1
            batch['inpainting_mask_with_ligands'] = inpainting_mask_with_ligands.unsqueeze(-1)
    full_sequence = ''.join([processed_pdb['sequence'][i] for i in processed_pdb['sequence']])
    print(f'Beginning sequence optimization for length {len(full_sequence)}')
    print(torch.tensor(processed_pdb['pocket_constraint_residue_indices']))
    designed_sequence = optimise_pocket_sequence_bindcraft(_torch_model, full_sequence, processed_pdb['pocket_constraint_residue_indices'], batch, num_iterations=num_iterations, lr=lr, predict_args={'recycling_steps': recycling_steps, 'sampling_steps': sampling_steps, 'diffusion_samples': num_samples})
    final_processed_data = complex_obj.alter_sequence(designed_sequence)
    predictions, batch = infer(_torch_model, final_processed_data['sequence'], final_processed_data['pocket_constraint_residue_indices'], final_processed_data['pocket_coords'], final_processed_data['whole_pocket_atom_indices'], final_processed_data['whole_pocket_and_ligand_atom_indices'], final_processed_data['other_hetatms'], final_processed_data['modified_residues'], ligand_info=final_processed_data["smiles"], out_dir=f'{out_dir}/{pid}_final', msa_dir=None, no_msa=no_msa, use_constraints=use_constraints, cache_dir=cache_dir, predict_args={'recycling_steps': 3, 'sampling_steps': 200, 'diffusion_samples': 1})
    write_complex(predictions, Path(f'{out_dir}/{pid}_final/processed/structures'), Path(f'{out_dir}/{pid}_final/predictions'), batch, output_format='pdb')
    print('Sequence optimized')
    print('Initial sequence:', full_sequence)
    print('Designed sequence:', designed_sequence)
    print('Predicted final sequence:', final_processed_data['sequence'])
    # wandb.finish()
if __name__ == '__main__':
    main()