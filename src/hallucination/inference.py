from boltz.model.model import Boltz1
from boltz.main import BoltzDiffusionParams 
from boltz.data.write.writer import BoltzWriter
from dataclasses import asdict
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pytorch_lightning import Trainer, LightningModule
from typing import Any, Dict
import torch.optim as optim
from boltz.main import process_inputs, BoltzProcessedInput, download
from boltz.data.types import MSA, Manifest, Record
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.model.modules.confidence_utils import compute_aggregated_metric
from boltz.data.const import tokens, prot_letter_to_token
import io
from hallucination.constants import create_conversion_matrix
from datetime import datetime
import numpy as np
import random
# import wandb
from hallucination.loss import bindcraft_like_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer_with_gradients(model: LightningModule, batch: Any, predict_args: Dict[str, Any], protein_index=None, pocket_residues=None) -> Dict[str, torch.Tensor]:
    """
    Perform inference using the Lightning model while retaining gradients for specific metrics.
    
    Args:
        model (LightningModule): The trained Lightning module.
        batch (Any): The input batch for inference.
        predict_args (Dict[str, Any]): Prediction arguments such as recycling_steps, sampling_steps, etc.
    
    Returns:
        Dict[str, torch.Tensor]: A dictionary containing predictions and metrics with gradients.
    """
    # Ensure model is in evaluation mode and gradient tracking is enabled
    model.train()

    # Enable gradient tracking
    with torch.set_grad_enabled(True):
        # Forward pass
        out = model(
            {k:v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()},
            recycling_steps=predict_args.get("recycling_steps", 0),
            num_sampling_steps=predict_args.get("sampling_steps", 0),
            diffusion_samples=predict_args.get("diffusion_samples", 1),
            run_confidence_sequentially=predict_args.get("run_confidence_sequentially", True),
        )
        total_plddt_loss = compute_aggregated_metric(out["plddt_logits"]).mean()
        protein_plddt_loss = compute_aggregated_metric(out["plddt_logits"][:,:protein_index, :]).mean()
        pocket_plddt_loss = compute_aggregated_metric(out["plddt_logits"][:, pocket_residues, :]).mean()
        ligand_pldtt_loss = compute_aggregated_metric(out["plddt_logits"][:,protein_index:, :]).mean()
        # print("Total plddt loss:", compute_aggregated_metric(total_plddt_loss).mean())
        # print("Protein plddt loss:", compute_aggregated_metric(protein_plddt_loss).mean())
        # print("Pocket plddt loss:", compute_aggregated_metric(pocket_plddt_loss).mean())
        # print("Ligand plddt loss:", compute_aggregated_metric(ligand_pldtt_loss).mean())
        # wandb.log({
        #     "Total plddt loss": total_plddt_loss,
        #     "Protein plddt loss": protein_plddt_loss,
        #     "Pocket plddt loss": pocket_plddt_loss,
        #     "Ligand plddt loss": ligand_pldtt_loss,
        # })
        # Compute confidence score
        confidence_score = (
            out["complex_plddt"] -
            (out["ligand_iptm"] if not torch.allclose(out["iptm"], torch.zeros_like(out["iptm"])) else out["ptm"])
        ) / 1

        # Retain gradients for specific metrics if needed
        confidence_score.retain_grad()
        

        # Prepare the output dictionary
        pred_dict = {
            "coords": out["sample_atom_coords"], # predicted coordinates
            # "mask": out["masks"],  # Mask for valid atoms
            "confidence_score": confidence_score,  # Confidence score with gradients
            "pocket_plddt": pocket_plddt_loss,
            # "bindcraft_like_loss": bindcraft_like_loss(out, protein_index, pocket_residues),
        }

        # Include additional metrics, ensuring gradients are retained
        # for key in ["plddt", "ptm", "iptm", "complex_iplddt", "complex_plddt", "pde", "complex_pde", "complex_ipde", "pae", "ptm", "iptm", "ligand_iptm", "protein_iptm", "pair_chains_iptm"]:
        for key in ["plddt", "ligand_iptm", "complex_iplddt", "complex_plddt"]:
            if key in out:
                pred_dict[key] = out[key]
                if isinstance(out[key], torch.Tensor):
                    out[key].retain_grad()
        return pred_dict