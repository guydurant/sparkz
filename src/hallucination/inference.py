import torch
from pytorch_lightning import LightningModule
from typing import Any, Dict


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:   
    device = torch.device("cpu")

def infer_with_gradients(model: LightningModule, batch: Any, predict_args: Dict[str, Any], protein_index=None, pocket_residues=None) -> Dict[str, torch.Tensor]:
    """
    Perform hallucination inference using the Lightning model while retaining gradients for specific metrics.
    
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
            guidance_type="rigid",
        )
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
            "confidence_score": confidence_score,  # Confidence score with gradients
            # "pocket_plddt": pocket_plddt_loss, # Turned off for now
            # "bindcraft_like_loss": bindcraft_like_loss(out, protein_index, pocket_residues), # Turned off for now
        }

        # Include additional metrics, ensuring gradients are retained
        for key in ["plddt", "ligand_iptm", "complex_iplddt", "complex_plddt"]:
            if key in out:
                pred_dict[key] = out[key]
                if isinstance(out[key], torch.Tensor):
                    out[key].retain_grad()
        return pred_dict