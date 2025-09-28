import torch
from hallucination.constants import create_conversion_matrix
from hallucination.inference import infer_with_gradients
from utils.sequence import one_hot_encode_sequence, decode_sampled_one_hot
from utils.batch_generate import (
    adjust_batch_from_pocket_sequence,
)
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F
# import wandb


def optimise_pocket_sequence_bindcraft(
    model, alanine_sequence, sequence_mask_positions, batch, num_iterations=300, lr=0.01, datetime=None, predict_args=None
):
    """
    Optimize the pocket protein sequence to maximize the confidence score using a staged approach similar to BindCraft.
    
    Args:
        model (LightningModule): The trained Lightning module.
        alanine_sequence (str): The sequence with alanine residues for pocket residues.
        sequence_mask_positions (list): List of positions in the sequence that can be modified.
        batch (dict): The input batch for inference.
        num_iterations (int): Number of optimization iterations.
        lr (float): Learning rate for the optimizer.
        datetime (str): A string representing the current date and time for logging purposes.
        predict_args (dict): Prediction arguments such as recycling_steps, sampling_steps, etc.
    
    Returns:
        str: The optimized pocket sequence.
    """
    
    # Ensure the sequence is a tensor and requires gradients
    sequence = torch.rand_like(
        one_hot_encode_sequence(alanine_sequence), requires_grad=True
    )
    sequence_mask = torch.zeros(len(sequence), dtype=torch.int64)
    sequence_mask[torch.tensor(sequence_mask_positions)] = 1
    conversion_table = create_conversion_matrix().to(sequence.device)
    og_batch = batch.copy()
    og_batch["res_type"] = og_batch["res_type"].float().requires_grad_(True)
    for key, value in og_batch.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            # make sure the tensor requires gradients
            og_batch[key] = value.clone().detach().requires_grad_(True)
    # Define the optimizer
    optimizer = optim.Adam([sequence], lr=lr)

    # Define stage switch points
    stage_switch_points = {
        1: int(num_iterations / 2),  # Stage 1 ends at iteration 25
        2: int((num_iterations / 20) * 18),
    }

    # Define temperature schedule
    def get_temperature(iteration, stage):
        if stage == 1 or stage == 3 or stage == 4:
            return 1.0
        elif stage == 2:
            return (
                1e-2
                + (1 - 1e-2)
                * (
                    1
                    - (
                        (iteration - stage_switch_points[1] + 1)
                        / (stage_switch_points[2] - stage_switch_points[1])
                    )
                )
                ** 2
            )
        else:
            raise ValueError(f"Invalid stage: {stage}")

    # Define lambda schedule
    def get_lambda(iteration, stage):
        if stage == 1:
            return (iteration + 1) / stage_switch_points[1]
        else:
            return 1.0

    # Optimization loop
    current_stage = 1
    for iteration in tqdm(range(num_iterations), desc="Optimizing sequence"):
        batch = og_batch.copy()
        # Check if we need to switch stages
        if iteration >= stage_switch_points.get(current_stage, num_iterations):
            current_stage += 1
            if current_stage == 4:
                raise ValueError("Stage 4 is not implemented yet.")

        temperature = get_temperature(iteration, current_stage)
        lambda_val = get_lambda(iteration, current_stage)

        optimizer.zero_grad()

        # Stage-specific sequence representation
        if current_stage == 1:
            sampled_sequence = (1 - lambda_val) * sequence + lambda_val * torch.softmax(
                sequence / temperature, dim=-1
            )
        elif current_stage == 2:
            sampled_sequence = torch.softmax(sequence / temperature, dim=-1)
        elif current_stage == 3:
            softmax_out = torch.softmax(sequence / temperature, dim=-1)

            # Sample from softmax using argmax (hard selection)
            hard_sample = F.one_hot(
                softmax_out.argmax(dim=-1), num_classes=softmax_out.shape[-1]
            ).float()

            # Straight-through estimator: Use hard sample in forward pass but softmax_out in backward pass
            sampled_sequence = hard_sample + (softmax_out - softmax_out.detach())
        elif current_stage == 4:
            raise ValueError("Stage 4 is not implemented yet.")


        # Adjust batch and infer
        batch = adjust_batch_from_pocket_sequence(
            sampled_sequence, sequence_mask, conversion_table, batch
        )
        predictions = infer_with_gradients(
            model, batch, predict_args, len(sequence) - 1, sequence_mask_positions
        )
        confidence_score = predictions["ligand_iptm"]

        # Compute loss and backpropagate
        loss = (
            -confidence_score.mean()
        )  # Negative because we want to maximize confidence
        loss.backward()
        sequence.grad *= sequence_mask.unsqueeze(
            -1
        )  # ignore gradients for fixed positions
        grad_norm = sequence.grad.norm()
        torch.nn.utils.clip_grad_norm_(sequence, 2.0)
        optimizer.step()

        # Clamp sequence values to ensure valid probabilities
        with torch.no_grad():
            sequence.data = torch.clamp(sequence.data, 0, 1)

        if iteration % 5 == 0:
            entropy = -torch.sum(
                sampled_sequence * torch.log(sampled_sequence + 1e-8), dim=-1
            ).mean()
            print(
                f"Iteration {iteration}: Loss={loss.item()}, Confidence={confidence_score.mean()}, Entropy={entropy}, Grad Norm={grad_norm}, Temperature={temperature}"
            )
    final_sequence = "".join(
        [
            i if mask == 1 else j
            for i, j, mask in zip(
                decode_sampled_one_hot(sequence), alanine_sequence, sequence_mask
            )
        ]
    )
    print("Final confidence score:", confidence_score.mean().item())
    return final_sequence
