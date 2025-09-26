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
    Optimize the pocket protein sequence to improve binding affinity or other metrics.

    Args:
        model (LightningModule): The trained Boltz model.
        batch (dict): The input batch containing the protein sequence and other information.
        num_iterations (int): Number of optimization iterations.
        lr (float): Learning rate for the optimizer.

    Returns:
        torch.Tensor: The optimized protein sequence.
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
    # random_decoded = decode_sampled_one_hot(sequence)
    # random_sequence = "".join([i if mask == 1 else j for i, j, mask in zip(random_decoded, initial_sequence, sequence_mask)])
    # # print("Random sequence:", random_sequence)
    # random_batch = make_batch_from_sequence(random_sequence, ligand_info, "random", datetime, interacting_residues)
    # predictions = infer_with_gradients(model, random_batch.copy(), predict_args, len(sequence)-1, sequence_positions)
    # print("Random sequence pLLDT:", predictions["plddt"].mean())

    # Define the optimizer
    optimizer = optim.Adam([sequence], lr=lr)

    # Define stage switch points
    stage_switch_points = {
        1: int(num_iterations / 2),  # Stage 1 ends at iteration 25
        # 2: 95,  # Stage 2 ends at iteration 95
        # 2: int((num_iterations / 20) * 19),
        2: num_iterations,
        # 3: 100,  # Stage 3 ends at iteration 100
        # 3: num_iterations,
        # 4: 101,  # Stage 4 ends at iteration 115
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
    best_sequence = {
        "sequence": None,
        "confidence": 0.0,
    }
    for iteration in tqdm(range(num_iterations), desc="Optimizing sequence"):
        batch = og_batch.copy()
        # Check if we need to switch stages
        if iteration >= stage_switch_points.get(current_stage, num_iterations):
            current_stage += 1
            print(f"Switching to Stage {current_stage} at iteration {iteration}")
            if current_stage == 4:
                raise ValueError("Stage 4 is not implemented yet.")
                decoded_sequence = decode_sampled_one_hot(sequence)
                best_sequence["sequence"] = "".join(
                    [
                        i if mask == 1 else j
                        for i, j, mask in zip(
                            decoded_sequence, alanine_sequence, sequence_mask
                        )
                    ]
                )
                batch = make_batch_from_pocket_sequence(
                    best_sequence["sequence"],
                    sequence_mask,
                    initial_sequence,
                    ligand_info,
                    iteration,
                    datetime,
                    interacting_residues,
                )
                confidence = infer_with_gradients(
                    model, batch, predict_args, len(sequence) - 1, sequence_positions
                )["ligand_iptm"]
                best_sequence["confidence"] = confidence.mean().item()

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
            # sampled_sequence = torch.softmax(sequence / temperature, dim=-1)
            # # Straight-through estimator
            # sampled_sequence = (sampled_sequence - sampled_sequence.detach()) + sampled_sequence.detach().round()
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
        # confidence_score = predictions["confidence_score"]
        confidence_score = predictions["ligand_iptm"]
        # confidence_score = predictions["pocket_plddt"]

        # Compute loss and backpropagate
        loss = (
            -confidence_score.mean()
        )  # Negative because we want to maximize confidence
        loss.backward()
        sequence.grad *= sequence_mask.unsqueeze(
            -1
        )  # ignore gradients for fixed positions
        grad_norm = sequence.grad.norm()
        
        # global_max = 0.0

        # for group in optimizer.param_groups:
        #     for param in group["params"]:
        #         if param.grad is None:
        #             continue
        #         state = optimizer.state[param]
        #         if "exp_avg" in state and "exp_avg_sq" in state:
        #             m = state["exp_avg"]
        #             v = state["exp_avg_sq"]
        #             denom = (v.sqrt() + group["eps"])
        #             effective_step = group["lr"] * (m / denom)
        #             param_max = effective_step.abs().max().item()
        #             global_max = max(global_max, param_max)

        # print("Global max effective step size:", global_max)
        # clip gradients
        
        # param_values = []

        # for group in optimizer.param_groups:
        #     for param in group["params"]:
        #         if param.grad is None:
        #             continue
        #         state = optimizer.state[param]
        #         if "exp_avg" in state and "exp_avg_sq" in state:
        #             m = state["exp_avg"]
        #             v = state["exp_avg_sq"]
        #             denom = (v.sqrt() + group["eps"])
        #             effective_step = group["lr"] * (m / denom)
                    
        #             # Collect all effective steps for each parameter
        #             param_values.append(effective_step.abs().cpu().numpy())  # Add abs values to list for median

        # # Flatten the list of effective steps for all parameters and compute the median
        # param_values = torch.tensor(param_values)
        # median_param_value = param_values.median().item()

        # print("Median of all parameters:", median_param_value)
        torch.nn.utils.clip_grad_norm_(sequence, 2.0)
        optimizer.step()

        # Clamp sequence values to ensure valid probabilities
        with torch.no_grad():
            sequence.data = torch.clamp(sequence.data, 0, 1)

        if iteration % 5 == 0:
            entropy = -torch.sum(
                sampled_sequence * torch.log(sampled_sequence + 1e-8), dim=-1
            ).mean()
            # wandb.log(
            #     {
            #         "Loss": loss.item(),
            #         "Confidence": confidence_score.mean().item(),
            #         "Entropy": entropy.item(),
            #         "Grad Norm": grad_norm.item(),
            #         "Temperature": temperature,
            #     }
            # )
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
