from boltz.model.modules.diffusion import AtomDiffusion
from math import sqrt
from boltz.model.modules.utils import (
    default,
)
from boltz.model.loss.diffusion import (
    weighted_rigid_align,
)
from boltz.model.modules.utils import (
    LinearNoBias,
    compute_random_augmentation,
    center_random_augmentation,
    default,
    log,
)
import torch.nn.functional as F
from boltz.model.modules.utils import random_rotations
import torch
from torch import Tensor, nn
from typing import Any, Dict, Optional
from einops import rearrange
from potentials.sidechain_clash import get_potentials

class GuidedAtomDiffusion(AtomDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(
        self,
        atom_mask,
        atom_coords_true,
        atom_coords_true_mask_no_ligands,
        atom_coords_true_mask_with_ligands,
        ligand_mask=None,
        num_sampling_steps=None,
        multiplicity=1,
        train_accumulate_token_repr=False,
        steering_args=None,
        **network_condition_kwargs,
    ):
        potentials = get_potentials()
        if steering_args["fk_steering"]:
            multiplicity = multiplicity * steering_args["num_particles"]
            energy_traj = torch.empty((multiplicity, 0), device=self.device)
            resample_weights = torch.ones(multiplicity, device=self.device).reshape(
                -1, steering_args["num_particles"]
            )
        if steering_args["guidance_update"]:
            scaled_guidance_update = torch.zeros(
                (multiplicity, *atom_mask.shape[1:], 3),
                dtype=torch.float32,
                device=self.device,
            )

        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        atom_coords_true_mask_no_ligands = atom_coords_true_mask_no_ligands.repeat_interleave(
            multiplicity, 0
        )
        atom_coords_true_mask_with_ligands = atom_coords_true_mask_with_ligands.repeat_interleave(
            multiplicity, 0
        )
        ligand_mask = ligand_mask.repeat_interleave(multiplicity, 0) if ligand_mask is not None else None
        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        # atom position is noise at the beginning
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        atom_coords_denoised = None
        model_cache = {} if self.use_inference_model_cache else None

        token_repr = None
        token_a = None
        if self.guide_ligand:
            atom_coords_true_mask = atom_coords_true_mask_with_ligands
        else:
            atom_coords_true_mask = atom_coords_true_mask_no_ligands
        # gradually denoise
        for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            # check if over 90% of the time steps are used
            # if step_idx > num_sampling_steps * 0.8 and self.guide_ligand:
            #     print(step_idx, "Switching to not including the ligand mask", self.guide_ligand)
            #     atom_coords_true_mask = atom_coords_true_mask_no_ligands
            # print(atom_coords_true_mask)
            # get guidance as an exponentially decaying rate for the ratio of time steps
            # guidance_rate = (
            #     torch.exp(
            #         torch.tensor(-5 * (step_idx / num_sampling_steps), dtype=torch.float32)
            #     ))
            # guidance_rate = torch.tensor(1 - (step_idx / num_sampling_steps))
            progress = step_idx / num_sampling_steps
            # guidance_rate = torch.tensor(max(0.0, 1 - (progress / 0.95)))
            guidance_rate = torch.tensor(max(0.0,
                -1*(progress/0.95)**2 + 1.0
            ))
            # guidance_rate = 0.0
            # print("Guidance rate", guidance_rate.item())
            random_R, random_tr = compute_random_augmentation(
                multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
            )
            atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
            atom_coords = (
                torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
            )
            if atom_coords_denoised is not None:
                atom_coords_denoised -= atom_coords_denoised.mean(dim=-2, keepdims=True)
                atom_coords_denoised = (
                    torch.einsum("bmd,bds->bms", atom_coords_denoised, random_R)
                    + random_tr
                )
            if atom_coords_true is not None:
                atom_coords_true_augmented = atom_coords_true - atom_coords_true.mean(
                    dim=-2, keepdims=True
                )
                atom_coords_true_augmented = (
                    torch.einsum("bmd,bds->bms", atom_coords_true_augmented, random_R)
                    + random_tr
                )
            # if docking_sphere_centre is not None:
            #     docking_sphere_centre_augmented = docking_sphere_centre -= atom_coords_true.mean(dim=-2, keepdims=True)
            #     docking_sphere_centre = (
            #         torch.einsum("bmd,bds->bms", docking_sphere_centre, random_R)
            #         + random_tr
            #     ) 
            if steering_args["guidance_update"] and scaled_guidance_update is not None:
                scaled_guidance_update = torch.einsum(
                    "bmd,bds->bms", scaled_guidance_update, random_R
                )

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            t_hat = sigma_tm * (1 + gamma)
            steering_t = 1.0 - (step_idx / num_sampling_steps)
            noise_var = self.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = sqrt(noise_var) * torch.randn(shape, device=self.device)
            atom_coords_noisy = atom_coords + eps
            atom_coords_true_noisy = atom_coords_true_augmented + eps * (
                sigma_t / (t_hat + 1e-8)
            )
            
            # # put docking sphere coords in network_condtion_kwargs["feats"]
            # if docking_sphere_centre is not None:
            #     network_condition_kwargs["feats"]["docking_sphere_centre"] = (
            #         docking_sphere_centre.unsqueeze(1)
            #     )

            with torch.no_grad():
                atom_coords_denoised, token_a = self.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    training=False,
                    network_condition_kwargs=dict(
                        multiplicity=multiplicity,
                        model_cache=model_cache,
                        **network_condition_kwargs,
                    ),
                )
                # TODO MAKE THIS LESS UGLY
                # atom_coords_denoised[:, -1, :] = atom_coords_true_noisy[:, -1, :]

                if steering_args["fk_steering"] and (
                    (
                        step_idx % steering_args["fk_resampling_interval"] == 0
                        and noise_var > 0
                    )
                    or step_idx == num_sampling_steps - 1
                ):
                    # Compute energy of x_0 prediction
                    energy = torch.zeros(multiplicity, device=self.device)
                    for potential in potentials:
                        parameters = potential.compute_parameters(steering_t)
                        if parameters["resampling_weight"] > 0:
                            component_energy = potential.compute(
                                atom_coords_denoised,
                                network_condition_kwargs["feats"],
                                parameters,
                            )
                            energy += parameters["resampling_weight"] * component_energy
                    energy_traj = torch.cat((energy_traj, energy.unsqueeze(1)), dim=1)

                    # Compute log G values
                    if step_idx == 0:
                        log_G = -1 * energy
                    else:
                        log_G = energy_traj[:, -2] - energy_traj[:, -1]

                    # Compute ll difference between guided and unguided transition distribution
                    if steering_args["guidance_update"] and noise_var > 0:
                        ll_difference = (
                            eps**2 - (eps + scaled_guidance_update) ** 2
                        ).sum(dim=(-1, -2)) / (2 * noise_var)
                    else:
                        ll_difference = torch.zeros_like(energy)

                    # Compute resampling weights
                    resample_weights = F.softmax(
                        (ll_difference + steering_args["fk_lambda"] * log_G).reshape(
                            -1, steering_args["num_particles"]
                        ),
                        dim=1,
                    )

                # Compute guidance update to x_0 prediction
                if (
                    steering_args["guidance_update"]
                    and step_idx < num_sampling_steps - 1
                ):
                    guidance_update = torch.zeros_like(atom_coords_denoised)
                    for guidance_step in range(steering_args["num_gd_steps"]):
                        energy_gradient = torch.zeros_like(atom_coords_denoised)
                        for potential in potentials:
                            parameters = potential.compute_parameters(steering_t)
                            if (
                                parameters["guidance_weight"] > 0
                                and (guidance_step) % parameters["guidance_interval"]
                                == 0
                            ):
                                energy_gradient += parameters[
                                    "guidance_weight"
                                ] * potential.compute_gradient(
                                    atom_coords_denoised + guidance_update,
                                    network_condition_kwargs["feats"],
                                    parameters,
                                )
                        guidance_update -= energy_gradient
                    atom_coords_denoised += guidance_update
                    scaled_guidance_update = (
                        guidance_update
                        * -1
                        * self.step_scale
                        * (sigma_t - t_hat)
                        / t_hat
                    )

                if steering_args["fk_steering"] and (
                    (
                        step_idx % steering_args["fk_resampling_interval"] == 0
                        and noise_var > 0
                    )
                    or step_idx == num_sampling_steps - 1
                ):
                    resample_indices = (
                        torch.multinomial(
                            resample_weights,
                            resample_weights.shape[1]
                            if step_idx < num_sampling_steps - 1
                            else 1,
                            replacement=True,
                        )
                        + resample_weights.shape[1]
                        * torch.arange(
                            resample_weights.shape[0], device=resample_weights.device
                        ).unsqueeze(-1)
                    ).flatten()

                    atom_coords = atom_coords[resample_indices]
                    atom_coords_noisy = atom_coords_noisy[resample_indices]
                    atom_mask = atom_mask[resample_indices]
                    atom_coords_true_noisy = atom_coords_true_noisy[resample_indices]
                    atom_coords_true_mask = atom_coords_true_mask[resample_indices]
                    ligand_mask = (
                        ligand_mask[resample_indices]
                        if ligand_mask is not None
                        else None
                    )
                    if atom_coords_denoised is not None:
                        atom_coords_denoised = atom_coords_denoised[resample_indices]
                    energy_traj = energy_traj[resample_indices]
                    if steering_args["guidance_update"]:
                        scaled_guidance_update = scaled_guidance_update[
                            resample_indices
                        ]
                    if token_repr is not None:
                        token_repr = token_repr[resample_indices]
                    if token_a is not None:
                        token_a = token_a[resample_indices]

            if self.accumulate_token_repr:
                if token_repr is None:
                    token_repr = torch.zeros_like(token_a)

                with torch.set_grad_enabled(train_accumulate_token_repr):
                    sigma = torch.full(
                        (atom_coords_denoised.shape[0],),
                        t_hat,
                        device=atom_coords_denoised.device,
                    )
                    token_repr = self.out_token_feat_update(
                        times=self.c_noise(sigma), acc_a=token_repr, next_a=token_a
                    )

            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )
                    # if sigma_t != sigmas[-1] and atom_coords_true is not None:
                    if atom_coords_true is not None:
                        atom_coords_true_noisy = weighted_rigid_align(
                            atom_coords_true_noisy.float(),
                            atom_coords_denoised.float(),
                            atom_coords_true_mask.squeeze(-1).float(),
                            atom_coords_true_mask.squeeze(-1).float(),
                        )

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy
                + self.step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )
            
            if atom_coords_true is not None:# and sigma_t != sigmas[-1]:
                
                # Move to be inside the docking sphere if necessary
                # target_index = atom_coords_true_noisy.shape[1] - 1
                # docking_center = atom_coords_true_noisy[:, target_index, :]
                # ligand_positions = []
                # for b in range(atom_coords_next.shape[0]):  # batch size
                #     ligand_b = atom_coords_next[b][ligand_mask[b]]  # shape: [17, 3] per sample (if fixed)
                #     ligand_positions.append(ligand_b)

                # ligand_positions = torch.stack(ligand_positions, dim=0)  # shape: [3, 17, 3]
                # dist_squared = ((ligand_positions - docking_center[:, None, :]) ** 2).sum(dim=-1)
                
                # radius = 25
                # if (dist_squared > radius**2).any():
                #     # print("WARNING: Ligand positions outside docking sphere, translating them back in.")

                #     dist = dist_squared.sqrt()  # [B, N_ligand]
                #     max_dist, max_idx = dist.max(dim=1)  # [B], [B]

                #     for b in range(ligand_positions.shape[0]):
                #         if max_dist[b] > radius:
                #             atom_pos = ligand_positions[b, max_idx[b]]  # [3]
                #             center = docking_center[b]  # [3]
                #             direction = atom_pos - center
                #             norm = torch.norm(direction)
                #             if norm > 1e-6:
                #                 direction = direction / norm
                #                 shift = direction * (max_dist[b] - radius)
                #                 ligand_positions[b] -= shift  # translate all atoms in batch b

                #     # Assign back to atom_coords_next using boolean mask
                #     flat_mask = ligand_mask.view(-1)                   # [B * N_atoms]
                #     flat_coords = atom_coords_next.view(-1, 3)         # [B * N_atoms, 3]
                #     flat_coords[flat_mask] = ligand_positions.reshape(-1, 3)
                #     atom_coords_next = flat_coords.view_as(atom_coords_next)

                # Finally overwrite the atom coordinates with the true coordinates for the rest
                # print("origina÷l final atom coords", atom_coords_next[:, -1, :])
                atom_coords_next = (atom_coords_next * (1 - atom_coords_true_mask)) + (
                    (
                        atom_coords_next
                        + (
                            (atom_coords_true_noisy - atom_coords_next)
                            * guidance_rate
                        )
                    )
                    * atom_coords_true_mask
                )
                # print("where it should be", atom_coords_true_noisy[:, -1, :])
                # print("overwritten final atom coords", atom_coords_next[:, -1, :])
                backbone_to_token = network_condition_kwargs["feats"]["atom_to_token"].clone()
                backbone_to_token[atom_coords_true_mask[0].T == 0] = 0
                guidance_vectors = atom_coords_true_noisy - atom_coords_next
                backbone_to_token = backbone_to_token.expand(guidance_vectors.shape[0], -1, -1)
                # print(backbone_to_token.shape, "backbone to token shape")
                # print(guidance_vectors.shape, "guidance vectors shape")
                guidance_vectors_expanded = guidance_vectors.unsqueeze(2)
                backbone_mask_expanded = backbone_to_token.unsqueeze(-1) 
                # sum_vectors = guidance_vectors.T @ backbone_to_token  # [3, N_atoms] x [N_atoms, N_tokens] -> [3, N_tokens]
                masked_guidance = guidance_vectors_expanded * backbone_mask_expanded
                # print(masked_guidance.shape, "masked guidance shape")
                sum_guidance = masked_guidance.sum(dim=1)  # [B, N_tokens, 3]
                token_weights = backbone_to_token.sum(dim=1).unsqueeze(-1).clamp(min=1e-8)
                average_masked_guidance = sum_guidance / token_weights
                # print(average_masked_guidance.shape, "average masked guidance shape")
                
                sidechain_to_token = network_condition_kwargs["feats"]["atom_to_token"].clone()
                sidechain_to_token[network_condition_kwargs["feats"]["sidechain_atom_mask"] == 0] = 0
                sidechain_to_token = sidechain_to_token.expand(guidance_vectors.shape[0], -1, -1)
                # sidechain_to_token = sidechain_to_token.permute(0, 2, 1)
                # print(sidechain_to_token.shape, "sidechain to token shape")
                atom_guidance = sidechain_to_token.float() @ average_masked_guidance
                
                # print("atom guidance shape", atom_guidance.shape)
                
                atom_coords_next = (
                    atom_coords_next
                    + atom_guidance * guidance_rate
                ) * atom_coords_true_mask + atom_coords_next * (1 - atom_coords_true_mask)
                # print("should be unchanged", atom_coords_next[:, -1, :])
                # Count atoms per token
                # counts = backbone_to_token.sum(dim=0).clamp(min=1).unsqueeze(1)  # [N_tokens, 1]

                # # Mean guidance vectors per token (residue)
                # mean_guidance_vectors = sum_vectors / counts
                # print(mean_guidance_vectors.shape, "mean guidance vectors shape")
            # print("where it should be", atom_coords_next[:, -1, :])
            # print("where it ends up", atom_coords_next[:, -1:, :])
            atom_coords = atom_coords_next
            # print("diff", step_idx, atom_coords[:, -1, :] - atom_coords_true_noisy[:, -1, :])
        print("Final atom coords",  atom_coords.shape)
        print("Final atom mask", atom_mask.shape, atom_mask[:, -25:])
        
        print("Final atom true coords", atom_coords_true_noisy.shape)
        print("Final atom mask", atom_coords_true_mask.shape, atom_coords_true_mask[:, -25:, :])
        # print("Final atom mask", atom_mask.shape, atom_mask[-25:, :])
        return dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr)

    def set_guidance(self, guidance_rate: float, guide_ligand: bool) -> None:
        self.guidance_rate = guidance_rate
        self.guide_ligand = guide_ligand