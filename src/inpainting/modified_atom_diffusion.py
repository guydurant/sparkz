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
from boltz.model.potentials.potentials import get_potentials

class GuidedAtomDiffusion(AtomDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self.score_model = self.score_model.to(device)

    def sample(
        self,
        atom_mask,
        atom_coords_true,
        atom_coords_true_mask_no_ligands,
        atom_coords_true_mask_with_ligands,
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
            if step_idx > num_sampling_steps * 0.8 and self.guide_ligand:
                # print(step_idx, "Switching to not including the ligand mask", self.guide_ligand)
                atom_coords_true_mask = atom_coords_true_mask_no_ligands
            # print(atom_coords_true_mask)
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
                    if sigma_t != sigmas[-1] and atom_coords_true is not None:
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
            if sigma_t != sigmas[-1] and atom_coords_true is not None:
                atom_coords_next = (atom_coords_next * (1 - atom_coords_true_mask)) + (
                    (
                        atom_coords_next
                        + (
                            (atom_coords_true_noisy - atom_coords_next)
                            * self.guidance_rate
                        )
                    )
                    * atom_coords_true_mask
                )
            atom_coords = atom_coords_next
        return dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr)

    def set_guidance(self, guidance_rate: float, guide_ligand: bool) -> None:
        self.guidance_rate = guidance_rate
        self.guide_ligand = guide_ligand