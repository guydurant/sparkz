from boltz.model.modules.diffusion import AtomDiffusion
from math import sqrt
from boltz.model.modules.utils import (
    default,
)
from boltz.model.loss.diffusion import (
    weighted_rigid_align,
)
from boltz.model.modules.utils import random_rotations
import torch
from torch import Tensor, nn
from typing import Any, Dict, Optional
from einops import rearrange


def randomly_rotate(
    coords, return_extra_coords=False, second_coords=None, third_coords=None
):
    R = random_rotations(len(coords), coords.dtype, coords.device)

    if return_extra_coords:
        return (
            torch.einsum("bmd,bds->bms", coords, R),
            (
                torch.einsum("bmd,bds->bms", second_coords, R)
                if second_coords is not None
                else None
            ),
            (
                torch.einsum("bmd,bds->bms", third_coords, R)
                if third_coords is not None
                else None
            ),
        )

    return torch.einsum("bmd,bds->bms", coords, R)


def center_random_augmentation(
    atom_coords,
    atom_mask,
    s_trans=1.0,
    augmentation=True,
    centering=True,
    return_extra_coords=False,
    second_coords=None,
    third_coords=None,
):
    """Center and randomly augment the input coordinates.

    Parameters
    ----------
    atom_coords : Tensor
        The atom coordinates.
    atom_mask : Tensor
        The atom mask.
    s_trans : float, optional
        The translation factor, by default 1.0
    augmentation : bool, optional
        Whether to add rotational and translational augmentation the input, by default True
    centering : bool, optional
        Whether to center the input, by default True

    Returns
    -------
    Tensor
        The augmented atom coordinates.

    """
    if centering:
        atom_mean = torch.sum(
            atom_coords * atom_mask[:, :, None], dim=1, keepdim=True
        ) / torch.sum(atom_mask[:, :, None], dim=1, keepdim=True)
        atom_coords = atom_coords - atom_mean

        if second_coords is not None:
            # apply same transformation also to this input
            second_coords = second_coords - atom_mean
        if third_coords is not None:
            third_coords = third_coords - atom_mean

    if augmentation:
        atom_coords, second_coords, third_coords = randomly_rotate(
            atom_coords,
            return_extra_coords=True,
            second_coords=second_coords,
            third_coords=third_coords,
        )
        random_trans = torch.randn_like(atom_coords[:, 0:1, :]) * s_trans
        atom_coords = atom_coords + random_trans

        if second_coords is not None:
            second_coords = second_coords + random_trans
        if third_coords is not None:
            third_coords = third_coords + random_trans

    if return_extra_coords:
        return atom_coords, second_coords, third_coords

    return atom_coords


class GuidedAtomDiffusion(AtomDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(
        self,
        atom_mask,
        atom_coords_true,
        atom_coords_true_mask,
        num_sampling_steps=None,
        multiplicity=1,
        train_accumulate_token_repr=False,
        steering_args=None,
        **network_condition_kwargs,
    ):

        # inpainting=True
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
        # atom_coords = atom_coords * (1 - atom_coords_true_mask) + atom_coords_true * atom_coords_true_mask
        atom_coords_denoised = None
        model_cache = {} if self.use_inference_model_cache else None

        token_repr = None
        token_a = None

        # gradually denoise
        for sigma_tm, sigma_t, gamma in sigmas_and_gammas:
            atom_coords, atom_coords_denoised, atom_coords_true_augmented = (
                center_random_augmentation(
                    atom_coords,
                    atom_mask,
                    augmentation=True,
                    return_extra_coords=True,
                    second_coords=atom_coords_denoised,
                    third_coords=atom_coords_true,
                )
            )

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            t_hat = sigma_tm * (1 + gamma)
            eps = (
                self.noise_scale
                * sqrt(t_hat**2 - sigma_tm**2)
                * torch.randn(shape, device=self.device)
            )
            atom_coords_noisy = atom_coords + eps
            # atom_coords_true_noisy = atom_coords_true_augmented + sigma_t * torch.randn_like(atom_coords_true_augmented)
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
                    atom_coords_true_noisy = weighted_rigid_align(
                        atom_coords_true_noisy.float(),
                        atom_coords_denoised.float(),
                        # atom_mask.float(),
                        atom_coords_true_mask.squeeze(-1).float(),
                        atom_coords_true_mask.squeeze(-1).float(),
                        # atom_mask.float(),
                    )

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy
                + self.step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )
            if sigma_t != sigmas[-1]:
                # print('Original atom_coords_next:', atom_coords_next[:, :10, :])
                # print('Original atom_coords_true:', atom_coords_true_noisy[:, :10, :])
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
                # if at the last step, we should use the predicted coordinates

                # print('Modified atom_coords_next:', atom_coords_next[:, :10, :])
            atom_coords = atom_coords_next

            # print('atom_coords:', atom_coords[:, :10, :])

        return dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr)

    def set_guidance_rate(self, guidance_rate: float) -> None:
        self.guidance_rate = guidance_rate
