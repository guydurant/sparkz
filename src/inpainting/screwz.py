from boltz.model.model import Boltz1
import torch
from typing import Any, Dict, Optional
from torch import Tensor, nn
from inpainting.modified_atom_diffusion import GuidedAtomDiffusion


class Screwz1(Boltz1):
    def __init__(  # noqa: PLR0915, C901, PLR0912
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        num_bins: int,
        training_args: dict[str, Any],
        validation_args: dict[str, Any],
        embedder_args: dict[str, Any],
        msa_args: dict[str, Any],
        pairformer_args: dict[str, Any],
        score_model_args: dict[str, Any],
        diffusion_process_args: dict[str, Any],
        diffusion_loss_args: dict[str, Any],
        confidence_model_args: dict[str, Any],
        atom_feature_dim: int = 128,
        confidence_prediction: bool = False,
        confidence_imitate_trunk: bool = False,
        alpha_pae: float = 0.0,
        structure_prediction_training: bool = True,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        compile_pairformer: bool = False,
        compile_structure: bool = False,
        compile_confidence: bool = False,
        nucleotide_rmsd_weight: float = 5.0,
        ligand_rmsd_weight: float = 10.0,
        no_msa: bool = False,
        no_atom_encoder: bool = False,
        ema: bool = False,
        ema_decay: float = 0.999,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        predict_args: Optional[dict[str, Any]] = None,
        steering_args: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            num_bins=num_bins,
            training_args=training_args,
            validation_args=validation_args,
            embedder_args=embedder_args,
            msa_args=msa_args,
            pairformer_args=pairformer_args,
            score_model_args=score_model_args,
            diffusion_process_args=diffusion_process_args,
            diffusion_loss_args=diffusion_loss_args,
            confidence_model_args=confidence_model_args,
            atom_feature_dim=atom_feature_dim,
            confidence_prediction=confidence_prediction,
            confidence_imitate_trunk=confidence_imitate_trunk,
            alpha_pae=alpha_pae,
            structure_prediction_training=structure_prediction_training,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            compile_pairformer=compile_pairformer,
            compile_structure=compile_structure,
            compile_confidence=compile_confidence,
            nucleotide_rmsd_weight=nucleotide_rmsd_weight,
            ligand_rmsd_weight=ligand_rmsd_weight,
            no_msa=no_msa,
            no_atom_encoder=no_atom_encoder,
            ema=ema,
            ema_decay=ema_decay,
            min_dist=min_dist,
            max_dist=max_dist,
            predict_args=predict_args,
            steering_args=steering_args,
        )
        use_accumulate_token_repr = (
            confidence_prediction
            and "use_s_diffusion" in confidence_model_args
            and confidence_model_args["use_s_diffusion"]
        )
        self.structure_module = GuidedAtomDiffusion(
            score_model_args={
                "token_z": token_z,
                "token_s": token_s,
                "atom_z": atom_z,
                "atom_s": atom_s,
                "atoms_per_window_queries": atoms_per_window_queries,
                "atoms_per_window_keys": atoms_per_window_keys,
                "atom_feature_dim": atom_feature_dim,
                **score_model_args,
            },
            compile_score=compile_structure,
            accumulate_token_repr=use_accumulate_token_repr,
            **diffusion_process_args,
        )



    def forward(
        self,
        feats: dict[str, Tensor],
        recycling_steps: int = 0,
        num_sampling_steps: Optional[int] = None,
        multiplicity_diffusion_train: int = 1,
        diffusion_samples: int = 1,
        run_confidence_sequentially: bool = False,
    ) -> dict[str, Tensor]:
        # Compute input embeddings
        with torch.set_grad_enabled(
            self.training and self.structure_prediction_training
        ):
            s_inputs = self.input_embedder(feats)

            # Initialize the sequence and pairwise embeddings
            s_init = self.s_init(s_inputs)
            z_init = (
                self.z_init_1(s_inputs)[:, :, None]
                + self.z_init_2(s_inputs)[:, None, :]
            )
            relative_position_encoding = self.rel_pos(feats)
            z_init = z_init + relative_position_encoding
            z_init = z_init + self.token_bonds(feats["token_bonds"].float())

            # Perform rounds of the pairwise stack
            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)

            # Compute pairwise mask
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]

            for i in range(recycling_steps + 1):
                with torch.set_grad_enabled(self.training and (i == recycling_steps)):
                    # Fixes an issue with unused parameters in autocast
                    if (
                        self.training
                        and (i == recycling_steps)
                        and torch.is_autocast_enabled()
                    ):
                        torch.clear_autocast_cache()

                    # Apply recycling
                    s = s_init + self.s_recycle(self.s_norm(s))
                    z = z_init + self.z_recycle(self.z_norm(z))

                    # Compute pairwise stack
                    if not self.no_msa:
                        z = z + self.msa_module(z, s_inputs, feats)

                    # Revert to uncompiled version for validation
                    if self.is_pairformer_compiled and not self.training:
                        pairformer_module = (
                            self.pairformer_module._orig_mod
                        )  # noqa: SLF001
                    else:
                        pairformer_module = self.pairformer_module

                    s, z = pairformer_module(s, z, mask=mask, pair_mask=pair_mask)

            pdistogram = self.distogram_module(z)
            dict_out = {"pdistogram": pdistogram}

        # Compute structure module
        if self.training and self.structure_prediction_training:
            dict_out.update(
                self.structure_module(
                    s_trunk=s,
                    z_trunk=z,
                    s_inputs=s_inputs,
                    feats=feats,
                    relative_position_encoding=relative_position_encoding,
                    multiplicity=multiplicity_diffusion_train,
                )
            )

        if (not self.training) or self.confidence_prediction:
            dict_out.update(
                self.structure_module.sample(
                    s_trunk=s,
                    z_trunk=z,
                    s_inputs=s_inputs,
                    feats=feats,
                    relative_position_encoding=relative_position_encoding,
                    steering_args=self.steering_args,
                    num_sampling_steps=num_sampling_steps,
                    atom_mask=feats["atom_pad_mask"],
                    multiplicity=diffusion_samples,
                    train_accumulate_token_repr=self.training,
                    atom_coords_true=feats["ref_pos"],
                    atom_coords_true_mask_no_ligands=feats["inpainting_mask"],
                    atom_coords_true_mask_with_ligands=feats["inpainting_mask_with_ligands"],
                    ligand_mask=feats["ligand_atom_mask"],
                )
            )

        if self.confidence_prediction:
            dict_out.update(
                self.confidence_module(
                    # s_inputs=s_inputs.detach(),
                    s_inputs=s_inputs,
                    # s=s.detach(),
                    s=s,
                    # z=z.detach(),
                    z = z,
                    s_diffusion=(
                        dict_out["diff_token_repr"]
                        if self.confidence_module.use_s_diffusion
                        else None
                    ),
                    x_pred=dict_out["sample_atom_coords"].detach(),
                    feats=feats,
                    pred_distogram_logits=dict_out["pdistogram"].detach(),
                    multiplicity=diffusion_samples,
                    run_sequentially=run_confidence_sequentially,
                )
            )
        if self.confidence_prediction and self.confidence_module.use_s_diffusion:
            dict_out.pop("diff_token_repr", None)
        return dict_out
