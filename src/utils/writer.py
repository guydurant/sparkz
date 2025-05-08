# from boltz.data.write.writer import BoltzWriter
from torch import Tensor
import numpy as np
import torch
from pathlib import Path
from boltz.data.types import Structure, Record, Interface
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb
from dataclasses import asdict, replace
import json

def write_complex(prediction: dict[str, Tensor], data_dir: Path, output_dir: Path, batch, output_format: str = "pdb"):

    # Get the records
    # records: list[Record] = batch["record"]
    # Get the predictions
    coords = prediction["coords"] if "coords" in prediction else prediction["sample_atom_coords"]
    coords = coords.unsqueeze(0)
    records = batch["record"]
    pad_masks = batch["atom_pad_mask"]

    # Get ranking
    argsort = torch.argsort(prediction["confidence_score"], descending=True)
    idx_to_rank = {idx.item(): rank for rank, idx in enumerate(argsort)}

    # Iterate over the records
    # CHANGED TO NO PADDING _ CHECK ORIGINAL CODE
    for record, coord, pad_mask in zip(records, coords, pad_masks):
        # Load the structure
        path = data_dir / f"{record.id}.npz"
        structure: Structure = Structure.load(path)

        # Compute chain map with masked removed, to be used later
        chain_map = {}
        for i, mask in enumerate(structure.mask):
            if mask:
                chain_map[len(chain_map)] = i

        # Remove masked chains completely
        structure = structure.remove_invalid_chains()

        for model_idx in range(coord.shape[0]):
            # Get model coord
            model_coord = coord[model_idx]
            # Unpad
            coord_unpad = model_coord[pad_mask.bool()]
            coord_unpad = coord_unpad.cpu().numpy()

            # New atom table
            atoms = structure.atoms
            atoms["coords"] = coord_unpad
            atoms["is_present"] = True

            # Mew residue table
            residues = structure.residues
            residues["is_present"] = True

            # Update the structure
            interfaces = np.array([], dtype=Interface)
            new_structure: Structure = replace(
                structure,
                atoms=atoms,
                residues=residues,
                interfaces=interfaces,
            )

            # Update chain info
            chain_info = []
            for chain in new_structure.chains:
                old_chain_idx = chain_map[chain["asym_id"]]
                old_chain_info = record.chains[old_chain_idx]
                new_chain_info = replace(
                    old_chain_info,
                    chain_id=int(chain["asym_id"]),
                    valid=True,
                )
                chain_info.append(new_chain_info)

            # Save the structure
            # struct_dir = output_dir / record.id # Edited from original code
            struct_dir = output_dir
            struct_dir.mkdir(exist_ok=True)

            # Get plddt's
            plddts = None
            if "plddt" in prediction:
                plddts = prediction["plddt"][model_idx]

            # Create path name
            outname = f"{record.id}_model_{idx_to_rank[model_idx]}"

            # Save the structure
            if output_format == "pdb":
                path = struct_dir / f"{outname}.pdb"
                with path.open("w") as f:
                    f.write(to_pdb(new_structure, plddts=plddts))
            elif output_format == "mmcif":
                path = struct_dir / f"{outname}.cif"
                with path.open("w") as f:
                    f.write(to_mmcif(new_structure, plddts=plddts))
            else:
                path = struct_dir / f"{outname}.npz"
                np.savez_compressed(path, **asdict(new_structure))

            # Save confidence summary
            if "plddt" in prediction:
                path = (
                    struct_dir
                    / f"confidence_{record.id}_model_{idx_to_rank[model_idx]}.json"
                )
                confidence_summary_dict = {}
                for key in [
                    "confidence_score",
                    "ptm",
                    "iptm",
                    "ligand_iptm",
                    "protein_iptm",
                    "complex_plddt",
                    "complex_iplddt",
                    "complex_pde",
                    "complex_ipde",
                ]:
                    confidence_summary_dict[key] = prediction[key][model_idx].item()
                # confidence_summary_dict["chains_ptm"] = {
                #     idx: prediction["pair_chains_iptm"][idx][idx][model_idx].item()
                #     for idx in prediction["pair_chains_iptm"]
                # }
                # confidence_summary_dict["pair_chains_iptm"] = {
                #     idx1: {
                #         idx2: prediction["pair_chains_iptm"][idx1][idx2][
                #             model_idx
                #         ].item()
                #         for idx2 in prediction["pair_chains_iptm"][idx1]
                #     }
                #     for idx1 in prediction["pair_chains_iptm"]
                # }
                if "time_taken" in prediction:
                    confidence_summary_dict["time_taken"] = prediction["time_taken"]
                with path.open("w") as f:
                    f.write(
                        json.dumps(
                            confidence_summary_dict,
                            indent=4,
                        )
                    )

                # Save plddt
                plddt = prediction["plddt"][model_idx]
                path = (
                    struct_dir
                    / f"plddt_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                )
                np.savez_compressed(path, plddt=plddt.detach().cpu().numpy())

            # Save pae
            if "pae" in prediction:
                pae = prediction["pae"][model_idx]
                path = (
                    struct_dir
                    / f"pae_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                )
                np.savez_compressed(path, pae=pae.detach().cpu().numpy())

            # Save pde
            if "pde" in prediction:
                pde = prediction["pde"][model_idx]
                path = (
                    struct_dir
                    / f"pde_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                )
                np.savez_compressed(path, pde=pde.detach().cpu().numpy())