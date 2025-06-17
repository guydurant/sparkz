import torch
import os
from pathlib import Path
from boltz.main import process_inputs
from boltz.data.types import Manifest
from boltz.data.module.inference import BoltzInferenceDataModule
from hallucination.constants import create_conversion_matrix


def get_sequence_mask(sequence, positions):
    mask = torch.zeros(len(sequence))
    mask[torch.tensor(positions, dtype=torch.long)] = 1
    return mask


def adjust_batch_from_pocket_sequence(sequence, sequence_mask, conversion_table, batch):
    """
    Adjust the batch to include the pocket sequence and mask.

    Args:
        sequence (torch.Tensor): The pocket sequence.
        sequence_mask (torch.Tensor): The pocket sequence mask.
        conversion_table (torch.Tensor): The conversion table.
        batch (dict): The original batch.

    Returns:
        dict: The adjusted batch.
    """
    device = batch["res_type"].device
    conversion_table = create_conversion_matrix().to(device)
    new_sequence = sequence @ conversion_table.to(device)
    new_sequence_mask = sequence_mask.unsqueeze(1).to(device)

    batch["res_type"] = batch["res_type"].clone()
    frozen_sequence = batch["res_type"][0, : new_sequence.size(0)].clone()
    # only replace the pocket sequence where new_sequence_mask
    updated_sequence = (
        new_sequence_mask * new_sequence + (1 - new_sequence_mask) * frozen_sequence
    )
    batch["res_type"][0, : new_sequence.size(0)] = updated_sequence
    return batch


# def make_batch_from_pocket_sequence(
#     sequence,
#     sequence_mask,
#     original_sequence,
#     ligand_info,
#     iteration,
#     datetime,
#     interacting_residues,
# ):
#     # where sequence_mask is 1, replace the sequence with the new sequence
#     # where sequence_mask is 0, keep the original sequence
#     sequence_mask = (
#         sequence_mask.cpu().numpy()
#         if isinstance(sequence_mask, torch.Tensor)
#         else sequence_mask
#     )
#     updated_sequence = "".join(
#         [
#             i if mask == 1 else j
#             for i, j, mask in zip(sequence, original_sequence, sequence_mask)
#         ]
#     )
#     print("Updated sequence:", updated_sequence)
#     return make_batch_from_sequence(
#         updated_sequence,
#         ligand_info,
#         iteration,
#         datetime,
#         positions=interacting_residues,
#     )


def next_letter(letter_list, nums=0):
    return chr(ord(letter_list[-1]) + 1 + nums)


def get_chain_for_position(pos, sequences):
    """Determine the chain and relative position given a global position and sequence lengths."""
    start = 0
    for chain, seq in sequences.items():
        end = start + len(seq) - 1
        if start <= pos <= end:
            return [
                chain,
                # WARNING is 1-indexed hence the +1
                pos - start + 1,
            ]  # Convert to relative position within the chain
        start = end + 1  # Move to the next chain range
    raise ValueError(f"Position {pos} is out of range")

def make_batch_from_sequence(
    sequences: dict,
    ligand_smiles: str,
    other_hetatms: list,
    modified_residues: dict,
    out_dir=None,
    cache_dir=None,
    msa_dir=None,
    positions=None,
    no_msa=False,
    use_constraints=False,
    create_docking_sphere=True,
):
    def get_modified_residue_yaml(chain, values):
        if not values:
            return ""
        pos, ccd = values
        return f"""
        modifications:
            - position: {pos + 1}
              ccd: "{ccd}"
        """

    def build_sequence_yaml():
        entries = []
        for chain_id, sequence in sequences.items():
            mod_res = get_modified_residue_yaml(chain_id, modified_residues.get(chain_id))
            msa_line = 'msa: empty' if no_msa else (f'msa: "{msa_dir}/{chain_id}.csv"' if msa_dir else "")
            entry = f"""
    - protein:
        id: {chain_id}
        sequence: "{sequence}"
        {mod_res}
        {msa_line}
            """.rstrip()
            entries.append(entry)
        return "\n".join(entries) + "\n" if entries else ""

    def build_hetatm_yaml():
        return "\n".join(
    f"""
    - ligand:
        smiles: "{hetatm}"
        id: {next_letter(list(sequences.keys()), nums=i)}
            """.rstrip()
            for i, hetatm in enumerate(other_hetatms)
        ) + "\n" if other_hetatms else ""

    def build_ligand_yaml():
        ligand_id = next_letter(list(sequences.keys()), nums=len(other_hetatms))
        return f"""
    - ligand:
        smiles: '{ligand_smiles}'
        id: {ligand_id}
        """

    def build_constraints_yaml():
        if not (ligand_smiles and positions and use_constraints):
            return ""
        binder_id = next_letter(list(sequences.keys()), nums=len(other_hetatms))
        contact_ids = [get_chain_for_position(pos, sequences) for pos in positions]
        return f"""
    constraints:
    - pocket:
        binder: {binder_id}
        contacts: {contact_ids}
        """

    def build_docking_sphere_yaml():
        if not create_docking_sphere:
            return ""
        docking_id = next_letter(list(sequences.keys()), nums=len(other_hetatms) + 1)
        return f"""
    - ligand:
        smiles: 'C'
        id: {docking_id}
        """

    # Build YAML
    yaml_str = f"""
        sequences:
        {build_sequence_yaml()}
        {build_hetatm_yaml()}
        {build_ligand_yaml()}
        {build_constraints_yaml()}
        version: 1
    """.strip()

    # Write to disk
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    input_yaml_path = Path(out_dir) / "input.yaml"
    input_yaml_path.write_text(yaml_str)

    # Process input and return dataloader
    process_inputs(
        data=[input_yaml_path],
        out_dir=Path(out_dir),
        ccd_path=Path(cache_dir) / "ccd.pkl",
        use_msa_server=True,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
    )
    data_module = BoltzInferenceDataModule(
        manifest=Manifest.load(Path(out_dir) / "processed" / "manifest.json"),
        target_dir=Path(out_dir) / "processed" / "structures",
        msa_dir=Path(out_dir) / "processed" / "msa",
        num_workers=1,
    )
    return next(iter(data_module.predict_dataloader()))

