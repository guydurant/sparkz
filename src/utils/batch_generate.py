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
):
    """
    Make a batch from the protein sequence and ligand information.

    Args:
        sequence (dict): Dictionary containing the protein sequence with chain ID as key.
        ligand_smiles (str): Ligand SMILES string.
        modified_residues (dict): Dictionary containing the modified residues for each chain.
        mode (str): The mode to run the model in, either 'nothing', 'inpainting', or 'guided'.
        datetime_ (str): The datetime string for the output file.
        positions (List[int]): The positions of the interacting residues



    Returns:
        torch.Tensor: The input batch for the model.
    """
    SEQUENCE_TEMPLATE = """
    - protein:
        id: {id}
        sequence: "{sequence}"
        {modified_residues}
    """
    if no_msa:
        SEQUENCE_TEMPLATE = (
            SEQUENCE_TEMPLATE
            + """
        msa: empty
    """
        )
    elif msa_dir is not None and os.path.exists(msa_dir):
        SEQUENCE_TEMPLATE = (
            SEQUENCE_TEMPLATE
            + """
        msa: "{msa}"
    """
        )
    else:
        SEQUENCE_TEMPLATE = (
            SEQUENCE_TEMPLATE
            + """
    """
        )
    MODIFIED_RESIDUE_TEMPLATE = """
        modifications:
            - position: {position}
              ccd: "{ccd}"
    """
    LIGAND_TEMPLATE = """
    - ligand:
        smiles: "{ligand}"
        id: {id}
    """
    HETATM_TEMPLATE = """
    - ligand:
        ccd: "{ligand}"
        id: {id}
    """
    CONSTRAINTS_TEMPLATE = """
    constraints:
    - pocket:
        binder: {binder}
        contacts: {contacts}
    """

    YAML_TEMPLATE = """
    sequences:
        {protein}
        {ligand}
        {constraints}
    version: 1
    """
    modified_residues_yamls = {
        chain: (
            # WARNING is 1-indexed
            MODIFIED_RESIDUE_TEMPLATE.format(position=values[0] + 1, ccd=values[1])
            if values
            else ""
        )
        for chain, values in modified_residues.items()
    }
    protein_yamls = "\n".join(
        [
            SEQUENCE_TEMPLATE.format(
                id=k,
                sequence=v,
                modified_residues=modified_residues_yamls.get(k, ""),
                msa=f"{msa_dir}/{k}.csv",
            )
            for k, v in sequences.items()
        ]
    )
    other_hetatms_yaml = "\n".join(
        [
            HETATM_TEMPLATE.format(
                id=next_letter(list(sequences.keys()), nums=i), ligand=hetatm
            )
            for i, hetatm in enumerate(other_hetatms)
        ]
    )
    if ligand_smiles is None:
        yaml = YAML_TEMPLATE.format(
            protein=protein_yamls,
            ligand="",
            constraints="",
        )
    elif ligand_smiles is not None and positions is not None and use_constraints:
        yaml = YAML_TEMPLATE.format(
            protein=protein_yamls,
            ligand=other_hetatms_yaml
            + LIGAND_TEMPLATE.format(
                id=next_letter(list(sequences.keys()), nums=len(other_hetatms)),
                ligand=ligand_smiles,
            ),
            constraints=CONSTRAINTS_TEMPLATE.format(
                binder=next_letter(list(sequences.keys()), nums=len(other_hetatms)),
                # WARNING is 1-indexed
                contacts=[get_chain_for_position(pos, sequences) for pos in positions],
            ),
        )
    else:
        yaml = YAML_TEMPLATE.format(
            protein=protein_yamls,
            ligand=other_hetatms_yaml
            + LIGAND_TEMPLATE.format(
                id=next_letter(list(sequences.keys()), nums=len(other_hetatms)),
                ligand=ligand_smiles,
            ),
            constraints="",
        )
    # make io yaml file
    Path(f"{out_dir}").mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/input.yaml", "w") as f:
        f.write(yaml)
    process_inputs(
        data=[Path(f"{out_dir}/input.yaml")],
        out_dir=Path(f"{out_dir}"),
        ccd_path=Path(f"{cache_dir}/ccd.pkl"),
        use_msa_server=True,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
    )
    data_module = BoltzInferenceDataModule(
        manifest=Manifest.load(Path(f"{out_dir}/processed/manifest.json")),
        target_dir=Path(f"{out_dir}/processed/structures"),
        msa_dir=Path(f"{out_dir}/processed/msa"),
        num_workers=1,
    )
    return next(iter(data_module.predict_dataloader()))
