import torch
import os
from pathlib import Path
from boltz.main import process_inputs
from boltz.data.types import Manifest
from boltz.data.module.inference import BoltzInferenceDataModule
from hallucination.constants import create_conversion_matrix


def get_sequence_mask(sequence, positions):
    """Get a mask for the specified positions in the sequence.

    Args:
        sequence (str): The input sequence.
        positions (list): A list of positions to mask.

    Returns:
        torch.Tensor: A tensor mask of the same length as the sequence.
    """
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


def next_letter(letter_list, nums=0):
    """Get the next letter in the alphabet after the last letter in the list, offset by nums.
    Args:
        letter_list (list): List of letters.
        nums (int, optional): Offset from the next letter. Defaults to 0.
    
    Returns:
        str: The next letter in the alphabet.
    """
    return chr(ord(letter_list[-1]) + 1 + nums)


def get_chain_for_position(pos, sequences):
    """Determine the chain and relative position given a global position and sequence lengths.
    
    Args:
        pos (int): The global position.
        sequences (dict): Dictionary of chain IDs to sequences.

    Returns:
        list: A list containing the chain ID and relative position.
    """
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
    dock=False,
):
    """Create a batch from the given sequences and ligand information.
    
    Args:
        sequences (dict): Dictionary of chain IDs to sequences.
        ligand_smiles (str): SMILES string of the ligand.
        other_hetatms (list): List of other heteroatoms (ligands) as SMILES strings.
        modified_residues (dict): Dictionary of chain IDs to tuples of (position, ccd).
        out_dir (str, optional): Output directory. Defaults to None.
        cache_dir (str, optional): Cache directory. Defaults to None.
        msa_dir (str, optional): MSA directory. Defaults to None.
        positions (list, optional): List of positions for constraints. Defaults to None.
        no_msa (bool, optional): Whether to use MSA. Defaults to False.
        use_constraints (bool, optional): Whether to use constraints. Defaults to False.
        dock (bool, optional): Whether to include a docking sphere. Defaults to False.
    
    Returns:
        dict: The generated batch.
    """
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
        if not dock:
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
        mol_dir=Path(cache_dir) / "mols.tar",
        ccd_path=Path(cache_dir) / "ccd.pkl",
        use_msa_server=True,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        boltz2=False,
    )
    data_module = BoltzInferenceDataModule(
        manifest=Manifest.load(Path(out_dir) / "processed" / "manifest.json"),
        target_dir=Path(out_dir) / "processed" / "structures",
        msa_dir=Path(out_dir) / "processed" / "msa",
        num_workers=1,
        constraints_dir=Path(out_dir) / "processed" / "constraints",
    )
    return next(iter(data_module.predict_dataloader()))

