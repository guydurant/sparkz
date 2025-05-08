import torch
from hallucination.constants import create_conversion_matrix


def adjust_batch_from_sequence(sequence, batch):
    """
    Adjust the batch with the new protein sequence.

    Args:
        sequence (str)): The protein sequence (one-hot encoded).
        batch (torch.Tensor): The input batch for the model.
        
    
    Returns:
        torch.Tensor: The input batch for the model.
    """
    conversion_table = create_conversion_matrix()
    new_sequence = torch.matmul(sequence, conversion_table).unsqueeze(0)
    batch['res_type'] = batch['res_type'].clone()
    batch['res_type'][0, :new_sequence.size(1)] = new_sequence
    return batch

def one_hot_encode_sequence(sequence: str) -> torch.Tensor:
    """
    Convert an amino acid sequence into a one-hot encoded PyTorch tensor.

    Args:
        sequence (str): The amino acid sequence (e.g., "ACDEFGHIKLMNPQRSTVWY").

    Returns:
        torch.Tensor: A one-hot encoded tensor of shape (sequence_length, 20).
    """
    # Define the standard amino acids and their corresponding indices
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}

    # Initialize a zero tensor of shape (sequence_length, 20)
    sequence_length = len(sequence)
    one_hot = torch.zeros(sequence_length, 20)

    # Fill the tensor with 1s at the appropriate positions
    for i, aa in enumerate(sequence):
        if aa in aa_to_index:
            one_hot[i, aa_to_index[aa]] = 1.0
        else:
            # Handle unknown amino acids (e.g., gaps or non-standard residues)
            raise ValueError(f"Unknown amino acid: {aa}")

    return one_hot


def decode_sampled_one_hot(one_hot):
    """
    Decode a one-hot encoded tensor into an amino acid sequence.

    Args:
        one_hot (torch.Tensor): A one-hot encoded tensor of shape (sequence_length, 20).

    Returns:
        str: The decoded amino acid sequence.
    """
    # Define the standard amino acids
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    # Find the index of the maximum value for each position
    indices = torch.argmax(one_hot, dim=1)

    # Convert the indices to amino acids
    sequence = "".join([amino_acids[i] for i in indices])

    return sequence
