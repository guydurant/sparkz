import torch


def create_conversion_matrix():
    """
    Create a conversion matrix to map amino acid sequences to token indices of Boltz-1x.
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    tokens = "--ARNDCQEGHILKMFPSTWYV----------_"

    # Initialize a zero matrix of size (20, 34)
    conversion_matrix = torch.zeros(len(amino_acids), len(tokens))

    # Populate the conversion matrix
    for aa_idx, aa in enumerate(amino_acids):
        token_idx = tokens.find(aa)  # Find the position in tokens
        if token_idx != -1:
            conversion_matrix[aa_idx, token_idx] = 1.0
    return conversion_matrix
