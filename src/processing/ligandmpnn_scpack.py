import torch
import os
import urllib.request
import numpy as np
import copy

from ligandmpnn.sc_utils import Packer, pack_side_chains
from ligandmpnn.data_utils import (
    alphabet,
    element_dict_rev,
    featurize,
    get_score,
    get_seq_rec,
    parse_PDB,
    restype_1to3,
    restype_int_to_str,
    restype_str_to_int,
    write_full_PDB,
)

restype_3to1 = {
"ALA": "A",
"ARG": "R",
"ASN": "N",
"ASP": "D",
"CYS": "C",
"GLN": "Q",
"GLU": "E",
"GLY": "G",
"HIS": "H",
"ILE": "I",
"LEU": "L",
"LYS": "K",
"MET": "M",
"PHE": "F",
"PRO": "P",
"SER": "S",
"THR": "T",
"TRP": "W",
"TYR": "Y",
"VAL": "V",
}



def get_weights(cache):
    url = "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_sc_v_32_002_16.pt"
    os.makedirs(cache, exist_ok=True)
    dest_path = os.path.join(cache, "ligandmpnn_sc_v_32_002_16.pt")
    if not os.path.isfile(dest_path):
        urllib.request.urlretrieve(url, dest_path)
    return dest_path

def get_scmodel(cache):
    get_weights(cache)
    sc_params = {
        "sc_num_denoising_steps": 3,
        "sc_num_samples": 16,
        "repack_everything": 0,
        "number_of_packs_per_design": 1,
    }

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    model_sc = Packer(
        node_features=128,
        edge_features=128,
        num_positional_embeddings=16,
        num_chain_embeddings=16,
        num_rbf=16,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        atom_context_num=16,
        lower_bound=0.0,
        upper_bound=20.0,
        top_k=32,
        dropout=0.0,
        augment_eps=0.0,
        atom37_order=False,
        device=device,
        num_mix=3,
    )

    checkpoint_sc = torch.load(
        f"{cache}/ligandmpnn_sc_v_32_002_16.pt",
        map_location=device,
    )
    model_sc.load_state_dict(checkpoint_sc["model_state_dict"])
    model_sc.to(device)
    model_sc.eval()
    return model_sc, sc_params, device
    
def get_protein_dict(pdb_path, device, fixed_residues=[]):
    parse_all_atoms_flag = True
    parse_these_chains_only_list = []
    protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
        pdb_path,
        device=device,
        chains=parse_these_chains_only_list,
        parse_all_atoms=parse_all_atoms_flag,
        parse_atoms_with_zero_occupancy=1,
    )
    # make chain_letter + residue_idx + insertion_code mapping to integers
    R_idx_list = list(protein_dict["R_idx"].cpu().numpy())  # residue indices
    chain_letters_list = list(protein_dict["chain_letters"])  # chain letters
    encoded_residues = []
    for i, R_idx_item in enumerate(R_idx_list):
        tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
        encoded_residues.append(tmp)
    fixed_positions = torch.tensor(
        [int(item not in fixed_residues) for item in encoded_residues],
        device=device,
    )
    interface_positions = torch.zeros_like(fixed_positions)
    buried_positions = torch.zeros_like(fixed_positions)
    protein_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
        1 - interface_positions
    ) + 1 * interface_positions * (1 - buried_positions)
    chains_to_design_list = protein_dict["chain_letters"]
    chain_mask = torch.tensor(
        np.array(
            [item in chains_to_design_list for item in protein_dict["chain_letters"]],
            dtype=np.int32,
        ),
        device=device,
    )
    protein_dict["chain_mask"] = chain_mask * fixed_positions
    return protein_dict, other_atoms, icodes

def get_ligandmpnn_sc_prior(
    input_file, fixed_residues, cache=".cache", outfile="test_packed.pdb"
):
    model_sc, sc_params, device = get_scmodel(cache)
    protein_dict, other_atoms, icodes = get_protein_dict(input_file, device, fixed_residues=fixed_residues)
    length = len(protein_dict["R_idx"])
    original_S = protein_dict["S"].to(device)
    protein_dict["S"] = original_S.view(1, length)
    # for key in modified_residues.keys():
    #     protein_dict["S"][0, key] = modified_residues[key]
    S_list = [protein_dict["S"].long().to(device)]
    protein_dict["S"] = protein_dict["S"].long().to(device)
    feature_dict_ = featurize(
        protein_dict,
        cutoff_for_score=8.0,
        use_atom_context=1,
        number_of_ligand_atoms=16,
        model_type="ligand_mpnn",
    )
    sc_feature_dict = copy.deepcopy(feature_dict_)
    B = 1

    for k, v in sc_feature_dict.items():
        if k != "S":
            try:
                num_dim = len(v.shape)
                if num_dim == 2:
                    sc_feature_dict[k] = v.repeat(B, 1)
                elif num_dim == 3:
                    sc_feature_dict[k] = v.repeat(B, 1, 1)
                elif num_dim == 4:
                    sc_feature_dict[k] = v.repeat(B, 1, 1, 1)
                elif num_dim == 5:
                    sc_feature_dict[k] = v.repeat(B, 1, 1, 1, 1)
            except:
                pass
    X_stack_list = []
    X_m_stack_list = []
    b_factor_stack_list = []
    for c_pack in range(sc_params["number_of_packs_per_design"]):
        X_list = []
        X_m_list = []
        b_factor_list = []
        for c in range(B):
            sc_feature_dict["S"] = S_list[c]
            sc_dict = pack_side_chains(
                sc_feature_dict,
                model_sc,
                sc_params["sc_num_denoising_steps"],
                num_samples=sc_params["sc_num_samples"],
                repack_everything=sc_params["repack_everything"],
            )
            X_list.append(sc_dict["X"])
            X_m_list.append(sc_dict["X_m"])
            b_factor_list.append(sc_dict["b_factors"])

        X_stack = torch.cat(X_list, 0)
        X_m_stack = torch.cat(X_m_list, 0)
        b_factor_stack = torch.cat(b_factor_list, 0)

        X_stack_list.append(X_stack)
        X_m_stack_list.append(X_m_stack)
        b_factor_stack_list.append(b_factor_stack)
        X_stack = X_stack_list[c_pack]
        X_m_stack = X_m_stack_list[c_pack]
        b_factor_stack = b_factor_stack_list[c_pack]

        write_full_PDB(
            outfile,
            X_stack[0].cpu().numpy(),
            X_m_stack[0].cpu().numpy(),
            b_factor_stack[0].cpu().detach().numpy(),
            protein_dict["R_idx"][None,][0].cpu().numpy(),
            protein_dict["chain_letters"],
            S_list[0][0].cpu().numpy(),
            other_atoms=other_atoms,
            icodes=icodes,
            force_hetatm=0,
        )