from boltz.model.potentials.potentials import (
    FlatBottomPotential,
    DistancePotential,
    SymmetricChainCOMPotential,
    VDWOverlapPotential,
    ConnectionsPotential,
    PoseBustersPotential,
    ChiralAtomPotential,
    StereoBondPotential,
    PlanarBondPotential,
)
import torch
from boltz.model.potentials.schedules import (
    ExponentialInterpolation,
    PiecewiseStepFunction,
)
from boltz.data import const
# from potentials.pocket_docking import DockingSpherePotential
# from potentials.pocket_docking import PocketDockingPotential

class SideChainVDWOverlapPotential(FlatBottomPotential, DistancePotential):
    """A potential for side-chain van der Waals overlap.
    
    Inherits from FlatBottomPotential and DistancePotential.
    """
    def compute_args(self, feats, parameters):
        """Compute the arguments for the side-chain VDW overlap potential.
        
        Follows the structure of VDWOverlapPotential but focuses on side-chain atoms.

        Args:
            feats (dict): The input features.
            parameters (dict): The potential parameters.

        Returns:
            tuple: A tuple containing the indexed pair, potential parameters, and any additional information.
        """
        # Valid side-chain atoms
        sidechain_mask = feats["sidechain_atom_mask"][0].bool()
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        valid_mask = (sidechain_mask & atom_pad_mask).squeeze(0)

        if valid_mask.sum() < 2:
            return None, None, None

        atom_indices = torch.arange(valid_mask.size(0), device=valid_mask.device)[valid_mask]

        # Use atom_to_token to identify residue membership
        atom_to_token = feats["atom_to_token"][0]  # shape: [num_atoms, num_residues]
        valid_atom_to_token = atom_to_token[valid_mask]  # [num_valid_atoms, num_residues]

        # VDW radii
        vdw_radii = torch.zeros(const.num_elements, dtype=torch.float32, device=valid_mask.device)
        vdw_radii[1:119] = torch.tensor(const.vdw_radii, dtype=torch.float32, device=valid_mask.device)
        atom_vdw_radii = (feats["ref_element"].float() @ vdw_radii.unsqueeze(-1)).squeeze(-1)[0][valid_mask]

        # Pair indices (upper triangle)
        n_atoms = atom_indices.shape[0]
        pair_index = torch.triu_indices(n_atoms, n_atoms, 1, device=valid_mask.device)

        # Remove pairs from the same residue
        same_residue = (valid_atom_to_token.float() @ valid_atom_to_token.T.float()) > 0
        same_residue_pairs = same_residue[pair_index[0], pair_index[1]]
        pair_index = pair_index[:, ~same_residue_pairs]

        # Lower bounds from VDW radii + buffer
        lower_bounds = atom_vdw_radii[pair_index].sum(dim=0) * (1.0 - parameters["buffer"])
        k = torch.ones_like(lower_bounds)

        # Map back to original atom indices
        indexed_pair = atom_indices[pair_index]
        return indexed_pair, (k, lower_bounds, None), None



def get_potentials(sidechain_clash=True):
    """
    Overwrites Boltz-1x function to add sidechain clash potential.
    
    Args:
        sidechain_clash (bool): Whether to include the sidechain clash potential.
    Returns:
        list: A list of instantiated potential classes with their parameters.
    """
    potentials = [
        SymmetricChainCOMPotential(
            parameters={
                'guidance_interval': 4,
                'guidance_weight': 0.5,
                'resampling_weight': 0.5,
                'buffer': ExponentialInterpolation(
                    start=1.0,
                    end=5.0,
                    alpha=-2.0
                )
            }
        ),
        VDWOverlapPotential(
            parameters={
                'guidance_interval': 5,
                'guidance_weight': PiecewiseStepFunction(
                    thresholds=[0.4],
                    values=[0.125, 0.0]
                ),
                'resampling_weight': PiecewiseStepFunction(
                    thresholds=[0.6],
                    values=[0.01, 0.0]
                ),
                'buffer': 0.225,
            }
        ),
        ConnectionsPotential(
            parameters={
                'guidance_interval': 1,
                'guidance_weight': 0.15,
                'resampling_weight': 1.0,
                'buffer': 2.0,
            }
        ),
        PoseBustersPotential(
            parameters={
                'guidance_interval': 1,
                'guidance_weight': 0.05,
                'resampling_weight': 0.1,
                'bond_buffer': 0.20,
                'angle_buffer': 0.20,
                'clash_buffer': 0.15
            }
        ),
        ChiralAtomPotential(
            parameters={
                'guidance_interval': 1,
                'guidance_weight': 0.10,
                'resampling_weight': 1.0,
                'buffer': 0.52360
            }
        ),
        StereoBondPotential(
            parameters={
                'guidance_interval': 1,
                'guidance_weight': 0.05,
                'resampling_weight': 1.0,
                'buffer': 0.52360
            }
        ),
        PlanarBondPotential(
            parameters={
                'guidance_interval': 5,
                'guidance_weight': 0.05,
                'resampling_weight': 1.0,
                'buffer': 0.26180
            }
        ),
    ]
    if sidechain_clash:
        potentials.append(
            SideChainVDWOverlapPotential(
                parameters={
                    'guidance_interval': 1,
                    'guidance_weight': 0.05,
                    'resampling_weight': 0.1,
                    'buffer': 0.37,
                }
            )
        )
    return potentials
