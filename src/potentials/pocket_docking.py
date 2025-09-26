# from boltz.model.potentials.potentials import (
#     FlatBottomPotential,
#     DistancePotential,
#     SymmetricChainCOMPotential,
#     VDWOverlapPotential,
#     ConnectionsPotential,
#     PoseBustersPotential,
#     ChiralAtomPotential,
#     StereoBondPotential,
#     PlanarBondPotential,
# )
# import torch
# from boltz.model.potentials.schedules import (
#     ExponentialInterpolation,
#     PiecewiseStepFunction,
# )
# from boltz.data import const

# class DockingSpherePotential(FlatBottomPotential, DistancePotential):
#     def compute_args(self, feats, parameters):
#         ligand_atom_index = torch.nonzero(feats['ligand_atom_mask'][0], as_tuple=False).squeeze(-1)

#         # Construct pair_index between each ligand atom and the -1 index (target atom)
#         # Assuming batch size is handled elsewhere or is 1
#         target_index = feats['atom_pad_mask'].shape[1] - 1  # last index
#         target_indices = torch.full_like(ligand_atom_index, target_index)  # same shape as ligand_atom_index

#         # Shape: [2, N_ligand_atoms] where row 0 = ligand, row 1 = -1 (target)
#         pair_index = torch.stack([ligand_atom_index, target_indices], dim=0)

#         lower_bounds = None
#         upper_bounds = torch.full((pair_index.shape[1],), parameters['buffer'], device=pair_index.device)
#         k = torch.ones_like(upper_bounds)
#         return pair_index, (k, lower_bounds, upper_bounds), None
        
        