from boltz.model.modules.confidence_utils import compute_aggregated_metric

def bindcraft_like_loss(model_outputs, protein_index, pocket_residues):
    loss_components = {"pocket_plddt": 0.33, "ligand_iptm":0.33, "pocket_pae":-0.33}#, "ipae":0.1}
    loss = 0
    for loss_component in loss_components:
        if loss_component == "pocket_pae":
            loss+=loss_components[loss_component]*compute_aggregated_metric(
                model_outputs["pae_logits"][:, pocket_residues, :]
                ).mean()/31.75
        elif loss_component == "pocket_plddt":
            loss+=loss_components[loss_component]*compute_aggregated_metric(
                model_outputs["plddt_logits"][:, pocket_residues, :]
                ).mean()
        else:
            loss+=loss_components[loss_component]*model_outputs[loss_component].mean()
    return loss

def pocket_plddt_loss(model_outputs):
    return model_outputs["pocket_plddt"]