"""
Checkpoint save/load. Saves the FULL set of values needed to reconstruct the
vanilla MACE model and the wrapper exactly, not just the weights -- avoids
the silent-mismatch risk of hardcoding architecture args from memory at load
time (see e.g. avg_num_neighbors, which is a dataset-derived statistic, not a
fixed hyperparameter, and must travel with the checkpoint).
"""
import torch
from e3nn import o3


def save_checkpoint(path, wrapper, optimizer, epoch, mace_cfg: dict, diffusion_cfg: dict):
    """
    mace_cfg: every keyword argument used to construct the vanilla MACE model
      (r_max, num_bessel, ..., avg_num_neighbors, atomic_energies, correlation,
      gate, radial_MLP, atomic_numbers, hidden_irreps as a STRING).
    diffusion_cfg: y_dim, cond_dim, num_layers, sigma_min, sigma_max, sigma_data, T.
    """
    checkpoint = {
        "model_state_dict": wrapper.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "mace_config": {**mace_cfg, "hidden_irreps": str(mace_cfg["hidden_irreps"])},
        "diffusion_config": diffusion_cfg,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, device):
    """Returns the raw checkpoint dict. Use rebuild_from_checkpoint() to get a
    ready-to-use (mace_model, wrapper) pair."""
    return torch.load(path, map_location=device)


def rebuild_from_checkpoint(checkpoint, device):
    from .model import build_mace_model, MaceDiffusionWrapper

    mc = dict(checkpoint["mace_config"])
    mc["hidden_irreps"] = o3.Irreps(mc["hidden_irreps"])
    dc = checkpoint["diffusion_config"]

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.R_MAX = mc["r_max"]
    cfg.NUM_BESSEL = mc["num_bessel"]
    cfg.NUM_POLYNOMIAL_CUTOFF = mc["num_polynomial_cutoff"]
    cfg.MAX_ELL = mc["max_ell"]
    cfg.INTERACTION_CLS = mc["interaction_cls"]
    cfg.INTERACTION_CLS_FIRST = mc["interaction_cls_first"]
    cfg.NUM_LAYERS = mc["num_interactions"]
    cfg.NUM_ELEMENTS = mc["num_elements"]
    cfg.HIDDEN_IRREPS = mc["hidden_irreps"]
    cfg.MLP_IRREPS = mc["MLP_irreps"]
    cfg.ATOMIC_NUMBERS = mc["atomic_numbers"]
    cfg.CORRELATION = mc["correlation"]
    cfg.GATE = mc["gate"]
    cfg.RADIAL_MLP = mc["radial_MLP"]

    mace_model = build_mace_model(cfg, mc["avg_num_neighbors"], mc["atomic_energies"], device)

    wrapper = MaceDiffusionWrapper(
        mace_model, y_dim=dc["y_dim"], cond_dim=dc["cond_dim"],
        r_max=mc["r_max"], num_layers=mc["num_interactions"],
    ).to(device)
    wrapper.load_state_dict(checkpoint["model_state_dict"])
    return mace_model, wrapper
