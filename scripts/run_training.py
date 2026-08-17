"""
End-to-end training entry point:
  data_pipeline outputs (chirality_data.csv) -> dataset -> model -> train -> save checkpoint
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `configs`/`src` import cleanly

from mace_diffusion import env_setup   # noqa: E402  (must be imported first)
from mace_diffusion.dataset import load_dataset
from mace_diffusion.diffusion import compute_sigma_data_and_schedule
from mace_diffusion.model import build_mace_model, MaceDiffusionWrapper
from mace_diffusion.train import compute_avg_num_neighbors, train
from mace_diffusion.checkpoint import save_checkpoint
from mace.tools.torch_geometric import DataLoader
import numpy as np
import torch

from configs import config as cfg

device = env_setup.device
print(f"Using device: {device}")

# ---- data ----
dataset, files_data = load_dataset("chirality_data.csv", large_cutoff=cfg.LARGE_CUTOFF)
loader = DataLoader(dataset, batch_size=12, shuffle=True)

sigma_data, sigma_min, sigma_max, sigmas = compute_sigma_data_and_schedule(dataset, T=1000, device=device)

# ---- model ----
avg_num_neighbors = compute_avg_num_neighbors(dataset, cfg.R_MAX)
atomic_energies = np.zeros(cfg.NUM_ELEMENTS)

mace_model = build_mace_model(cfg, avg_num_neighbors, atomic_energies, device)
wrapper = MaceDiffusionWrapper(
    mace_model, y_dim=cfg.Y_DIM, cond_dim=cfg.COND_DIM,
    r_max=cfg.R_MAX, num_layers=cfg.NUM_LAYERS,
).to(device)

# ---- train ----
optimizer = torch.optim.Adam(wrapper.parameters(), lr=1e-3, weight_decay=0.0)
NUM_EPOCHS = 50

print("----------------- training begins -----------------")
wrapper = train(wrapper, loader, sigmas, sigma_data, optimizer, num_epochs=NUM_EPOCHS)
print("----------------- training ends -----------------")

# ---- save ----
mace_cfg = dict(
    r_max=cfg.R_MAX, num_bessel=cfg.NUM_BESSEL, num_polynomial_cutoff=cfg.NUM_POLYNOMIAL_CUTOFF,
    max_ell=cfg.MAX_ELL, interaction_cls=cfg.INTERACTION_CLS, interaction_cls_first=cfg.INTERACTION_CLS_FIRST,
    num_interactions=cfg.NUM_LAYERS, num_elements=cfg.NUM_ELEMENTS, hidden_irreps=cfg.HIDDEN_IRREPS,
    MLP_irreps=cfg.MLP_IRREPS, atomic_energies=atomic_energies, avg_num_neighbors=avg_num_neighbors,
    atomic_numbers=cfg.ATOMIC_NUMBERS, correlation=cfg.CORRELATION, gate=cfg.GATE, radial_MLP=cfg.RADIAL_MLP,
)
diffusion_cfg = dict(
    y_dim=cfg.Y_DIM, cond_dim=cfg.COND_DIM, sigma_min=sigma_min, sigma_max=sigma_max,
    sigma_data=sigma_data, T=1000,
)
save_checkpoint("checkpoints/mace_diffusion_checkpoint.pt", wrapper, optimizer, NUM_EPOCHS - 1, mace_cfg, diffusion_cfg)
print("checkpoint saved to checkpoints/mace_diffusion_checkpoint.pt")
