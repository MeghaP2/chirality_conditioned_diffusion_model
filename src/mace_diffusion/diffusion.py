"""
VE-SDE forward noising process, EDM preconditioning coefficients, and the two
separate sigma schedules used by this pipeline:
  - the TRAINING schedule (log-uniform coverage, sampled from randomly per
    molecule every step)
  - the SAMPLING schedule (Karras/EDM rho-spaced, a small fixed number of
    steps, front-loaded toward low sigma where fine structural detail is
    resolved)
These are deliberately different: training wants broad, cheap coverage;
sampling wants to make the most of a small, expensive, fixed step budget.
"""
import numpy as np
import torch


def compute_sigma_data_and_schedule(dataset, T=1000, device="cpu"):
    """
    sigma_data, sigma_min, sigma_max are all derived from the RAW coordinate
    scale of the actual training set (not fixed constants) — this rescales
    the whole EDM preconditioning scheme to match your data's real units (Å).
    Returns (sigma_data, sigma_min, sigma_max, sigmas).
    """
    all_pos = torch.cat([dataset[i].positions for i in range(len(dataset))], dim=0)
    raw_std = all_pos.std().item()
    print(f"raw coordinate std (Å) = {raw_std:.4f}")

    sigma_data = raw_std
    sigma_min = 0.025 * raw_std
    sigma_max = 3.0 * raw_std
    sigmas = torch.exp(torch.linspace(np.log(sigma_min), np.log(sigma_max), T)).to(device)
    return sigma_data, sigma_min, sigma_max, sigmas


def forward_noise_ve_batched(x0, batch_idx, sigmas, t=None):
    """
    x_t = x0 + sigma * z  (Variance-Exploding SDE, closed-form marginal).
    t (and therefore sigma) is sampled independently PER MOLECULE, not once
    per batch -- this gives broad noise-level coverage within a single
    training step.
    """
    num_graphs = batch_idx.max().item() + 1
    if t is None:
        t = torch.randint(0, len(sigmas), (num_graphs,), device=x0.device)
    sigma = sigmas[t][batch_idx].unsqueeze(-1)   # molecule-level sigma, broadcast to atom-level
    z = torch.randn_like(x0)
    x_t = x0 + sigma * z
    return x_t, z, sigma, t


def edm_scalings(sigma, sigma_data):
    """EDM (Karras et al.) preconditioning coefficients c_in, c_skip, c_out."""
    c_in = 1.0 / torch.sqrt(sigma**2 + sigma_data**2)
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / torch.sqrt(sigma**2 + sigma_data**2)
    return c_in, c_skip, c_out


def get_karras_sigmas(num_steps, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """
    Non-uniform sigma schedule for SAMPLING: dense near sigma_min (where fine
    structural detail is resolved), sparse near sigma_max. sigma_schedule[0]
    always equals sigma_max exactly, by construction of this formula.
    """
    ramp = torch.linspace(0, 1, num_steps, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas_out = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas_out, torch.zeros(1, device=device)])
