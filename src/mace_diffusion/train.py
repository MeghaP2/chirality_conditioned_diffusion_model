"""
F-space EDM loss (train the network to predict the specific rescaled residual
that reconstructs x0 exactly, given c_skip/c_out -- not the noise, not x0
directly), plus the training loop and an auxiliary angle-error diagnostic.
"""
import torch

from .diffusion import forward_noise_ve_batched, edm_scalings


def compute_avg_num_neighbors(dataset, r_max):
    """
    Dataset-derived statistic (NOT a fixed hyperparameter) used by MACE to
    normalize aggregated neighbor messages. Must be recomputed (or loaded from
    a checkpoint) for any new dataset / r_max combination.
    """
    total_edges = 0
    total_atoms = 0
    for i in range(len(dataset)):
        g = dataset[i]
        lengths = g.edge_vec_clean.norm(dim=-1)
        total_edges += (lengths <= r_max).sum().item()
        total_atoms += g.positions.shape[0]
    avg = total_edges / total_atoms
    print(f"avg_num_neighbors = {avg:.2f}")
    return avg


def compute_mace_diffusion_loss(wrapper, batch, sigmas, sigma_data):
    x0 = batch.positions
    batch_idx = batch.batch
    y = batch.y   # (num_graphs, 5)

    x_t, z, sigma, t = forward_noise_ve_batched(x0, batch_idx, sigmas)
    c_in, c_skip, c_out = edm_scalings(sigma, sigma_data)

    x_in = c_in * x_t
    t_per_atom = t[batch_idx]
    F, _ = wrapper(batch.to_dict(), x_in, t_per_atom, sigma, x0, y)

    F_target = (x0 - c_skip * x_t) / c_out   # solved from D_theta = c_skip*x_t + c_out*F = x0
    loss = ((F - F_target) ** 2).mean()
    return loss, t


def compute_angle_error(coords, angle_index, true_angles_deg):
    """Mean absolute angle error, in degrees, between generated and true structure."""
    if angle_index.size(1) == 0:
        return float("nan")
    a, center, b = angle_index
    v1 = coords[a] - coords[center]
    v2 = coords[b] - coords[center]
    cos_angle = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-8)
    cos_angle = cos_angle.clamp(-1 + 1e-6, 1 - 1e-6)
    pred_angles_deg = cos_angle.arccos() * 180 / torch.pi
    return (pred_angles_deg - true_angles_deg).abs().mean().item()


def train(wrapper, loader, sigmas, sigma_data, optimizer, num_epochs, log_every=10):
    wrapper.train()
    for epoch in range(num_epochs):
        buckets = {"low": [], "mid": [], "high": []}
        for batch in loader:
            batch = batch.to(next(wrapper.parameters()).device)
            loss, t = compute_mace_diffusion_loss(wrapper, batch, sigmas, sigma_data)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm=1.0)
            optimizer.step()

            for t_i in t.tolist():
                if t_i <= 50:
                    buckets["low"].append(loss.item())
                elif t_i <= 500:
                    buckets["mid"].append(loss.item())
                else:
                    buckets["high"].append(loss.item())

        if epoch % log_every == 0:
            for name, vals in buckets.items():
                if vals:
                    print(f"epoch {epoch} [{name}] n={len(vals)} mean_loss={sum(vals) / len(vals):.4f}")

    return wrapper
