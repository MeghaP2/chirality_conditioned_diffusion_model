"""
F-space EDM loss (train the network to predict the specific rescaled residual
that reconstructs x0 exactly, given c_skip/c_out -- not the noise, not x0
directly), plus the training loop and an auxiliary angle-error diagnostic.
"""
import torch

from .diffusion import forward_noise_ve_batched, edm_scalings
from .dataset import unit_signed_volume

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


def compute_mace_diffusion_loss(wrapper, batch, sigmas, sigma_data, bond_weight=0.1,chirality_weight=0.5):
    x0 = batch.positions
    batch_idx = batch.batch
    y = batch.y   # (num_graphs, 5)

    x_t, z, sigma, t = forward_noise_ve_batched(x0, batch_idx, sigmas)
    c_in, c_skip, c_out = edm_scalings(sigma, sigma_data)

    x_in = c_in * x_t
    t_per_atom = t[batch_idx]
    F, _ = wrapper(batch.to_dict(), x_in, t_per_atom, sigma, x0, y)

    F_target = (x0 - c_skip * x_t) / c_out   # solved from D_theta = c_skip*x_t + c_out*F = x0
    coord_loss = ((F - F_target) ** 2).mean()

    i,j=batch.bond_edge_index
    x0_pred=F*c_out+x_t*skip
    d0_pred=(x0_pred[j]-x0_pred[i]).norm(dim=-1)
    d0=(x0[j]-x0[i]).norm(dim=-1)
    gate=(sigma_data**2) / (sigma[i].squeeze(-1)**2 + sigma_data**2)
    bond_loss=gate*((d0-d0_pred)**2).mean()

    q=batch.chiral_index
    if q.shape[1]>0:
        V=unit_signed_volume(x0_pred,q)
        h=torch.sign(y[batch_idx[q[0]]]).squeeze(-1)
        chirality_loss=nn.functional.softplus(-h*V/0.1).mean()
    else:
        chirality_loss=x0.new_zeros(())

    loss=coord_loss + bond_loss*bond_weight + chirality_loss*chirality_weight
        
    return loss , t, chirality_loss.detach()


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

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

num_epochs = 100
steps_per_epoch = len(loader)
total_steps = num_epochs * steps_per_epoch
warmup_steps = 3 * steps_per_epoch

#optimizer = torch.optim.AdamW(wrapper.parameters(), lr=3e-4)
chiral_params = list(wrapper.chiral_scale.parameters())
chiral_ids = {id(p) for p in chiral_params}
decay = [p for p in wrapper.parameters() if id(p) not in chiral_ids]
optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                               {"params": chiral_params, "weight_decay": 0.0}], lr=3e-4)
scheduler = SequentialLR(
    optimizer,
    schedulers=[
        LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6),
    ],
    milestones=[warmup_steps],
)

def train(wrapper, loader, sigmas, sigma_data, optimizer, num_epochs, log_every=10):
    wrapper.train()
    for epoch in range(num_epochs):

        # ---- train ----
        wrapper.train()
        total_loss, total_chi, n_batch = 0.0, 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            loss, t, chi = compute_mace_diffusion_loss(wrapper, batch, sigmas, sigma_data)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()                      # per batch, after optimizer.step()
            total_loss += loss.item()
            total_chi+=chi.item()
            n_batch += 1
        train_loss = total_loss / n_batch
        train_chi=total_chi/n_batch
    
        # ---- validate ----
        wrapper.eval()
        val_total, val_n = 0.0, 0
        with torch.enable_grad():                 # keep only if the model needs autograd internally
            for batch in val_loader:
                batch = batch.to(device)
                loss, _,_ = compute_mace_diffusion_loss(wrapper, batch, sigmas, sigma_data)
                val_total += loss.item()
                val_n += 1
        val_loss = val_total / val_n
    
        train_hist.append(train_loss)
        val_hist.append(val_loss)

    return wrapper
