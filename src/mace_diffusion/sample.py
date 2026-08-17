"""
Deterministic (no stochasticity in the integration loop -- only the initial
noise draw is random) Heun sampler over EDM's probability flow ODE, conditioned
on a chirality label y and a template structure (topology + atom count only;
this is a CONFORMER generator, not a de novo structure generator -- see
project README for what that distinction means and what would need to change).
"""
import torch
from mace.tools.torch_geometric import Batch

from .diffusion import get_karras_sigmas, edm_scalings


@torch.no_grad()
def sample_mace_heun(wrapper, template_graph, sigmas, sigma_data,
                      sample_sigma_min, sample_sigma_max, y, num_steps=500):
    device = template_graph.positions.device
    x0_template = template_graph.positions
    N = x0_template.shape[0]
    y = y.to(device).unsqueeze(0) if y.dim() == 1 else y.to(device)

    sigma_schedule = get_karras_sigmas(num_steps, sample_sigma_min, sample_sigma_max, device=device)
    x = x0_template + torch.randn_like(x0_template) * sigma_schedule[0]   # sigma_schedule[0] == sample_sigma_max exactly
    x = x - x.mean(dim=0, keepdim=True)

    # batch-of-one: forward() universally expects a `batch` tracking tensor,
    # which only exists on Batch objects, not raw AtomicData
    g_batch = Batch.from_data_list([template_graph]).to(device)

    for i in range(num_steps):
        sigma_cur = sigma_schedule[i]
        sigma_next = sigma_schedule[i + 1]

        # translate the continuous Karras sigma to its nearest TRAINING-schedule
        # index -- time_embed only ever learned to interpret training indices,
        # even though the actual trajectory math below uses sigma_cur directly
        t_idx_cur = torch.argmin((sigmas - sigma_cur).abs()).unsqueeze(0)
        t_per_atom_cur = t_idx_cur.expand(N)
        sigma_t = torch.full((N, 1), sigma_cur, device=device)

        c_in, c_skip, c_out = edm_scalings(sigma_t, sigma_data)
        x_in = c_in * x
        F, edge_index = wrapper(g_batch.to_dict(), x_in, t_per_atom_cur, sigma_t, x0_template, y)
        x0_pred = c_skip * x + c_out * F
        d_cur = (x - x0_pred) / sigma_cur
        x_next = x + (sigma_next - sigma_cur) * d_cur   # Euler predictor step

        if sigma_next > 0:
            t_idx_next = torch.argmin((sigmas - sigma_next).abs()).unsqueeze(0)
            t_per_atom_next = t_idx_next.expand(N)
            sigma_t_next = torch.full((N, 1), sigma_next, device=device)
            c_in2, c_skip2, c_out2 = edm_scalings(sigma_t_next, sigma_data)
            x_in2 = c_in2 * x_next
            F2, _ = wrapper(g_batch.to_dict(), x_in2, t_per_atom_next, sigma_t_next, x0_template, y)
            x0_pred2 = c_skip2 * x_next + c_out2 * F2
            d_next = (x_next - x0_pred2) / sigma_next
            x_next = x + (sigma_next - sigma_cur) * 0.5 * (d_cur + d_next)   # Heun corrector step

        x = x_next - x_next.mean(dim=0, keepdim=True)

    return x


def save_xyz(N, coords, atomtypes, filename):
    """atomtypes: 0/1 encoding (0 = H, 1 = C), matching atom_type_dir."""
    with open(filename, "w") as f:
        f.write(f"{N}\n")
        f.write("genAI structure\n")
        for i in range(2, N + 2):
            x, y, z = coords[i - 2]
            if atomtypes[i - 2].item() == 0:
                f.write(f"H \t{x}\t{y}\t{z}\n")
            elif atomtypes[i - 2].item() == 1:
                f.write(f"C \t{x}\t{y}\t{z}\n")
