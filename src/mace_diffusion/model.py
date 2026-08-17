"""
Architecture: wraps a vanilla (energy-model) MACE instance, replacing its
energy readout with an equivariant 1x1o vector readout (predicted noise per
atom), and adds two separate conditioning pathways:
  - timestep/noise-level, injected by ADDITION into the scalar node features
    (standard, DDPM-style)
  - chirality label y, injected by CONCATENATION + learned projection (more
    expressive than addition -- lets the projection learn arbitrary per-
    dimension combinations, at the cost of needing the concat_dim -> scalar_dim
    projection layer)
"""
import torch
import torch.nn as nn
from e3nn import o3

from configs.config import R_MAX


class TimeEmbedding(nn.Module):
    """Sinusoidal (Transformer-style) timestep embedding + small MLP."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device) * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = t.float()[:, None] * freqs[None, :]   # outer product via broadcasting
        return self.mlp(torch.cat([torch.sin(args), torch.cos(args)], dim=-1))


class MaceVectorReadout(nn.Module):
    """MACE's 'representative to the outside world': collapses the internal,
    multi-irrep, multi-layer feature representation down to one clean 1x1o
    (predicted noise) vector per atom, via an equivariant linear combination."""

    def __init__(self, in_irreps):
        super().__init__()
        # o3.Linear, not nn.Linear -- only mixes components of matching irrep
        # type, so equivariance is preserved (a plain nn.Linear would break it)
        self.readout = o3.Linear(o3.Irreps(in_irreps), o3.Irreps("1x1o"))

    def forward(self, node_feats):
        return self.readout(node_feats)


class MaceDiffusionWrapper(nn.Module):
    def __init__(self, mace_model, y_dim, cond_dim, r_max=R_MAX, num_layers=None):
        super().__init__()
        self.mace = mace_model
        self.r_max = r_max

        scalar_dim = mace_model.node_embedding.linear.irreps_out.count(o3.Irrep(0, 1))  # 0e count
        self.time_embed = TimeEmbedding(scalar_dim)

        # separate conditioning MLP, not summed into time_embed
        self.y_embed = nn.Sequential(
            nn.Linear(y_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        # concat [node_scalar_feats, y_cond] -> back down to scalar_dim
        self.cond_proj = nn.Linear(scalar_dim + cond_dim, scalar_dim)

        n_layers = num_layers if num_layers is not None else len(mace_model.interactions)
        hidden_irreps = mace_model.interactions[0].hidden_irreps
        scalar_only_irreps = o3.Irreps(str(hidden_irreps[0]))
        # readout sees full irreps from every layer except the last, and only
        # the scalar (0e) contribution from the last layer
        concat_irreps = hidden_irreps * (n_layers - 1) + scalar_only_irreps
        self.vector_readout = MaceVectorReadout(concat_irreps)

    def forward(self, data, x_t, t_per_atom, sigma, x0, y):
        node_attrs = data["node_attrs"]
        batch_idx = data["batch"]
        edge_index_large = data["edge_index"]
        edge_vec_clean = data["edge_vec_clean"]

        # incremental noisy edge-vector update: avoids recomputing from scratch
        dx = x_t - x0
        i, j = edge_index_large
        edge_vec_noisy = edge_vec_clean + dx[j] - dx[i]
        lengths_all = edge_vec_noisy.norm(dim=-1)

        # dynamic filtering: which candidate edges are actually within r_max
        # AT THIS NOISE LEVEL (connectivity changes as atoms get displaced)
        mask = lengths_all <= self.r_max
        edge_index = edge_index_large[:, mask]
        vectors = edge_vec_noisy[mask]
        lengths = lengths_all[mask].unsqueeze(-1)

        node_feats = self.mace.node_embedding(node_attrs) + self.time_embed(t_per_atom)

        # --- separate conditioning injection (concat + projection, not addition) ---
        y_emb = self.y_embed(y)                # (num_graphs, cond_dim)
        y_emb_per_atom = y_emb[batch_idx]        # (N, cond_dim) -- broadcast graph label to its atoms
        node_feats = self.cond_proj(torch.cat([node_feats, y_emb_per_atom], dim=-1))
        # -----------------------------------------------------------------------------

        edge_attrs = self.mace.spherical_harmonics(vectors)
        edge_feats, cutoff = self.mace.radial_embedding(lengths, node_attrs, edge_index, self.mace.atomic_numbers)

        node_feats_concat = []
        for idx_layer, (interaction, product) in enumerate(zip(self.mace.interactions, self.mace.products)):
            node_feats, sc = interaction(
                node_attrs=node_attrs, node_feats=node_feats, edge_attrs=edge_attrs,
                edge_feats=edge_feats, edge_index=edge_index, cutoff=cutoff,
                first_layer=(idx_layer == 0), lammps_class=None, lammps_natoms=None,
            )
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=node_attrs)
            node_feats_concat.append(node_feats)

        node_feats_out = torch.cat(node_feats_concat, dim=-1)   # multi-layer, multi-scale readout input
        return self.vector_readout(node_feats_out), edge_index


def build_mace_model(cfg, avg_num_neighbors, atomic_energies, device):
    """
    Constructs the vanilla (energy-model) MACE backbone. avg_num_neighbors and
    atomic_energies are dataset-derived and must be supplied explicitly (see
    train.py's compute_avg_num_neighbors) -- they are NOT fixed hyperparameters.
    """
    from mace.modules.models import MACE

    mace_model = MACE(
        r_max=cfg.R_MAX,
        num_bessel=cfg.NUM_BESSEL,
        num_polynomial_cutoff=cfg.NUM_POLYNOMIAL_CUTOFF,
        max_ell=cfg.MAX_ELL,
        interaction_cls=cfg.INTERACTION_CLS,
        interaction_cls_first=cfg.INTERACTION_CLS_FIRST,
        num_interactions=cfg.NUM_LAYERS,
        num_elements=cfg.NUM_ELEMENTS,
        hidden_irreps=cfg.HIDDEN_IRREPS,
        MLP_irreps=cfg.MLP_IRREPS,
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=cfg.ATOMIC_NUMBERS,
        correlation=cfg.CORRELATION,
        gate=cfg.GATE,
        radial_MLP=cfg.RADIAL_MLP,
    )
    return mace_model.to(device)
