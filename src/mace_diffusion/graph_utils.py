"""
Brute-force radius graph construction. NOT used in the main forward pass
(MaceDiffusionWrapper.forward() instead filters a precomputed large-cutoff
edge list at every noise level, which is cheaper than rebuilding the graph
from scratch every call) — kept here for debugging / diagnostic use.
"""
import torch


def radius_graph_manual(pos, r, batch, loop=False):
    """
    Fine for small molecules (dozens of atoms); would not scale to large
    systems, but that's not the use case here.
    """
    dist_matrix = torch.cdist(pos, pos)                     # (N, N) pairwise distances
    same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)     # only connect within same molecule
    within_cutoff = dist_matrix <= r
    mask = within_cutoff & same_graph
    if not loop:
        mask.fill_diagonal_(False)
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    return edge_index
