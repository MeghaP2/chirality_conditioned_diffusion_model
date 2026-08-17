"""
.mol2 parsing and PyTorch Dataset construction.

Produces AtomicData objects carrying:
  - a large-cutoff candidate edge list (edge_index), built ONCE per structure,
    filtered down to the real r_max at every noise level inside the model's
    forward() — never recomputed from scratch during training/sampling.
  - clean edge vectors (edge_vec_clean), for the incremental noisy-edge-vector
    update trick used in the wrapper's forward().
  - bond/angle/dihedral connectivity, derived from the file's explicit bond
    list (chemical topology — distinct from the geometric, distance-based
    edge_index).
  - a per-molecule chirality label (y).
"""
from collections import defaultdict

import numpy as np
import torch
from mace.data import Configuration, AtomicData

from configs.config import atom_type_dir, element_to_z, z_table, LARGE_CUTOFF


class NodEmbedding:
    """Manual .mol2 parser (no ASE/RDKit dependency)."""

    def node_embedding(self, file_name):
        atom_type_list, coords, bond_edges = [], [], []
        with open(file_name, "r") as f:
            lines = f.readlines()
            total_atoms = int(lines[3].split()[0])
            total_bonds = int(lines[3].split()[1])
            bds = 4 + total_atoms
            for i in range(4, 4 + total_atoms):
                fields = lines[i].split()
                atom_type_list.append(atom_type_dir[fields[3]])
                coords.append([float(fields[0]), float(fields[1]), float(fields[2])])
            for j in range(bds, bds + total_bonds):
                fields = lines[j].split()
                src, dst = int(fields[0]) - 1, int(fields[1]) - 1
                bond_edges.append([src, dst])
                bond_edges.append([dst, src])
        return atom_type_list, coords, bond_edges


def build_angle_index(bond_edge_index):
    """For every atom, form all (neighbor, center, neighbor) triples from its
    bonded neighbors -> used to compute true bond angles as an auxiliary loss."""
    neighbors = defaultdict(list)
    src, dst = bond_edge_index
    for s, d in zip(src.tolist(), dst.tolist()):
        neighbors[s].append(d)
    triples = []
    for center, nbrs in neighbors.items():
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                triples.append([nbrs[i], center, nbrs[j]])
    if len(triples) == 0:
        return torch.zeros((3, 0), dtype=torch.long)
    return torch.tensor(triples, dtype=torch.long).t().contiguous()


def compute_true_angles(coords, angle_index):
    """Bond angles (degrees) at each (a, center, b) triple in angle_index."""
    if angle_index.size(1) == 0:
        return torch.zeros(0)
    a, center, b = angle_index
    v1 = coords[a] - coords[center]
    v2 = coords[b] - coords[center]
    cos_angle = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-8)
    cos_angle = cos_angle.clamp(-1 + 1e-6, 1 - 1e-6)
    return cos_angle.arccos() * 180 / torch.pi


def build_dihedral_index(bond_edge_index):
    """For every bond (i,j), form (u,i,j,v) quadruples from i's and j's other
    neighbors -> used to compute true dihedral angles as an auxiliary loss."""
    src, dst = bond_edge_index
    edges = list(zip(src.tolist(), dst.tolist()))
    neighbors = defaultdict(list)
    for s, d in edges:
        neighbors[s].append(d)
    quads = []
    for i, j in edges:
        for u in neighbors[i]:
            if u == j:
                continue
            for v in neighbors[j]:
                if v == i:
                    continue
                quads.append([u, i, j, v])
    if len(quads) == 0:
        return torch.zeros((4, 0), dtype=torch.long)
    return torch.tensor(quads, dtype=torch.long).t().contiguous()


class MaceMolecularDataset(torch.utils.data.Dataset):
    def __init__(self, files_data, chirality_data, large_cutoff=LARGE_CUTOFF):
        self.files_data = files_data
        self.chirality_data = chirality_data   # passed explicitly (not a global)
        self.embedder = NodEmbedding()
        self.large_cutoff = large_cutoff

    def __len__(self):
        return len(self.files_data)

    def __getitem__(self, idx):
        atom_types, coords, bond_edges = self.embedder.node_embedding(self.files_data[idx])
        atomic_num = [element_to_z["C"] if a == 1 else element_to_z["H"] for a in atom_types]

        pos = torch.tensor(coords, dtype=torch.float)
        pos = pos - pos.mean(dim=0, keepdim=True)  # center on geometric centroid
        y = torch.tensor(self.chirality_data[idx, :], dtype=torch.float).unsqueeze(0)  # (1, 5)

        config = Configuration(
            atomic_numbers=np.array(atomic_num),
            positions=pos.numpy(),
            cell=np.zeros((3, 3)),
            pbc=(False, False, False),
            properties={}, property_weights={},
        )
        # built ONCE with the large cutoff -- a superset of edges the model could ever need
        atomic_data = AtomicData.from_config(config, z_table=z_table, cutoff=self.large_cutoff)

        # store the CLEAN edge vectors -- needed for the incremental update at every noise level
        i, j = atomic_data.edge_index
        atomic_data.edge_vec_clean = pos[j] - pos[i]  # matches get_edge_vectors_and_lengths convention

        bond_edge_index = torch.tensor(bond_edges, dtype=torch.long).t().contiguous()
        angle_index = build_angle_index(bond_edge_index)
        true_angles = compute_true_angles(pos, angle_index)
        dihedral_index = build_dihedral_index(bond_edge_index)

        atomic_data.bond_edge_index = bond_edge_index
        atomic_data.angle_index = angle_index
        atomic_data.true_angles = true_angles
        atomic_data.dihedral_index = dihedral_index
        atomic_data.y = y
        return atomic_data


def load_dataset(csv_path="chirality_data.csv", large_cutoff=LARGE_CUTOFF):
    """Reads chirality_data.csv (output of data_pipeline/02) and builds the dataset."""
    csv_data = np.loadtxt(csv_path, dtype=str)
    files_data = csv_data[:, 0]
    chirality_data = csv_data[:, 3:].astype(float)
    return MaceMolecularDataset(files_data, chirality_data, large_cutoff=large_cutoff), files_data
