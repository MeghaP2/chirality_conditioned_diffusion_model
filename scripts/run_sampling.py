"""
End-to-end sampling entry point: load checkpoint -> generate conditioned
structures for a list of dataset indices -> write .xyz files.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mace_diffusion import env_setup   # noqa: E402
from mace_diffusion.dataset import load_dataset, NodEmbedding
from mace_diffusion.checkpoint import load_checkpoint, rebuild_from_checkpoint
from mace_diffusion.sample import sample_mace_heun, save_xyz
import torch

device = env_setup.device

CHECKPOINT_PATH = "checkpoints/mace_diffusion_checkpoint.pt"
N_ATOMS = 29             # nonane
NUM_SAMPLING_STEPS = 200
INDICES = [0, 1, 5, 10, 20]   # which dataset entries' labels/topology to condition on

dataset, files_data = load_dataset("chirality_data.csv")

checkpoint = load_checkpoint(CHECKPOINT_PATH, device)
mace_model, wrapper = rebuild_from_checkpoint(checkpoint, device)
wrapper.eval()

sigmas = None  # TODO: reconstruct from checkpoint["diffusion_config"] via
               # torch.exp(linspace(log(sigma_min), log(sigma_max), T)) --
               # see diffusion.compute_sigma_data_and_schedule for the formula
dc = checkpoint["diffusion_config"]
import numpy as np
sigmas = torch.exp(torch.linspace(np.log(dc["sigma_min"]), np.log(dc["sigma_max"]), dc["T"])).to(device)

sample_sigma_max = 1.0 * dc["sigma_data"]   # smaller starting point for generation only
sample_sigma_min = dc["sigma_min"]

for idx in INDICES:
    template_graph = dataset[idx].to(device)
    y_true = template_graph.y
    torch.manual_seed(999)

    new_mol = sample_mace_heun(
        wrapper, template_graph, sigmas, dc["sigma_data"],
        sample_sigma_min, sample_sigma_max, y=y_true, num_steps=NUM_SAMPLING_STEPS,
    )

    atom_types, _, _ = NodEmbedding().node_embedding(files_data[idx])
    atom_types_tensor = torch.tensor(atom_types).to(device)
    out_path = f"outputs/conditioned_new_mol_{idx}.xyz"
    save_xyz(N=N_ATOMS, coords=new_mol, atomtypes=atom_types_tensor, filename=out_path)
    print(f"wrote {out_path}")
