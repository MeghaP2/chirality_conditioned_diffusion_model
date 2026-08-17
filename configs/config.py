"""
Fixed hyperparameters and element tables. Dataset-dependent statistics
(sigma_data, sigma_min/max, avg_num_neighbors) are NOT here — they depend on
the actual training set and are computed once in train.py / scripts, then
saved into the checkpoint's config dict so they never need to be re-derived
(or re-guessed) at load time.
"""
from e3nn import o3
from mace.tools import AtomicNumberTable
from mace.modules.blocks import RealAgnosticInteractionBlock, RealAgnosticResidualInteractionBlock
import torch

# ---- element / atom-type tables ----
atom_type_dir = {"C": 1, "H": 0}          # raw file label -> internal atom-type code
element_to_z = {"H": 1, "C": 6}            # internal atom-type code -> atomic number
z_table = AtomicNumberTable([1, 6])        # sorted ascending by atomic number

# ---- geometry / graph cutoffs ----
LARGE_CUTOFF = 15.0   # candidate edge list cutoff, built once per structure
R_MAX = 8.0            # real model cutoff, matches radial_embedding's envelope

# ---- MACE architecture ----
NUM_LAYERS = 3
HIDDEN_IRREPS = o3.Irreps("96x0e + 96x1o + 96x2e")
MLP_IRREPS = o3.Irreps("16x0e")   # unused energy-readout head, required by MACE ctor
NUM_BESSEL = 16
NUM_POLYNOMIAL_CUTOFF = 6
MAX_ELL = 3
NUM_ELEMENTS = 2
CORRELATION = 4
GATE = torch.nn.functional.silu
RADIAL_MLP = [64, 64, 64]
ATOMIC_NUMBERS = [1, 6]

INTERACTION_CLS = RealAgnosticResidualInteractionBlock       # layers 2..N (residual)
INTERACTION_CLS_FIRST = RealAgnosticInteractionBlock          # layer 1 (no residual)

# ---- conditioning ----
Y_DIM = 5           # chirality descriptor length
COND_DIM = 96        # y_embed output dim — independent of hidden_irreps' scalar
                      # count; chosen here to match it, but not required to
