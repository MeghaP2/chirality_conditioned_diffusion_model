"""
Determinism / environment setup. Must be imported before torch/CUDA touch
anything else — that's why this is its own module, imported first in every
entry-point script.
"""
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"          # restricts cuBLAS to deterministic algorithm variants
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"        # allow full unpickling of our own checkpoints

import numpy as np
import torch
import torch.serialization

torch.serialization.add_safe_globals([slice])  # extend the safe-unpickling allowlist

torch.set_num_threads(64)
torch.manual_seed(0)
np.random.seed(42)
torch.use_deterministic_algorithms(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
