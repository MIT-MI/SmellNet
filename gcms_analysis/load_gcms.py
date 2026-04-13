from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
gcms = np.load(_REPO_ROOT / "gcms_processed" / "gcms_food_vectors.npz", allow_pickle=True)
gcms_X = gcms["vectors"]          # (N_foods, D)
gcms_food_labels = gcms["food_labels"]  # (N_foods,)

print(gcms_X)