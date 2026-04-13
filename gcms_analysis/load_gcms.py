import numpy as np
import torch
from torch.utils.data import Dataset

gcms = np.load("gcms_processed/gcms_food_vectors.npz", allow_pickle=True)
gcms_X = gcms["vectors"]          # (N_foods, D)
gcms_food_labels = gcms["food_labels"]  # (N_foods,)

print(gcms_X)