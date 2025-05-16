import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import random


class PairedDataset(Dataset):
    """
    Expects 'data' to be a list (or array-like) of length N,
    where each item is (gcms_vector, smell_vector).

    Each vector could be:
      - a NumPy array of shape [feature_dim]
      - a Python list
      - etc.
    We'll just return them as Tensors.
    """

    def __init__(self, data, transformer=False):
        self.data = data  # data = [(gcms_vec, smell_vec), (gcms_vec, smell_vec), ...]
        self.transformer = transformer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gcms_vec, smell_vec = self.data[idx]

        # Convert to torch.FloatTensors safely
        if isinstance(gcms_vec, torch.Tensor):
            gcms_vec = gcms_vec.detach().clone()
        else:
            gcms_vec = torch.tensor(gcms_vec, dtype=torch.float)

        if isinstance(smell_vec, torch.Tensor):
            smell_vec = smell_vec.detach().clone()
        else:
            smell_vec = torch.tensor(smell_vec, dtype=torch.float)

        # Add time dimension for transformer
        if self.transformer:
            if smell_vec.dim() == 1:
                smell_vec = smell_vec.unsqueeze(0)  # → (1, feature_dim)

        return gcms_vec, smell_vec
    


class FusionDataset(Dataset):
    def __init__(self, data, le, transformer=False):
        """
        data: list of (gcms_vec, smell_vec, ingredient_name)
        le: fitted LabelEncoder to convert ingredient_name → label
        transformer: whether to add time dim to smell_vec
        """
        self.data = data
        self.le = le
        self.transformer = transformer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gcms_vec, smell_vec, ingredient = self.data[idx]

        # Safe tensor conversion
        if isinstance(gcms_vec, torch.Tensor):
            gcms_vec = gcms_vec.detach().clone()
        else:
            gcms_vec = torch.tensor(gcms_vec, dtype=torch.float)

        if isinstance(smell_vec, torch.Tensor):
            smell_vec = smell_vec.detach().clone()
        else:
            smell_vec = torch.tensor(smell_vec, dtype=torch.float)

        # Add time dim if using transformer
        if self.transformer:
            smell_vec = smell_vec.unsqueeze(0)  # → (1, feature_dim)

        label = ingredient

        return gcms_vec, smell_vec, label


class UniqueGCMSampler(Sampler):
    """
    Ensures each batch has unique gcms vectors.
    """

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

        # Build gcms → smell index mapping
        self.gcms_to_indices = {}
        for idx, (gcms_vec, _) in enumerate(data):
            gcms_key = tuple(gcms_vec)
            self.gcms_to_indices.setdefault(gcms_key, []).append(idx)

        # Shuffle within groups
        for idx_list in self.gcms_to_indices.values():
            random.shuffle(idx_list)

        self.unique_gcms = list(self.gcms_to_indices.keys())

    def __iter__(self):
        gcms_queues = {key: list(indices) for key, indices in self.gcms_to_indices.items()}
        all_batches = []

        # Keep looping until all queues are empty
        while any(gcms_queues.values()):
            random.shuffle(self.unique_gcms)
            current_batch = []
            for gcms_key in self.unique_gcms:
                queue = gcms_queues[gcms_key]
                if queue:
                    current_batch.append(queue.pop())
                    if len(current_batch) == self.batch_size:
                        all_batches.append(current_batch)
                        current_batch = []
            if current_batch:
                all_batches.append(current_batch)

        # Flatten into index list
        all_indices = [idx for batch in all_batches for idx in batch]
        return iter(all_indices)

    def __len__(self):
        return len(self.data)