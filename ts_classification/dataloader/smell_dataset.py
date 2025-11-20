import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    label: int
    label_name: str


def _infer_numeric_columns(csv_path: Path) -> List[str]:
    df = pd.read_csv(csv_path)
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_columns:
        raise ValueError(
            f"No numeric columns detected in {csv_path}. "
            "Please provide explicit feature_columns."
        )
    return numeric_columns


def _validate_feature_columns(
    available_columns: Sequence[str], requested_columns: Sequence[str]
) -> List[str]:
    missing = [col for col in requested_columns if col not in available_columns]
    if missing:
        raise ValueError(
            f"Columns {missing} are not present in the dataset. "
            f"Available columns: {available_columns}"
        )
    return list(requested_columns)


class SmellDataset(Dataset):
    """Dataset that loads smell sensor CSV files and returns padded sequences."""

    def __init__(
        self,
        samples: Sequence[SampleRecord],
        feature_columns: Optional[Sequence[str]] = None,
        seq_len: int = 512,
        normalization: str = "zscore",
    ) -> None:
        if not samples:
            raise ValueError("SmellDataset requires at least one sample.")

        self.samples = list(samples)
        self.seq_len = seq_len
        self.normalization = normalization.lower() if normalization else "none"

        auto_columns = _infer_numeric_columns(self.samples[0].path)
        if feature_columns:
            self.feature_columns = _validate_feature_columns(
                auto_columns, feature_columns
            )
        else:
            self.feature_columns = auto_columns

        self.num_features = len(self.feature_columns)
        self.series, self.masks, self.labels = self._preload_all()

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.series[idx], self.masks[idx], self.labels[idx]

    def _preload_all(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        data_tensors: List[torch.Tensor] = []
        mask_tensors: List[torch.Tensor] = []
        label_tensors: List[torch.Tensor] = []
        for sample in self.samples:
            seq, mask = self._load_sample(sample.path)
            data_tensors.append(seq)
            mask_tensors.append(mask)
            label_tensors.append(torch.tensor(sample.label, dtype=torch.long))
        return data_tensors, mask_tensors, label_tensors

    def _load_sample(self, csv_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        df = pd.read_csv(csv_path, usecols=self.feature_columns)
        series = df.to_numpy(dtype=np.float32)
        series = self._normalize(series)

        padded = np.zeros((self.seq_len, self.num_features), dtype=np.float32)
        mask = np.zeros((self.seq_len,), dtype=np.float32)

        effective_len = min(self.seq_len, series.shape[0])
        padded[:effective_len] = series[:effective_len]
        mask[:effective_len] = 1.0

        return torch.from_numpy(padded), torch.from_numpy(mask)

    def _normalize(self, series: np.ndarray) -> np.ndarray:
        if self.normalization == "zscore":
            mean = series.mean(axis=0, keepdims=True)
            std = series.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            return (series - mean) / std
        if self.normalization == "minmax":
            min_val = series.min(axis=0, keepdims=True)
            max_val = series.max(axis=0, keepdims=True)
            denom = max_val - min_val
            denom[denom == 0] = 1.0
            return (series - min_val) / denom
        return series


def gather_samples(
    data_root: Path, class_names: Sequence[str]
) -> Tuple[List[SampleRecord], Dict[str, int]]:
    data_root = Path(data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root {data_root} does not exist.")

    label_map: Dict[str, int] = {}
    samples: List[SampleRecord] = []
    for label, class_name in enumerate(class_names):
        class_dir = data_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Directory {class_dir} does not exist.")
        label_map[class_name] = label
        csv_files = sorted(class_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {class_dir}. Please verify the dataset."
            )
        for csv_file in csv_files:
            samples.append(SampleRecord(csv_file, label, class_name))

    return samples, label_map


def split_samples(
    samples: Sequence[SampleRecord], val_split: float, seed: int = 42
) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1.")

    samples = list(samples)
    random.Random(seed).shuffle(samples)
    val_size = max(1, int(len(samples) * val_split))
    train_size = max(1, len(samples) - val_size)
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]
    # In the edge case of extremely small datasets ensure both splits non-empty
    if not val_samples:
        val_samples = train_samples[-1:]
        train_samples = train_samples[:-1]
    return train_samples, val_samples


def create_dataloaders(
    data_root: str,
    classes: Sequence[str],
    feature_columns: Optional[Sequence[str]] = None,
    seq_len: int = 512,
    batch_size: int = 16,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
    normalization: str = "zscore",
) -> Tuple[DataLoader, DataLoader, Dict[str, int], List[str]]:
    samples, label_map = gather_samples(Path(data_root), classes)
    train_samples, val_samples = split_samples(samples, val_split, seed)

    train_dataset = SmellDataset(
        train_samples,
        feature_columns=feature_columns,
        seq_len=seq_len,
        normalization=normalization,
    )
    val_dataset = SmellDataset(
        val_samples,
        feature_columns=train_dataset.feature_columns,
        seq_len=seq_len,
        normalization=normalization,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, label_map, train_dataset.feature_columns

