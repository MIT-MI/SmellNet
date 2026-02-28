import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
    """Dataset that loads smell sensor CSV files and returns padded sequences.

    Args:
        samples: List of SampleRecord objects (path, label, label_name).
        feature_columns: Columns to use; None = auto-detect all numeric columns.
        seq_len: Fixed sequence length (pad/truncate).
        normalization: "zscore" | "minmax" | "none".
        window_stride: If set, extract multiple sliding windows per CSV with this
            stride. None = single window per CSV (default, current behaviour).
            Only intended for use on the training split.
        temporal_diff: If True, apply first-order temporal differencing to every
            channel before windowing. Applied identically on all splits so that
            the model always sees the same feature representation.
        diff_lag: Lag p for temporal differencing: ∆xt = xt - xt-p (default 1).
    """

    def __init__(
        self,
        samples: Sequence[SampleRecord],
        feature_columns: Optional[Sequence[str]] = None,
        seq_len: int = 512,
        normalization: str = "zscore",
        window_stride: Optional[int] = None,
        temporal_diff: bool = False,
        diff_lag: int = 1,
    ) -> None:
        if not samples:
            raise ValueError("SmellDataset requires at least one sample.")

        self.samples = list(samples)
        self.seq_len = seq_len
        self.normalization = normalization.lower() if normalization else "none"
        self.window_stride = window_stride
        self.temporal_diff = temporal_diff
        self.diff_lag = diff_lag

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preload_all(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        data_tensors: List[torch.Tensor] = []
        mask_tensors: List[torch.Tensor] = []
        label_tensors: List[torch.Tensor] = []
        for sample in self.samples:
            windows = self._load_windows(sample.path)
            for seq, mask in windows:
                data_tensors.append(seq)
                mask_tensors.append(mask)
                label_tensors.append(torch.tensor(sample.label, dtype=torch.long))
        return data_tensors, mask_tensors, label_tensors

    def _load_windows(self, csv_path: Path) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Load a CSV and return one or more (padded_seq, mask) windows."""
        df = pd.read_csv(csv_path, usecols=self.feature_columns)
        series = df.to_numpy(dtype=np.float32)
        series = self._normalize(series)

        if self.temporal_diff:
            series = self._apply_temporal_diff(series)

        if self.window_stride is None:
            # Single window — original behaviour
            return [self._pad_to_seq_len(series)]

        # Sliding windows
        n = len(series)
        windows = []
        start = 0
        while start < n:
            chunk = series[start : start + self.seq_len]
            windows.append(self._pad_to_seq_len(chunk))
            if start + self.seq_len >= n:
                break
            start += self.window_stride
        return windows

    def _pad_to_seq_len(self, series: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad or truncate a series to seq_len; return (tensor, mask)."""
        padded = np.zeros((self.seq_len, self.num_features), dtype=np.float32)
        mask = np.zeros((self.seq_len,), dtype=np.float32)

        effective_len = min(self.seq_len, series.shape[0])
        padded[:effective_len] = series[:effective_len]
        mask[:effective_len] = 1.0

        return torch.from_numpy(padded), torch.from_numpy(mask)

    def _apply_temporal_diff(self, series: np.ndarray) -> np.ndarray:
        """First-order temporal difference: ∆xt = xt - xt-p, dropping first p rows."""
        p = self.diff_lag
        return series[p:] - series[:-p]

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


def discover_all_classes(data_root: Path) -> List[str]:
    """
    Discover all class directories in data_root.

    Args:
        data_root: Root directory containing class folders

    Returns:
        Sorted list of class directory names
    """
    data_root = Path(data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root {data_root} does not exist.")

    class_dirs = [d.name for d in data_root.iterdir() if d.is_dir()]
    return sorted(class_dirs)


def resolve_classes(
    data_root: Path,
    classes: Union[str, int, Sequence[str]],
    seed: int = 42
) -> List[str]:
    """
    Resolve class specification to actual class list.

    Args:
        data_root: Root directory containing class folders
        classes: Class specification - can be:
            - "all": discover all available classes
            - integer N: randomly select N classes
            - list of strings: use explicit class names
        seed: Random seed for reproducibility when randomly selecting classes

    Returns:
        List of resolved class names

    Examples:
        >>> resolve_classes(data_root, "all")
        ['apple', 'banana', 'mango', ...]
        >>> resolve_classes(data_root, 5, seed=42)
        ['banana', 'mango', 'apple', 'pear', 'peach']
        >>> resolve_classes(data_root, ['banana', 'apple'])
        ['banana', 'apple']
    """
    if isinstance(classes, str) and classes.lower() == "all":
        return discover_all_classes(data_root)
    elif isinstance(classes, int):
        all_classes = discover_all_classes(data_root)
        if classes > len(all_classes):
            raise ValueError(
                f"Requested {classes} classes but only {len(all_classes)} available"
            )
        random.Random(seed).shuffle(all_classes)
        return all_classes[:classes]
    else:
        return list(classes)


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
    # Obtaining the sample records which basically tells you the csv paths, label and label_name
    return samples, label_map


def split_samples(
    samples: Sequence[SampleRecord],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42
) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
    """
    Split samples into train, validation, and test sets with stratification by class.

    Each class is split separately to ensure all classes appear in all splits proportionally,
    then the splits are combined. This prevents rare classes from being excluded from
    validation or test sets.

    Args:
        samples: List of samples to split
        train_split: Proportion for training set (0-1, can be 0 to omit)
        val_split: Proportion for validation set (0-1, can be 0 to omit)
        test_split: Proportion for test set (0-1, can be 0 to omit)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_samples, val_samples, test_samples). Any can be empty if split is 0.
    """
    if not all(0 <= s <= 1 for s in [train_split, val_split, test_split]):
        raise ValueError("All splits must be between 0 and 1.")

    total_split = train_split + val_split + test_split
    if not (0.99 <= total_split <= 1.01):  # Allow small floating point error
        raise ValueError(f"Splits must sum to 1.0, got {total_split}")

    # Group samples by class for stratified splitting
    samples_by_class: Dict[str, List[SampleRecord]] = {}
    for sample in samples:
        if sample.label_name not in samples_by_class:
            samples_by_class[sample.label_name] = []
        samples_by_class[sample.label_name].append(sample)

    # Split each class separately to ensure representation in all splits
    train_samples = []
    val_samples = []
    test_samples = []

    rng = random.Random(seed)
    for class_name, class_samples in samples_by_class.items():
        # Shuffle this class's samples
        class_samples = list(class_samples)
        rng.shuffle(class_samples)
        n_samples = len(class_samples)

        # Calculate sizes for this class
        train_size = int(n_samples * train_split)
        val_size = int(n_samples * val_split)
        test_size = n_samples - train_size - val_size  # Remainder to test

        # Split this class and add to combined splits
        if train_size > 0:
            train_samples.extend(class_samples[:train_size])
        if val_size > 0:
            val_samples.extend(class_samples[train_size:train_size + val_size])
        if test_size > 0:
            test_samples.extend(class_samples[train_size + val_size:])

    return train_samples, val_samples, test_samples


def create_dataloaders(
    data_root: str,
    classes: Union[str, int, Sequence[str]],
    feature_columns: Optional[Sequence[str]] = None,
    seq_len: int = 512,
    batch_size: int = 16,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
    normalization: str = "zscore",
    train_window_stride: Optional[int] = None,
    temporal_diff: bool = False,
    diff_lag: int = 1,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], Dict[str, int], List[str], List[str]]:
    """
    Create train, validation, and test dataloaders with automatic class resolution.

    Args:
        classes: Can be:
            - "all": auto-discover all available classes
            - integer N: randomly select N classes
            - list of strings: use explicit class names
        train_window_stride: Sliding window stride for the training split only.
            None = single window per CSV (default). Val/test always use a single
            window to preserve evaluation robustness.
        temporal_diff: Apply first-order temporal differencing (∆xt = xt - xt-p)
            to all splits before windowing.
        diff_lag: Lag p for temporal differencing (default 1).

    Returns:
        Tuple of (train_loader, val_loader, test_loader, label_map, feature_columns, resolved_classes).
        Any loader can be None if corresponding split is 0.
    """
    # Resolve classes if needed (handle "all", integer, or explicit list)
    if isinstance(classes, (str, int)):
        resolved_classes = resolve_classes(Path(data_root), classes, seed)
    else:
        resolved_classes = list(classes)

    samples, label_map = gather_samples(Path(data_root), resolved_classes)
    train_samples, val_samples, test_samples = split_samples(
        samples, train_split, val_split, test_split, seed
    )

    # Create train dataset and loader
    train_loader = None
    feature_cols = feature_columns
    if train_samples:
        train_dataset = SmellDataset(
            train_samples,
            feature_columns=feature_columns,
            seq_len=seq_len,
            normalization=normalization,
            window_stride=train_window_stride,
            temporal_diff=temporal_diff,
            diff_lag=diff_lag,
        )
        feature_cols = train_dataset.feature_columns
        print(f"Train windows: {len(train_dataset)} (from {len(train_samples)} files)")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # Create validation dataset and loader (single window per file)
    val_loader = None
    if val_samples:
        val_dataset = SmellDataset(
            val_samples,
            feature_columns=feature_cols,
            seq_len=seq_len,
            normalization=normalization,
            window_stride=None,
            temporal_diff=temporal_diff,
            diff_lag=diff_lag,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # Create test dataset and loader (single window per file)
    test_loader = None
    if test_samples:
        test_dataset = SmellDataset(
            test_samples,
            feature_columns=feature_cols,
            seq_len=seq_len,
            normalization=normalization,
            window_stride=None,
            temporal_diff=temporal_diff,
            diff_lag=diff_lag,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, test_loader, label_map, feature_cols, resolved_classes
