import pandas as pd
import torch
import os
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils import ingredient_to_category
import re


def subtract_first_row(df):
    return df - df.iloc[0]


def load_sensor_data(
    training_path,
    testing_path,
    ingredients=None,
    categories=None,
    real_time_testing_path=None,
    removed_filtered_columns=[],
):
    training_data = defaultdict(list)
    testing_data = defaultdict(list)

    # Helper: subtract first row
    def subtract_first_row(df):
        return df - df.iloc[0]

    # Walk through the training directory
    for folder_name in os.listdir(training_path):
        folder_path = os.path.join(training_path, folder_name)
        if os.path.isdir(folder_path):  # Make sure it's a folder
            for filename in os.listdir(folder_path):
                if filename.endswith(".csv"):
                    cur_path = os.path.join(folder_path, filename)
                    df = pd.read_csv(cur_path)
                    df = subtract_first_row(df)
                    df = df.drop(columns=removed_filtered_columns)
                    training_data[folder_name].append(df)

    for folder_name in os.listdir(testing_path):
        folder_path = os.path.join(testing_path, folder_name)

        if ingredients:
            if folder_name in ingredients:
                if os.path.isdir(folder_path):  # Make sure it's a folder
                    for filename in os.listdir(folder_path):
                        if filename.endswith(".csv"):
                            cur_path = os.path.join(folder_path, filename)
                            df = pd.read_csv(cur_path)
                            df = subtract_first_row(df)
                            df = df.drop(columns=removed_filtered_columns)
                            testing_data[folder_name].append(df)
        else:
            if categories is None or ingredient_to_category[folder_name] in categories:
                if os.path.isdir(folder_path):  # Make sure it's a folder
                    for filename in os.listdir(folder_path):
                        if filename.endswith(".csv"):
                            cur_path = os.path.join(folder_path, filename)
                            df = pd.read_csv(cur_path)
                            df = subtract_first_row(df)
                            df = df.drop(columns=removed_filtered_columns)
                            testing_data[folder_name].append(df)

    real_time_testing_data = defaultdict(list)
    for folder_name in os.listdir(real_time_testing_path):
        folder_path = os.path.join(real_time_testing_path, folder_name)

        if os.path.isdir(folder_path):  # Make sure it's a folder
            for filename in os.listdir(folder_path):
                if filename.endswith(".csv"):
                    cur_path = os.path.join(folder_path, filename)
                    df = pd.read_csv(cur_path)
                    df = subtract_first_row(df)
                    df = df.drop(columns=removed_filtered_columns)
                    real_time_testing_data[folder_name].append(df)

    return training_data, testing_data, real_time_testing_data


def load_gcms_data(path):
    df = pd.read_csv(path)

    feature_cols = df.columns[1:]
    label_col = df.columns[0]

    # Extract features and labels
    X = df[feature_cols].values
    y = df[label_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X_scaled, y_encoded, le, scaler


def load_text_data(path, le=None):
    text_embeddings = np.load(path, allow_pickle=True).item()

    X = np.array([value for _, value in text_embeddings.items()])
    y = list(text_embeddings.keys())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode labels
    if le is None:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = le.transform(y)

    return X_scaled, y_encoded, le, scaler


def make_sliding_window_dataset(
    data: dict[str, list[pd.DataFrame]],
    le,
    window_size: int = 100,
    stride: int = 50,
):
    """
    Build a windowed time-series dataset from {label: [DataFrame, ...]}.

    Returns
    -------
    X : np.ndarray, shape [N, window_size, C]
        Stacked sliding windows of features.
    y : np.ndarray, shape [N]
        Label-encoded class IDs aligned with X.
    label_encoder : same as input
        Returned unchanged; must be pre-fitted (uses .transform()).
    """
    X = []
    y = []

    for ingredient, dfs in data.items():
        for df in dfs:
            for start in range(0, len(df) - window_size + 1, stride):
                window = df.iloc[start : start + window_size].values
                X.append(window)
                y.append(ingredient)

    y = le.transform(y)
    X = np.array(X)  # shape: [N, T, C]

    return X, y


def diff_data_like(
    data: dict,
    periods: int = 25,
):
    out = {}
    for label, dfs in data.items():
        out_list = []
        for df in dfs:
            diff_df = df.diff(periods=periods).iloc[periods:]
            out_list.append(diff_df)
        out[label] = out_list
    return out


def create_pair_data(smell_data, smell_label, gcms_data, le):
    pair_data = []

    for i in range(len(smell_label)):
        gcms_ix = smell_label[i]
        pair_data.append((gcms_data[gcms_ix], smell_data[i]))
    return pair_data


def apply_random_feature_dropout(X, dropout_fraction=0.25, seed=None):
    """
    Apply random feature dropout to a batch or dataset.

    Parameters:
    - X: torch.Tensor or np.ndarray, shape [batch_size, time_steps, feature_dim] or [batch_size, feature_dim]
    - dropout_fraction: float, fraction of features to zero out (e.g., 0.25 → drop 25%)
    - seed: int or None, random seed for reproducibility

    Returns:
    - X_dropped: same type as input, with specified features zeroed out
    """
    if seed is not None:
        np.random.seed(seed)

    if isinstance(X, np.ndarray):
        X = torch.tensor(X)

    feature_dim = X.shape[-1]
    num_features_to_drop = int(feature_dim * dropout_fraction)

    # Randomly select feature indices to drop
    drop_indices = np.random.choice(feature_dim, num_features_to_drop, replace=False)
    mask = torch.ones(feature_dim)
    mask[drop_indices] = 0

    # Apply mask
    X_dropped = X * mask.to(X.device)

    return X_dropped


def apply_noise_injection(X, noise_scale=0.05, seed=None):
    """
    Add Gaussian noise to the input tensor.

    Parameters:
    - X: torch.Tensor, shape [batch_size, time_steps, feature_dim] or [batch_size, feature_dim]
    - noise_scale: float, standard deviation of Gaussian noise
    - seed: int or None, for reproducibility

    Returns:
    - X_noisy: torch.Tensor, same shape as input
    """
    if seed is not None:
        torch.manual_seed(seed)

    noise = torch.randn_like(X) * noise_scale
    X_noisy = X + noise
    return X_noisy


def highpass_fft_batch(X, sampling_rate=1.0, cutoff=0.05):
    """
    High-pass via FFT zeroing for an entire batch of windows.
    X: np.ndarray (N, T, C)
    cutoff: Hz (frequencies < cutoff are removed)
    """
    X = np.asarray(X)
    N, T, C = X.shape
    # Real FFT along time
    F = np.fft.rfft(X, axis=1)                       # (N, T//2+1, C)
    freqs = np.fft.rfftfreq(T, d=1.0 / sampling_rate)  # (T//2+1,)
    mask = (freqs >= cutoff)[None, :, None]          # broadcast to (1, F, 1)
    F *= mask
    X_clean = np.fft.irfft(F, n=T, axis=1)           # (N, T, C)
    return X_clean


def load_smell_recognition_data(directory_path):
    ALL_INGREDIENTS = [
        'banana', 'orange', 'pear', 'apple', 'mango', 'peach',
        'strawberry', 'clove', 'coriander', 'garlic', 'almond', 'cumin'
    ]

    filenames = set()

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            filenames.add(file.split(".")[0])  # Remove extension

    data = []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            name = file.split(".")[0]
            name_cleaned = name.lower().replace("__", "_").replace("-", "_")

            # Load your CSV (features) first
            df = pd.read_csv(os.path.join(root, file))

            # Initialize all zeros
            ingredient_percentages = {ingredient: 0 for ingredient in ALL_INGREDIENTS}

            parts = name_cleaned.split("_")

            def fill_from_pairs(parts_list):
                """Parse ingredient/percentage pairs like ['banana','50','mango','50']."""
                for i in range(0, len(parts_list), 2):
                    ing = parts_list[i]
                    pct_str = parts_list[i + 1]
                    if not ing.isalpha():
                        raise ValueError(f"Invalid ingredient token '{ing}' in filename '{name}'.")
                    if not pct_str.isdigit():
                        raise ValueError(f"Invalid percentage token '{pct_str}' in filename '{name}'.")
                    pct = int(pct_str)
                    if not (0 <= pct <= 100):
                        raise ValueError(f"Percentage out of range {pct} in filename '{name}'.")
                    if ing in ingredient_percentages:
                        ingredient_percentages[ing] = pct
                    else:
                        raise ValueError(f"Unknown ingredient '{ing}' in filename '{name}'.")

            parsed = False

            # Case A: clean even-length list of pairs
            if len(parts) % 2 == 0 and len(parts) > 0:
                try:
                    fill_from_pairs(parts)
                    parsed = True
                except ValueError:
                    parsed = False  # fall back to regex

            # Case B: fallback — match mashed styles like 'banana50_orange50'
            if not parsed:
                pairs = re.findall(r'([a-z]+)[_]?(\d+)', name_cleaned)
                if pairs:
                    # Ensure everything in the filename is covered by the pairs we found
                    # (optional strictness). We'll trust pairs if present.
                    for ing, pct_str in pairs:
                        pct = int(pct_str)
                        if not (0 <= pct <= 100):
                            raise ValueError(f"Percentage out of range {pct} in filename '{name}'.")
                        if ing in ingredient_percentages:
                            ingredient_percentages[ing] = pct
                        else:
                            raise ValueError(f"Unknown ingredient '{ing}' in filename '{name}'.")
                    parsed = True

            # Case C: single ingredient with no percentage -> assume 100
            if not parsed:
                # If all tokens are alpha and only one unique ingredient, assume 100
                if all(tok.isalpha() for tok in parts) and len(set(parts)) == 1:
                    ing = parts[0]
                    if ing in ingredient_percentages:
                        ingredient_percentages[ing] = 100
                        parsed = True

            if not parsed:
                raise ValueError(f"Unrecognized filename format: '{name}'")

            label_vector = [ingredient_percentages[ing] / 100 for ing in ALL_INGREDIENTS]

            # Validate sum is exactly 100
            total = sum(label_vector)
            if total < 0.99:
                raise ValueError(
                    f"Percentages must sum to 100 (got {total}) for file '{file}' "
                    f"-> vector {label_vector}"
                )

            data.append((df[:600], label_vector))

    return data