#!/usr/bin/env python
import argparse
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from load_data import (
    load_sensor_data,
    make_sliding_window_dataset,
    diff_data_like,
)


def build_sliding_data(sensor_data_dict, le, window, stride=None):
    """Wrapper around your existing helper."""
    if stride is None:
        stride = max(1, window // 2)
    X, y = make_sliding_window_dataset(
        sensor_data_dict, le, window_size=window, stride=stride
    )
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--real-test-dir", default=None)

    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--gradient", type=int, default=0)

    parser.add_argument(
        "--mode",
        choices=["env_only", "sensory_only", "full"],
        default="env_only",
        help="Which channels to feed to the tree.",
    )
    parser.add_argument("--n-trees", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # --------- Load raw sensor data ----------
    train_data, test_data, _ = load_sensor_data(
        training_path=args.train_dir,
        testing_path=args.test_dir,
        real_time_testing_path=args.real_test_dir,
        removed_filtered_columns=[]
    )

    # Optional differencing
    if args.gradient and args.gradient > 0:
        train_data = diff_data_like(train_data, periods=args.gradient)
        test_data  = diff_data_like(test_data,  periods=args.gradient)

    # --------- LabelEncoder ----------
    all_labels = sorted(train_data.keys())
    le = LabelEncoder()
    le.fit(all_labels)

    # --------- Build sliding windows ----------
    Xtr, ytr = build_sliding_data(train_data, le, args.window_size, args.stride)
    Xte, yte = build_sliding_data(test_data,  le, args.window_size, args.stride)

    # --------- Collapse windows to per-channel mean features ----------
    Xtr_feat = Xtr.mean(axis=1)  # (Ntr, C)
    Xte_feat = Xte.mean(axis=1)  # (Nte, C)

    # --------- Train RandomForest ----------
    rf = RandomForestClassifier(
        n_estimators=args.n_trees,
        max_depth=args.max_depth,
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(Xtr_feat, ytr)

    # --------- Evaluate: acc@1, acc@5, F1 (macro) ----------
    # Top-1 predictions
    y_pred = rf.predict(Xte_feat)

    # Acc@1
    acc1 = accuracy_score(yte, y_pred)

    # Acc@5 via predict_proba
    prob = rf.predict_proba(Xte_feat)          # (N, K)
    top5_idx = np.argsort(prob, axis=1)[:, -5:]  # last 5 = top 5
    y_true = yte.reshape(-1, 1)
    hits_top5 = (top5_idx == y_true).any(axis=1)
    acc5 = hits_top5.mean()

    # Macro F1
    f1 = f1_score(yte, y_pred, average="macro")

    print(f"[RESULT] RandomForest mode={args.mode}")
    print(f"  Acc@1: {acc1 * 100:.2f}%")
    print(f"  Acc@5: {acc5 * 100:.2f}%")
    print(f"  F1 (macro): {f1 * 100:.2f}%")


if __name__ == "__main__":
    main()
