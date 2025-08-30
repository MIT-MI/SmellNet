from __future__ import annotations
import os
import random
import json
from load_data import *
from models import *

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from joblib import load
from sklearn.preprocessing import StandardScaler
from run_chi_model import SmellDataset, SmellTemporalCNN
import torch.nn.functional as F

from config import (
    BASE_DIR, MODEL_PATH, SCALERS_PATH, TEST_INDEX, TEST_INDEX2, TRAIN_INDEX,
    INGREDIENTS, WIN_LEN, BATCH_SIZE, THRESHOLD1, THRESHOLD2, THRESHOLD3, TEST_DIR, TEST_DIR2
)

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom loss function for KL divergence on logits vs prob targets
class KLDivLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # pred: (batch_size, n_classes) - logits
        # target: (batch_size, n_classes) - target probabilities (sum to 1 or soft ratios)
        log_pred = torch.log_softmax(pred, dim=1)
        kl_div = torch.sum(target * (torch.log(target + 1e-8) - log_pred), dim=1)
        return torch.mean(kl_div)

def scent_ratio_accuracy(pred_probs, gt_probs, threshold=0.05):
    """
    Average threshold accuracy over all ingredients per sample:
    fraction of ingredients where |pred - gt| <= threshold, then mean over batch.
    """
    assert pred_probs.shape == gt_probs.shape, "pred and gt must have the same shape"
    diff = torch.abs(pred_probs - gt_probs)
    correct_per_sample = (diff <= threshold).float().mean(dim=1)  # (B,)
    return correct_per_sample.float().sum().item()

def scent_ratio_accuracy_strict(pred_probs, gt_probs, threshold=0.05):
    """
    Strict threshold accuracy: counts a sample as correct only if *all* ingredients
    are within the threshold.
    """
    assert pred_probs.shape == gt_probs.shape, "pred and gt must have the same shape"
    diff = torch.abs(pred_probs - gt_probs)
    correct_per_sample = (diff <= threshold).all(dim=1)  # (B,)
    return correct_per_sample.float().sum().item()

def scent_ratio_accuracy_nonzero(pred_probs, gt_probs, threshold=0.05):
    """
    Threshold accuracy computed only over non-zero ground-truth ingredients.
    """
    assert pred_probs.shape == gt_probs.shape, "pred and gt must have the same shape"
    nonzero_mask = gt_probs > 0.0  # (B, C)
    diff = torch.abs(pred_probs - gt_probs)
    total_accuracy = 0.0
    for i in range(pred_probs.size(0)):
        sample_nonzero = nonzero_mask[i]
        if sample_nonzero.sum() == 0:
            continue
        sample_diff = diff[i][sample_nonzero]
        sample_accuracy = (sample_diff <= threshold).float().mean().item()
        total_accuracy += sample_accuracy
    return total_accuracy

# ---------------- Metrics ----------------
class MetricsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.mae_sum = 0.0
        self.kl_sum = 0.0
        self.total = 0

        # User-specified thresholded Top-1 / Top-3
        self.correct_top1 = 0   # |pred - gt| < 0.1 at argmax(pred)
        self.correct_top3 = 0   # any of top-3 indices has |pred - gt| < 0.3

        # Threshold accuracies (avg / strict / nonzero)
        self.correct_01 = 0
        self.correct_02 = 0
        self.correct_03 = 0
        self.correct_01_strict = 0
        self.correct_02_strict = 0
        self.correct_03_strict = 0
        self.correct_01_nonzero = 0
        self.correct_02_nonzero = 0
        self.correct_03_nonzero = 0
    
    def update(self, pred_logits, target_probs):
        self.total += pred_logits.size(0)

        # Convert logits to probabilities for metric computations
        pred_probs = F.softmax(pred_logits, dim=1)

        row_sums = pred_probs.sum(dim=1)   # shape (B,)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            raise ValueError("Probabilities do not sum to 1 (within tolerance)")

        # MAE (on probabilities)
        mae = torch.mean(torch.abs(pred_probs - target_probs))
        self.mae_sum += mae.item() * pred_logits.size(0)

        # KL divergence (use custom that internally log-softmaxes logits)
        kl_loss = KLDivLoss()(pred_logits, target_probs)
        self.kl_sum += kl_loss.item() * pred_logits.size(0)

        # --- User-defined Top-1 and Top-3 thresholded accuracies ---
        # Top-1: check argmax(pred_probs)
        top1_idx = torch.argmax(pred_probs, dim=1)  # (B,)
        top1_pred_vals = pred_probs.gather(1, top1_idx.unsqueeze(1)).squeeze(1)  # (B,)
        top1_true_vals = target_probs.gather(1, top1_idx.unsqueeze(1)).squeeze(1)  # (B,)
        self.correct_top1 += (torch.abs(top1_pred_vals - top1_true_vals) < 0.1).float().sum().item()

        # Top-3: any of top3 indices has |pred - gt| < 0.3
        top3_idx = torch.topk(pred_probs, k=min(3, pred_probs.shape[1]), dim=1).indices  # (B, K<=3)
        top3_pred_vals = pred_probs.gather(1, top3_idx)  # (B, K)
        top3_true_vals = target_probs.gather(1, top3_idx)  # (B, K)
        top3_ok = (torch.abs(top3_pred_vals - top3_true_vals) < 0.3).any(dim=1).float()  # (B,)
        self.correct_top3 += top3_ok.sum().item()

        # Threshold metrics (use probabilities)
        self.correct_01 += scent_ratio_accuracy(pred_probs, target_probs, threshold=THRESHOLD1)
        self.correct_02 += scent_ratio_accuracy(pred_probs, target_probs, threshold=THRESHOLD2)
        self.correct_03 += scent_ratio_accuracy(pred_probs, target_probs, threshold=THRESHOLD3)

        self.correct_01_strict += scent_ratio_accuracy_strict(pred_probs, target_probs, threshold=THRESHOLD1)
        self.correct_02_strict += scent_ratio_accuracy_strict(pred_probs, target_probs, threshold=THRESHOLD2)
        self.correct_03_strict += scent_ratio_accuracy_strict(pred_probs, target_probs, threshold=THRESHOLD3)

        self.correct_01_nonzero += scent_ratio_accuracy_nonzero(pred_probs, target_probs, threshold=THRESHOLD1)
        self.correct_02_nonzero += scent_ratio_accuracy_nonzero(pred_probs, target_probs, threshold=THRESHOLD2)
        self.correct_03_nonzero += scent_ratio_accuracy_nonzero(pred_probs, target_probs, threshold=THRESHOLD3)
    
    def get_metrics(self):
        return {
            'mae': self.mae_sum / self.total,
            'kl': self.kl_sum / self.total,
            # NOTE: top1/top3 now follow the user's thresholded definitions
            'top1': self.correct_top1 / self.total,
            'top3': self.correct_top3 / self.total,
            '01': self.correct_01 / self.total,
            '02': self.correct_02 / self.total,
            '03': self.correct_03 / self.total,
            '01_strict': self.correct_01_strict / self.total,
            '02_strict': self.correct_02_strict / self.total,
            '03_strict': self.correct_03_strict / self.total,
            '01_nonzero': self.correct_01_nonzero / self.total,
            '02_nonzero': self.correct_02_nonzero / self.total,
            '03_nonzero': self.correct_03_nonzero / self.total}

def get_ingredient_combination(target_sample: torch.Tensor, threshold: float = 0.01) -> tuple[str, ...]:
    active_indices = torch.where(target_sample > threshold)[0]
    active_ingredients = tuple(INGREDIENTS[i] for i in active_indices.tolist())
    return active_ingredients

def load_training_combinations() -> set[tuple[str, ...]]:
    train_df = pd.read_csv(TRAIN_INDEX)
    combinations = set()
    for _, row in train_df.iterrows():
        target = torch.tensor([row[f'label_{ingredient}'] for ingredient in INGREDIENTS])
        combination = get_ingredient_combination(target)
        combinations.add(combination)
    return combinations

def analyze_combinations(all_predictions: torch.Tensor, all_targets: torch.Tensor) -> Dict[str, Any]:
    """
    all_predictions: probabilities (not logits)
    all_targets: probabilities
    """
    training_combinations = load_training_combinations()

    # Group samples by combination
    combination_groups = {}
    for i in range(all_targets.size(0)):
        combination = get_ingredient_combination(all_targets[i])
        combination_groups.setdefault(combination, []).append(i)
    
    # Separate into training vs unseen combinations
    training_combos = {}
    unseen_combos = {}
    for combo, indices in combination_groups.items():
        (training_combos if combo in training_combinations else unseen_combos)[combo] = indices
    
    def calculate_group_accuracy(combo_dict):
        total_samples = sum(len(indices) for indices in combo_dict.values())
        if total_samples == 0:
            return 0.0, 0.0, 0.0, {}
        all_indices = [i for indices in combo_dict.values() for i in indices]
        group_predictions = all_predictions[all_indices]
        group_targets = all_targets[all_indices]
        acc_01 = scent_ratio_accuracy_nonzero(group_predictions, group_targets, THRESHOLD1) / len(all_indices)
        acc_02 = scent_ratio_accuracy_nonzero(group_predictions, group_targets, THRESHOLD2) / len(all_indices)
        acc_03 = scent_ratio_accuracy_nonzero(group_predictions, group_targets, THRESHOLD3) / len(all_indices)
        combo_details = {}
        for combo, indices in combo_dict.items():
            combo_preds = all_predictions[indices]
            combo_targets = all_targets[indices]
            combo_acc_01 = scent_ratio_accuracy_nonzero(combo_preds, combo_targets, THRESHOLD1) / len(indices)
            combo_acc_02 = scent_ratio_accuracy_nonzero(combo_preds, combo_targets, THRESHOLD2) / len(indices)
            combo_details[combo] = {
                'count': len(indices),
                'accuracy_01': combo_acc_01,
                'accuracy_02': combo_acc_02
            }
        return acc_01, acc_02, acc_03, combo_details
    
    training_acc_01, training_acc_02, training_acc_03, training_details = calculate_group_accuracy(training_combos)
    unseen_acc_01, unseen_acc_02, unseen_acc_03, unseen_details = calculate_group_accuracy(unseen_combos)
    
    return {
        'training_combinations': {
            'count': len(training_combos),
            'total_samples': sum(len(indices) for indices in training_combos.values()),
            'accuracy_01': training_acc_01,
            'accuracy_02': training_acc_02,
            'accuracy_03': training_acc_03,
            'details': training_details
        },
        'unseen_combinations': {
            'count': len(unseen_combos),
            'total_samples': sum(len(indices) for indices in unseen_combos.values()),
            'accuracy_01': unseen_acc_01,
            'accuracy_02': unseen_acc_02,
            'accuracy_03': unseen_acc_03,
            'details': unseen_details
        }
    }

def add_standardization(batch: tuple[torch.Tensor, torch.Tensor], scalers: List[StandardScaler], D: int):
    """
    Apply standardization to the batch without augmentation.
    """
    means = torch.tensor([sc.mean_[0] for sc in scalers], dtype=torch.float32, device=DEVICE)
    scales = torch.tensor([sc.scale_[0] for sc in scalers], dtype=torch.float32, device=DEVICE)
    
    x, y = batch
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    
    # x: (batch_size, WIN_LEN, D)
    pad_rows = torch.all(x == 0.0, dim=-1)  # (batch_size, WIN_LEN)
    x = (x - means) / torch.clamp(scales, min=1e-8)
    
    # Restore structural padding to 0
    x = torch.where(pad_rows.unsqueeze(-1), torch.zeros_like(x), x)
    return x, y

def evaluate_model(model: nn.Module, test_loader) -> Dict[str, Any]:
    """
    Evaluate the model on the test set.
    Returns (metrics_dict, all_predictions_probs, all_targets).
    """
    model.eval()
    test_metrics = MetricsTracker()
    test_loss_sum = 0.0
    criterion = KLDivLoss()  # use custom KL on logits vs prob targets
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            # x, y = add_standardization((x, y), scalers, D)
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            # Forward pass (assume model returns logits)
            criterion = torch.nn.KLDivLoss(reduction="batchmean")
            pred_logits = model(x)

            # Update metrics
            test_metrics.update(pred_logits, y)
            
            # Store probabilities and targets for detailed analysis
            pred_probs = F.softmax(pred_logits, dim=1)
            all_predictions.append(pred_probs.cpu())
            all_targets.append(y.cpu())
    
    # Average loss
    test_loss_avg = test_loss_sum / max(1, len(test_loader))
    
    # Get metrics
    metrics = test_metrics.get_metrics()
    metrics['loss'] = test_loss_avg
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)  # probs
    all_targets = torch.cat(all_targets, dim=0)
    
    return metrics, all_predictions, all_targets

def print_detailed_results(metrics: Dict[str, Any], all_predictions: torch.Tensor, all_targets: torch.Tensor):
    """
    Print detailed evaluation results.
    all_predictions are probabilities in [0,1]
    """
    print("\n" + "="*60)
    print("TEST SET EVALUATION RESULTS")
    print("="*60)
    print(f"Total samples: {all_targets.size(0)}")
    print(f"Loss (KL Divergence): {metrics['loss']:.6f}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.6f}")
    print(f"KL Divergence: {metrics['kl']:.6f}")
    print(f"Top-1 Accuracy (|pred-gt|<0.1 at argmax): {metrics['top1']:.4f}")
    print(f"Top-3 Accuracy (any top-3 |pred-gt|<0.3): {metrics['top3']:.4f}")
    print(f"Threshold 0.1 Accuracy (avg): {metrics['01']:.4f}")
    print(f"Threshold 0.2 Accuracy (avg): {metrics['02']:.4f}")
    print(f"Threshold 0.3 Accuracy (avg): {metrics['03']:.4f}")
    print(f"Threshold 0.1 Accuracy (strict): {metrics['01_strict']:.4f}")
    print(f"Threshold 0.2 Accuracy (strict): {metrics['02_strict']:.4f}")
    print(f"Threshold 0.3 Accuracy (strict): {metrics['03_strict']:.4f}")
    print(f"Threshold 0.1 Accuracy (non-zero): {metrics['01_nonzero']:.4f}")
    print(f"Threshold 0.2 Accuracy (non-zero): {metrics['02_nonzero']:.4f}")
    print(f"Threshold 0.3 Accuracy (non-zero): {metrics['03_nonzero']:.4f}")
    
    # Per-ingredient analysis
    print("\n" + "-"*60)
    print("PER-INGREDIENT ANALYSIS")
    print("-"*60)
    
    pred_probs = all_predictions  # already probabilities
    
    print(f"{'Ingredient':15s} | {'MAE':>6s} | {'MSE':>8s} | {'Non-zero GTs':>12s} | {'Within 0.2':>10s}")
    print("-" * 70)
    
    for i, ingredient in enumerate(INGREDIENTS):
        pred_i = pred_probs[:, i]
        target_i = all_targets[:, i]
        
        mae = torch.mean(torch.abs(pred_i - target_i)).item()
        mse = torch.mean((pred_i - target_i) ** 2).item()
        
        non_zero_mask = target_i > 0.01
        num_non_zero = non_zero_mask.sum().item()
        
        if num_non_zero > 0:
            nz_preds = pred_i[non_zero_mask]
            nz_targets = target_i[non_zero_mask]
            within_threshold = torch.abs(nz_preds - nz_targets) <= 0.2
            num_within_threshold = within_threshold.sum().item()
            accuracy_within = (num_within_threshold / num_non_zero) * 100
            within_str = f"{num_within_threshold}/{num_non_zero} ({accuracy_within:.1f}%)"
        else:
            within_str = "N/A"
        
        print(f"{ingredient:15s} | {mae:6.4f} | {mse:8.6f} | {int(num_non_zero):>12d} | {within_str:>10s}")
    
    # Combination analysis
    print("\n" + "-"*60)
    print("INGREDIENT COMBINATION ANALYSIS")
    print("-"*60)
    
    combination_analysis = analyze_combinations(pred_probs, all_targets)
    
    print(f"Training Combinations:")
    training_info = combination_analysis['training_combinations']
    print(f"  Total unique combinations: {training_info['count']}")
    print(f"  Total samples: {training_info['total_samples']}")
    print(f"  Threshold 0.1 Accuracy (non-zero): {training_info['accuracy_01']:.4f}")
    print(f"  Threshold 0.2 Accuracy (non-zero): {training_info['accuracy_02']:.4f}")
    print(f"  Threshold 0.3 Accuracy (non-zero): {training_info['accuracy_03']:.4f}")
    
    print(f"\nUnseen Combinations:")
    unseen_info = combination_analysis['unseen_combinations']
    print(f"  Total unique combinations: {unseen_info['count']}")
    print(f"  Total samples: {unseen_info['total_samples']}")
    print(f"  Threshold 0.1 Accuracy (non-zero): {unseen_info['accuracy_01']:.4f}")
    print(f"  Threshold 0.2 Accuracy (non-zero): {unseen_info['accuracy_02']:.4f}")
    print(f"  Threshold 0.3 Accuracy (non-zero): {unseen_info['accuracy_03']:.4f}")
    
    # Show top combinations by sample count
    print(f"\nTop Training Combinations (by sample count):")
    training_sorted = sorted(training_info['details'].items(), key=lambda x: x[1]['count'], reverse=True)
    for combo, details in training_sorted[:10]:
        combo_str = ", ".join(combo) if combo else "None"
        print(f"  {combo_str:35s}: {details['count']:>3d} samples, 0.1 acc: {details['accuracy_01']:.3f}, 0.2 acc: {details['accuracy_02']:.3f}")
    
    if unseen_info['details']:
        print(f"\nTop Unseen Combinations (by sample count):")
        unseen_sorted = sorted(unseen_info['details'].items(), key=lambda x: x[1]['count'], reverse=True)
        for combo, details in unseen_sorted[:10]:
            combo_str = ", ".join(combo) if combo else "None"
            print(f"  {combo_str:35s}: {details['count']:>3d} samples, 0.1 acc: {details['accuracy_01']:.3f}, 0.2 acc: {details['accuracy_02']:.3f}")
    
    # Prediction examples
    print("\n" + "-"*60)
    print("PREDICTION EXAMPLES (Ground Truth vs Model Prediction)")
    print("-"*60)
    print(f"{'Ingredient':>12s} | {'GT':>6s} | {'Pred':>6s} | {'Diff':>6s}")
    print("-" * 40)
    
    sample_idxs = random.sample(range(all_targets.size(0)), min(5, all_targets.size(0)))
    for sample_idx in sample_idxs:
        print(f"\nSample {sample_idx + 1}:")
        gt_sample = all_targets[sample_idx]
        pred_sample = pred_probs[sample_idx]
        for i, ingredient in enumerate(INGREDIENTS):
            gt_val = gt_sample[i].item()
            pred_val = pred_sample[i].item()
            diff = abs(gt_val - pred_val)
            print(f"{ingredient:>12s} | {gt_val:6.3f} | {pred_val:6.3f} | {diff:6.3f}")


def main():
    """
    Main evaluation function.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate scent model')
    parser.add_argument('--test-set', choices=['seen', 'unseen'], default='seen',
                        help='Which test set to evaluate on (default: seen)')
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    
    # Select test index based on argument
    test_index = TEST_INDEX if args.test_set == 'seen' else TEST_INDEX2
    
    print(f"[1/4] Loading {args.test_set} test dataset...")
    if test_index == TEST_INDEX:
        pairs = load_smell_recognition_data(TEST_DIR)
    else:
        pairs = load_smell_recognition_data(TEST_DIR2)
    test_dataset = SmellDataset(pairs, max_len=600)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("[2/4] Loading trained model...")
    model = SmellTemporalCNN()

    # model = build_model(input_shape=(WIN_LEN, D), n_classes=len(INGREDIENTS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print(f"    -> Model loaded from {MODEL_PATH}")
    print(f"    -> Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("[3/4] Loading scalers...")
    # scalers = load(SCALERS_PATH)
    # print(f"    -> Scalers loaded from {SCALERS_PATH}")
    
    print("[4/4] Evaluating model...")
    metrics, all_predictions, all_targets = evaluate_model(model, test_loader)
    
    # Print results
    print_detailed_results(metrics, all_predictions, all_targets)
    
    # Save results
    results_dir = Path(BASE_DIR) / "evaluation_results"
    results_dir.mkdir(exist_ok=True)
    
    results_filename = f"test_{args.test_set}_evaluation_results.json"
    results_path = results_dir / results_filename
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[OK] Results saved to: {results_path}")
    
    # Optionally save predictions/targets
    # np.save(results_dir / "test_predictions.npy", all_predictions.numpy())
    # np.save(results_dir / "test_targets.npy", all_targets.numpy())

if __name__ == "__main__":
    main()
