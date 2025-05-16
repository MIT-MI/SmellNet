from models import *
from load_data import *
from train import *
from evaluate import *
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
import time
from dataset import *
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib.ticker as ticker

ingredient_to_category = {
    # Nuts
    "peanuts": "Nuts",
    "cashew": "Nuts",
    "chestnuts": "Nuts",
    "pistachios": "Nuts",
    "almond": "Nuts",
    "hazelnut": "Nuts",
    "walnuts": "Nuts",
    "pecans": "Nuts",
    "brazil_nut": "Nuts",
    "pili_nut": "Nuts",
    
    # Spices
    "cumin": "Spices",
    "star_anise": "Spices",
    "nutmeg": "Spices",
    "cloves": "Spices",
    "ginger": "Spices",
    "allspice": "Spices",
    "chervil": "Spices",
    "mustard": "Spices",
    "cinnamon": "Spices",
    "saffron": "Spices",
    
    # Herbs
    "angelica": "Herbs",
    "garlic": "Herbs",
    "chives": "Herbs",
    "turnip": "Herbs",
    "dill": "Herbs",
    "mugwort": "Herbs",
    "chamomile": "Herbs",
    "coriander": "Herbs",
    "oregano": "Herbs",
    "mint": "Herbs",
    
    # Fruits
    "kiwi": "Fruits",
    "pineapple": "Fruits",
    "banana": "Fruits",
    "lemon": "Fruits",
    "mandarin_orange": "Fruits",
    "strawberry": "Fruits",
    "apple": "Fruits",
    "mango": "Fruits",
    "peach": "Fruits",
    "pear": "Fruits",
    
    # Vegetables
    "cauliflower": "Vegetables",
    "brussel_sprouts": "Vegetables",
    "broccoli": "Vegetables",
    "sweet_potato": "Vegetables",
    "asparagus": "Vegetables",
    "avocado": "Vegetables",
    "radish": "Vegetables",
    "tomato": "Vegetables",
    "potato": "Vegetables",
    "cabbage": "Vegetables",
}

log_dir = "/home/dewei/workspace/smell-net/logs"

log_file_path = os.path.join(log_dir, f"{time.time()}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(),
    ],
)

category_colors = {
    "Nuts": "#d8a65880",        # muted orange with 50% transparency
    "Spices": "#b195c980",      # muted purple with 50% transparency
    "Herbs": "#90b49480",       # muted green with 50% transparency
    "Fruits": "#d27a7a80",      # muted red with 50% transparency
    "Vegetables": "#7da7d980"   # muted blue with 50% transparency
}

# ingredients = sorted(total_counts.keys())
# accuracies = [correct_counts[ing] / total_counts[ing] for ing in ingredients]
# colors = [category_colors[ingredient_to_category[ing]] for ing in ingredients]


def plot_per_ingredient_accuracy(sorted_ingredients, sorted_accuracies, sorted_colors, ingredient_to_category, category_colors):
    # Set figure and axis
    sorted_ingredients = [ingredient.replace("_", " ").capitalize() for ingredient in sorted_ingredients]
    fig, ax = plt.subplots(figsize=(16, 8))

    # Bar plot
    bars = ax.bar(sorted_ingredients, sorted_accuracies, color=sorted_colors)

    # Add value labels on top of bars
    for bar, acc in zip(bars, sorted_accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{acc:.2f}",
            ha='center',
            va='bottom',
            fontsize=11,
            rotation=90,
        )

    # Axis formatting
    ax.set_ylabel("Top-1 Accuracy", fontsize=20, labelpad=15)
    ax.set_title("Per-Ingredient Accuracy by Category", fontsize=24, pad=20, fontweight='bold')
    ax.set_xticks(range(len(sorted_ingredients)))
    ax.set_xticklabels(sorted_ingredients, rotation=65, ha='right', fontsize=18)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.1f}"))
    ax.tick_params(axis='y', labelsize=14)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in category_colors.values()]
    ax.legend(handles, category_colors.keys(), title="Category", fontsize=16, title_fontsize=15, loc='upper right')

    # Style
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig("sorted_per_ingredient_accuracy_aesthetic.png", dpi=300)
    plt.show()
    
def main():
    logger = logging.getLogger()

    training_path = "/home/dewei/workspace/smell-net/training"
    testing_path = "/home/dewei/workspace/smell-net/testing"
    real_time_testing_path = "/home/dewei/workspace/smell-net/real_time_testing_spice"
    gcms_path = "/home/dewei/workspace/smell-net/processed_full_gcms_dataframe.csv"

    period_len = 50

    # Load all ingredients (remove ingredients=["cashew"])
    training_data, testing_data, real_time_testing_data, min_len = load_sensor_data(
        training_path, testing_path, real_time_testing_path=real_time_testing_path
    )

    gcms_scaled, y_encoded, le, scaler = load_gcms_data(gcms_path)

    training_data, training_label, _ = prepare_data_gradient(training_data, period_len=period_len, le=le)
    testing_data, testing_label, _ = prepare_data_gradient(testing_data, period_len=period_len, le=le)

    training_pair_data, _ = create_pair_data(training_data, training_label, gcms_scaled, le)

    train_dataset = PairedDataset(training_pair_data)
    sensor_model = Encoder(input_dim=12, output_dim=32)
    gcms_model = Encoder(input_dim=17, output_dim=32)

    # Load pretrained models
    sensor_model.load_state_dict(torch.load(f'saved_models/contrastive/gradient_period_{period_len}_sensor_model_weights.pth'))
    gcms_model.load_state_dict(torch.load(f'saved_models/contrastive/gradient_period_{period_len}_gcms_model_weights.pth'))

    # Get predictions for all test data
    _, _, top5_pred, top5_sim = contrastive_evaluate(testing_data, gcms_scaled, testing_label, gcms_model, sensor_model, logger)


    # Calculate per-ingredient Top-1 accuracy
    total_counts = defaultdict(int)
    correct_counts = defaultdict(int)

    for true_label, pred_label in zip(testing_label, top5_pred):
        ingredient = le.inverse_transform([true_label])[0]
        pred_ingredient = le.inverse_transform([pred_label[0]])[0]
        total_counts[ingredient] += 1
        if ingredient == pred_ingredient:
            correct_counts[ingredient] += 1

    # Compute per-ingredient accuracy from your evaluation
    ingredients = sorted(total_counts.keys())
    accuracies = [correct_counts[ing] / total_counts[ing] for ing in ingredients]

    # Zip and sort by accuracy descending
    sorted_data = sorted(zip(ingredients, accuracies), key=lambda x: x[1], reverse=True)
    sorted_ingredients, sorted_accuracies = zip(*sorted_data)
    sorted_colors = [category_colors[ingredient_to_category[ing]] for ing in sorted_ingredients]

    plot_per_ingredient_accuracy(sorted_ingredients, sorted_accuracies, sorted_colors, ingredient_to_category, category_colors)

if __name__ == "__main__":
    main()

