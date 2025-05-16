import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from openTSNE import TSNE
import umap
from collections import defaultdict

import os

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

sensor_columns = [
    'NO2', 'C2H5OH', 'VOC', 'CO', 'Alcohol', 'LPG', 'Benzene',
    'Temperature', 'Pressure', 'Humidity', 'Gas_Resistance', 'Altitude'
]

def load_data(training_path, testing_path):
    training_data = defaultdict(list)
    testing_data = defaultdict(list)

    min_len = float('inf')  # Track minimum length across all series

    # Walk through the training directory
    for folder_name in os.listdir(training_path):
        folder_path = os.path.join(training_path, folder_name)
        
        if os.path.isdir(folder_path):  # Make sure it's a folder
            for filename in os.listdir(folder_path):
                if filename.endswith(".csv"):
                    cur_path = os.path.join(folder_path, filename)
                    df = pd.read_csv(cur_path)
                    training_data[folder_name].append(df)
                    min_len = min(min_len, df.shape[0])  # Update minimum length

    for folder_name in os.listdir(testing_path):
        folder_path = os.path.join(testing_path, folder_name)
        
        if os.path.isdir(folder_path):  # Make sure it's a folder
            for filename in os.listdir(folder_path):
                if filename.endswith(".csv"):
                    cur_path = os.path.join(folder_path, filename)
                    df = pd.read_csv(cur_path)
                    testing_data[folder_name].append(df)
                    min_len = min(min_len, df.shape[0])  # Update minimum length

    return training_data, testing_data, min_len

def aggregate_data(training_data, testing_data):
    aggregated_training = []
    aggregated_testing = []

    # Aggregate training data
    for ingredient, dfs in training_data.items():
        for i, df in enumerate(dfs):
            df = df.copy()  # Safe copy
            df['ingredient'] = ingredient
            df['file_id'] = f"{ingredient}_train_{i}"
            df['time_step'] = range(len(df))
            aggregated_training.append(df)

    # Aggregate testing data
    for ingredient, dfs in testing_data.items():
        for i, df in enumerate(dfs):
            df = df.copy()
            df['ingredient'] = ingredient
            df['file_id'] = f"{ingredient}_test_{i}"
            df['time_step'] = range(len(df))
            aggregated_testing.append(df)

    # Concatenate into final DataFrames
    aggregated_training = pd.concat(aggregated_training, ignore_index=True)
    aggregated_testing = pd.concat(aggregated_testing, ignore_index=True)

    # Map the ingredient to category
    aggregated_training['category'] = aggregated_training['ingredient'].map(ingredient_to_category)
    aggregated_testing['category'] = aggregated_testing['ingredient'].map(ingredient_to_category)
    return aggregated_training, aggregated_testing

def prepare_features(df, feature_cols):
    print("Preparing features...")
    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, df['ingredient']

def run_pca(X_scaled):
    print("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
    return X_pca

def run_umap(X_scaled):
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    return X_umap

def run_tsne(X_scaled):
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, n_jobs=-1, verbose=True)
    X_tsne = tsne.fit(X_scaled)
    return X_tsne

def plot_embedding(X_embedded, labels, title, save_path):
    print(f"Plotting {title}...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1],
                    hue=labels, palette='tab20', s=10, legend=False)
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    training_path = "/home/dewei/workspace/smell-net/training"
    testing_path = "/home/dewei/workspace/smell-net/testing"

    output_dir = "/home/dewei/workspace/smell-net/data_stats"

    training_data, testing_data, min_len = load_data(training_path, testing_path)

    aggregated_training, aggregated_testing = aggregate_data(training_data, testing_data)

    os.makedirs(output_dir, exist_ok=True)

    X_scaled, labels = prepare_features(aggregated_training, sensor_columns)

    # X_pca = run_pca(X_scaled)
    # plot_embedding(X_pca, labels, "PCA Projection", os.path.join(output_dir, "pca_plot.png"))

    # X_umap = run_umap(X_scaled)
    # plot_embedding(X_umap, labels, "UMAP Projection", os.path.join(output_dir, "umap_plot.png"))

    X_tsne = run_tsne(X_scaled)
    plot_embedding(X_tsne, labels, "t-SNE Projection", os.path.join(output_dir, "tsne_plot.png"))

    print(f"All plots saved in {output_dir}")

if __name__ == "__main__":
    main()
