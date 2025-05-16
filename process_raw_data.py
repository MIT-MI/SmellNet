from collections import defaultdict
import os
import pandas as pd

root_dir = "/home/dewei/workspace/smell-net/real_time_testing_spice"
test_dir = "/home/dewei/workspace/smell-net/processed_real_time_testing_spice"

data_paths = defaultdict(list)
min_len = float('inf')  # Track minimum length across all series

ingredients = defaultdict(list)

# Load and trim data
for root, dirs, files in os.walk(root_dir):
    for filename in files:
        if filename.endswith(".csv"):
            cur_path = os.path.join(root, filename)
            df = pd.read_csv(cur_path)

            df = df[df.columns[1:13]]

            cur_ingredient = filename.split(".")[0]

            ingredients[cur_ingredient].append((filename, df))


for ingredient in ingredients:
    for filename, df in ingredients[ingredient]:
        os.makedirs(os.path.join(test_dir, ingredient), exist_ok=True)
        
        df.to_csv(os.path.join(test_dir, ingredient, filename), index=False)
