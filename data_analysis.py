import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

training_data = defaultdict(list)
testing_data = defaultdict(list)

training_path = "/Users/derre/Documents/workspace/smell-net/training"
testing_path = "/Users/derre/Documents/workspace/smell-net/testing"
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

