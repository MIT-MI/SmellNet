#!/usr/bin/env python
"""
Serial Data Collection CSV

Collects raw data in CSV form over a serial connection and saves them to files.

Install dependencies:

    python -m pip install pyserial

The first line should be header information. Each sample should be on a newline.
Here is a raw accelerometer data sample (in m/s^2):

    accX,accY,accZ
    -0.22,0.82,10.19
    -0.05,0.77,9.63
    -0.01,1.10,8.50
    ...

The end of the sample should contain an empty line (e.g. \r\n\r\n).

Call this script as follows:

    python serial-data-collect-csv.py
    
Author: Shawn Hymel (EdgeImpulse, Inc.)
Date: June 17, 2022
License: Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""

import argparse
import os
import uuid
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import time

# Third-party libraries
import serial
import serial.tools.list_ports

# Settings
DEFAULT_BAUD = 115200       # Must match transmitting program baud rate
DEFAULT_LABEL = "_unknown"  # Label prepended to all CSV files

# Generate unique ID for file (last 12 characters from uuid4 method)
uid = str(uuid.uuid4())[-12:]

class SensorDataEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class GCMSDataEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
def evaluate_retrieval(smell_matrix, gcms_data, gcms_encoder, sensor_encoder, device='cpu'):
    gcms_encoder.eval()
    sensor_encoder.eval()
    smell_matrix = torch.tensor(smell_matrix, dtype=torch.float)
    gcms_data = torch.tensor(gcms_data, dtype=torch.float)

    with torch.no_grad():
        gcms_data = gcms_data.to(device)
        smell_matrix = smell_matrix.to(device)

        z_gcms = gcms_encoder(gcms_data) # (15 x 16)
        z_sensor = sensor_encoder(smell_matrix) # (n x 16)

        # L2 normalize if that’s how your model was trained
        z_gcms = F.normalize(z_gcms, dim=1)
        z_sensor = F.normalize(z_sensor, dim=1)

    # Compute similarity matrix: shape [N, N]
    # sim[i, j] = dot( z_gcms[i], z_sensor[j] )
    sim = torch.matmul(z_sensor, z_gcms.t()) # (n x 15)
    
    # For each row i, find the column j with the highest similarity
    # If j == i, it means we matched the correct sensor embedding
    # predicted = sim.argmax()  # [N]
    return sim.numpy()

# Create a file with unique filename and write CSV data to it
def write_csv(data, dir, label):

    # Keep trying if the file exists
    # exists = True
    # while exists:
    filename = label + "." + uid + ".csv"
    
    # Create and write to file if it does not exist
    out_path = os.path.join(dir, filename)

    with open(out_path, 'a+') as file:
        lines = data.split(',')
        if len(lines) == 14:
            file.write(data)
    # print("Data written to:", out_path)
        # if not os.path.exists(out_path):
        #     exists = False
        #     try:
        #     except IOError as e:
        #         print("ERROR", e)
        #         return
    df = pd.read_csv(out_path)
    if df.shape[1] > 14:
        df.drop(df.columns[[14,]], inplace=True, axis=1)
    df.to_csv(out_path, index=False)
    return out_path
    

# Command line arguments
parser = argparse.ArgumentParser(description="Serial Data Collection CSV")
parser.add_argument('-p',
                    '--port',
                    dest='port',
                    type=str,
                    required=True,
                    help="Serial port to connect to")
parser.add_argument('-b',
                    '--baud',
                    dest='baud',
                    type=int,
                    default=DEFAULT_BAUD,
                    help="Baud rate (default = " + str(DEFAULT_BAUD) + ")")
parser.add_argument('-d',
                    '--directory',
                    dest='directory',
                    type=str,
                    default=".",
                    help="Output directory for files (default = .)")
parser.add_argument('-l',
                    '--label',
                    dest='label',
                    type=str,
                    default=DEFAULT_LABEL,
                    help="Label for files (default = " + DEFAULT_LABEL + ")")
                    
# Print out available serial ports
print()
print("Available serial ports:")
available_ports = serial.tools.list_ports.comports()
for port, desc, hwid in sorted(available_ports):
    print("  {} : {} [{}]".format(port, desc, hwid))
    
# Parse arguments
args = parser.parse_args()
port = args.port
baud = args.baud
out_dir = args.directory
label = args.label

# Configure serial port
ser = serial.Serial()
ser.port = port
ser.baudrate = baud

# Attempt to connect to the serial port
try:
    ser.open()
except Exception as e:
    print("ERROR:", e)
    exit()
print()
print("Connected to {} at a baud rate of {}".format(port, baud))
print("Press 'ctrl+c' to exit")

# Serial receive buffer
rx_buf = b''

# Make output directory
try:
    os.makedirs(out_dir)
except FileExistsError:
    pass

# Loop forever (unless ctrl+c is captured)

gcms_model_path = "/Users/christoumedialab/Downloads/demo_gcms_encoder_2025-04-04 18_09_38.637551.pt"
sensor_model_path = "/Users/christoumedialab/Downloads/demo_sensor_encoder_2025-04-04 18_09_38.637643.pt"

hidden_dim = 128
embedding_dim = 16
# Initialize the model again (must match original architecture)
gcms_encoder = GCMSDataEncoder(10, hidden_dim, embedding_dim)
sensor_encoder = SensorDataEncoder(7, hidden_dim, embedding_dim)
# Load the saved parameters
gcms_encoder.load_state_dict(torch.load(gcms_model_path))
sensor_encoder.load_state_dict(torch.load(sensor_model_path))

df = pd.read_csv("/Users/christoumedialab/Desktop/Smell_Research/smell_sensors/csv_collect/gcms_dataframe.csv")

# adding ambient to the df
ambient_row = pd.DataFrame([{'food_name': "ambient", 'C': 0.04, "Ca": 0, "H": 0.00005, "K": 0, "Mg": 0, "N": 78.08, "Na":0, "O": 20.95, "P": 0, "Se":0,}])

df = pd.concat([df, ambient_row], ignore_index=True)

# getting rid of names and keeping only numerical values
top_5_food = ["oregano", "cumin", "basil", "ambient", "peanuts"]
# {0: 'basil', 1: 'cloves', 2: 'cumin', 3: 'oregano', 4: 'ambient'}

df = df[df['food_name'].isin(top_5_food)]

df_dropped = df.drop(columns=["food_name"], errors="ignore")

gcms_data = df_dropped.values

scaler = StandardScaler()

scaler.fit(gcms_data)

gcms_data = scaler.transform(gcms_data)

available_food_names = df["food_name"].to_list()

ix_to_name = {0: 'cloves', 1: 'cumin', 2: 'oregano', 3: 'ambient', 4: 'peanuts'}
name_to_ix = {name: i for i, name in enumerate(available_food_names)}

data_mean = np.array([2.70191194e+00, 1.89716107e+00, 3.78447277e+00, 3.80648899e-01,
       5.67786790e-02, 1.51796060e-01, 6.22098392e+07])

data_std = np.array([5.84832099e+00, 4.33619372e+00, 5.82249125e+00, 2.08146843e+00,
       2.49279793e+00, 2.54532829e+00, 5.43490188e+08])

def standardize_values(data, mean, std):
    return (data - mean) / std


"/Users/christoumedialab/Desktop/Smell_Research/smell_sensors/csv_collect"
print("Model loaded successfully!")

header = "timestamp,NO2,C2H50H,VOC,CO,Alcohol,LPG,Benzene,Temperature,Pressure,Humidity,Gas_Resistance,Altitude,State"

# Initialize a history buffer for the last 50 predictions
prediction_history = deque(maxlen=50)
ingredients = ['cloves', 'Cumin', 'Oregano', 'Ambient', 'Peanuts']
colors = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#bd7ebe"]

# Set up the plot
plt.ion()  # Interactive mode on
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Real-time Smell Prediction Trends')
ax.set_xlabel('Time (samples)')
ax.set_ylabel('Similarity Score')
ax.set_ylim(-1, 1) 
ax.grid(True)
ax.set_xticklabels([])

# Create lines for each ingredient
lines = []
for i, (ing, color) in enumerate(zip(ingredients, colors)):
    line, = ax.plot([], [], label=ing, color=color, linewidth=3)
    lines.append(line)

line_labels = []
for i, line in enumerate(lines):
    ingredient_label = ax.text(0, 0, ingredients[i], color=colors[i], 
                   fontsize=10, backgroundcolor=(1,1,1,0.7))
    line_labels.append(ingredient_label)

pbar = tqdm(total=50, desc="Collecting data", position=0, leave=False)

write_csv(header, out_dir, label)
try:
    while True:
        
        # Read bytes from serial port
        if ser.in_waiting > 0:
            while(ser.in_waiting):
                
                # Read bytes
                rx_buf += ser.read()
                
                # Look for an empty line
                if rx_buf[-2:] == b'\r\n':

                    # Strip extra newlines (convert \r\n to \n)
                    buf_str = rx_buf.decode('utf-8').strip()

                    with open('state.txt', 'r') as f:
                        state = f.read()
                    buf_str = buf_str.replace('\r', '') + state + '\n'

                    # Write contents to file
                    out_path = write_csv(buf_str, out_dir, label)
                    if os.path.exists(out_path):
                        substance_df = pd.read_csv(out_path)
                        substance_df.drop(columns=["timestamp", "Temperature", "Pressure", "Humidity", "Gas_Resistance", "Altitude", "State"], inplace=True)
                        if len(substance_df) > 50:
                            diff_data = substance_df.diff(periods=50)  # This is the key change
                            
                            sensor_cols = [col for col in diff_data.columns if col not in ["State", "label"]]
                            diff_data = diff_data[~(diff_data[sensor_cols] == 0).all(axis=1)]
                            diff_data = diff_data.iloc[50:]
                            substance_values = diff_data.values[-10:]

                            standardized = standardize_values(substance_values, data_mean, data_std)
                            
                            predicted_from_smell = evaluate_retrieval(standardized, gcms_data, gcms_encoder, sensor_encoder)
                            current_pred = predicted_from_smell[-1]
                            prediction_history.append(current_pred)
                            
                            # Update plot data
                            x_data = range(len(prediction_history))
                            for i, line in enumerate(lines):
                                y_data = [pred[i] for pred in prediction_history]
                                line.set_data(x_data, y_data)
                            
                            for i, (line, ingredient_label) in enumerate(zip(lines, line_labels)):
                                y_data = [pred[i] for pred in prediction_history]
                                line.set_data(x_data, y_data)
                                
                                # Update label position to follow the end of the line
                                if len(y_data) > 0:
                                    ingredient_label.set_position((x_data[-1], y_data[-1]))
                                    ingredient_label.set_text(f"{ingredients[i]}: {y_data[-1]:.2f}")
                            
                            # Update title and current prediction display
                            current_ingredient = ingredients[np.argmax(current_pred)]
                            current_score = np.max(current_pred)
                            ax.set_title(f'Real-time Smell Prediction - Current: {current_ingredient} ({current_score:.2f})', 
                                        fontsize=14, pad=20)
                            
                            # Update prediction text box
                            pred_text = "Current Prediction:\n"
                            for ing, score in zip(ingredients, current_pred):
                                pred_text += f"{ing}: {score:.3f}\n"
                            
                            # Adjust axes
                            ax.set_xlim(0, max(50, len(prediction_history)))
                            fig.canvas.draw()
                            fig.canvas.flush_events()
                        else:
                            # Create a progress bar showing how close we are to reaching 50 rows
                            pbar.n = len(substance_df)
                            pbar.refresh()
                            time.sleep(0.1)  # Adjust sleep time as needed
                    rx_buf = b''


# Look for keyboard interrupt (ctrl+c)
except KeyboardInterrupt:
    pass

# Close serial port
print("Closing serial port")
ser.close()