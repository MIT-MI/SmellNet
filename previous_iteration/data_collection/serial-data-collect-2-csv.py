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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Third-party libraries
import serial
import serial.tools.list_ports

# Settings
DEFAULT_BAUD = 115200       # Must match transmitting program baud rate
DEFAULT_LABEL = "_unknown"  # Label prepended to all CSV files

# Generate unique ID for file (last 12 characters from uuid4 method)
uid = str(uuid.uuid4())[-12:]

class ClassifierNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # No softmax here because CrossEntropyLoss applies it automatically
        return x
    
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
    print("Data written to:", out_path)
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

def create_state_average_df(df):
    df["Group"] = (df["State"] != df["State"].shift()).cumsum()
    averaged_df = df.groupby("Group").mean().reset_index()
    averaged_df["State"] = df.groupby("Group")["State"].first().values
    averaged_df = averaged_df.drop(columns=["Group"])
    averaged_df = averaged_df[averaged_df["State"] < 2]
    averaged_df.reset_index(drop=True)
    return averaged_df


def calculate_state_difference(df):
    if len(df) % 2 != 0:
        df = df[:-1]
    odd_rows = df.iloc[1::2].reset_index(drop=True)
    even_rows = df.iloc[0::2].reset_index(drop=True)
    result = odd_rows - even_rows
    return result

# Initialize the model again (must match original architecture)
model = ClassifierNN(input_size=13, num_classes=3)
# Load the saved parameters
model.load_state_dict(torch.load("/Users/christoumedialab/Downloads/model-5.pth"))
model.eval()  # Set model to evaluation mode
print("Model loaded successfully!")

header = "timestamp,NO2,C2H50H,VOC,CO,Alcohol,LPG,Benzene,Temperature,Pressure,Humidity,Gas_Resistance,Altitude,State"
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
                    # print(state)
                    buf_str = buf_str.replace('\r', '') + state + '\n'

                    # Write contents to file
                    out_path = write_csv(buf_str, out_dir, label)
                    if os.path.exists(out_path):
                        substance_df = pd.read_csv(out_path)
                        substance_df.drop(columns="timestamp", inplace=True)
                        if substance_df['State'].isin([0, 1]).all():
                            avg_substance_df = create_state_average_df(substance_df)
                            diff_substance_df = calculate_state_difference(avg_substance_df)
                            substance_values = diff_substance_df.values
                            
                            if substance_values.shape[0] != 0 and substance_values.shape[1] != 0:
                                # loaded_pipeline = joblib.load("/Users/christoumedialab/Downloads/pca_model_pipeline-2.pkl")
                                # predictions = loaded_pipeline.predict(substance_values)
                                with torch.no_grad():
                                    sample_input = torch.tensor(substance_values, dtype=torch.float32)  # Example test data
                                    predictions = model(sample_input)
                                    _, predicted_labels = torch.max(predictions, 1)

                                    # label_mapping = ["ambient", "alcohol", "apple_juice", "balsamic_vinegar", "basil", 
                                    #                  "black_pepper", "cayenne", "chili_powder", "cinnamon", "cloves", 
                                    #                  "coffee_beans", "cumin", "garlic_powder", "matcha", "mediterranean_blend", 
                                    #                  "mint_leaves", "nutmeg", "olive_oil", "onion_powder", "oregano", "paprika", 
                                    #                  "rosemary", "soybean_oil", "vanilla_extract"]
                                    label_mapping = ["ambient", "alcohol", "coffee_beans"]
                                    translated_labels = [label_mapping[idx] for idx in predicted_labels.numpy()]
                                    print("Predicted Labels:", translated_labels)

                    rx_buf = b''


# Look for keyboard interrupt (ctrl+c)
except KeyboardInterrupt:
    pass

# Close serial port
print("Closing serial port")
ser.close()