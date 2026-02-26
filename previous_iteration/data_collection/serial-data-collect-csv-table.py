#!/usr/bin/env python

import argparse
import os
import uuid
import threading
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ========== SETTINGS ==========
DEFAULT_BAUD = 115200
DEFAULT_LABEL = "_unknown"
features = ["NO2", "C2H50H", "VOC", "CO", "Alcohol", "LPG", "Benzene",
            "Temperature", "Pressure", "Humidity", "Gas_Resistance", "Altitude"]
num_features = len(features)

# ========== GLOBAL BUFFERS ==========
uid = str(uuid.uuid4())[-12:]
rx_buf = b''
all_times = []
feature_data = {f: [] for f in features}

# ========== FILE WRITING FUNCTION ==========
def write_csv(data, directory, label):
    filename = label + "." + uid + ".csv"
    out_path = os.path.join(directory, filename)
    with open(out_path, 'a+') as file:
        file.write(data)
    print("Data written to:", out_path)

# ========== SERIAL READING LOOP ==========
def serial_read_loop():
    global rx_buf, all_times, feature_data
    while True:
        if ser.in_waiting > 0:
            while ser.in_waiting:
                rx_buf += ser.read()
                if rx_buf[-2:] == b'\r\n':
                    buf_str = rx_buf.decode('utf-8').strip()
                    try:
                        with open('state.txt', 'r') as f:
                            state = f.read()
                    except FileNotFoundError:
                        state = ''
                    buf_str = buf_str.replace('\r', '') + state + '\n'
                    write_csv(buf_str, out_dir, label)
                    values = buf_str.strip().split(',')
                    if len(values) >= num_features + 1:
                        all_times.append(len(all_times))  # Use sample index
                        for i, feature in enumerate(features):
                            try:
                                feature_data[feature].append(float(values[i + 1]))
                            except ValueError:
                                feature_data[feature].append(0.0)
                    rx_buf = b''

# ========== COMMAND-LINE ARGUMENTS ==========
parser = argparse.ArgumentParser(description="Serial Data Collection CSV")
parser.add_argument('-p', '--port', dest='port', type=str, required=True, help="Serial port to connect to")
parser.add_argument('-b', '--baud', dest='baud', type=int, default=DEFAULT_BAUD, help=f"Baud rate (default = {DEFAULT_BAUD})")
parser.add_argument('-d', '--directory', dest='directory', type=str, default=".", help="Output directory (default = .)")
parser.add_argument('-l', '--label', dest='label', type=str, default=DEFAULT_LABEL, help=f"Label for files (default = {DEFAULT_LABEL})")

print("\nAvailable serial ports:")
available_ports = serial.tools.list_ports.comports()
for port, desc, hwid in sorted(available_ports):
    print(f"  {port} : {desc} [{hwid}]")

args = parser.parse_args()
port = args.port
baud = args.baud
out_dir = args.directory
label = args.label

try:
    os.makedirs(out_dir)
except FileExistsError:
    pass

# ========== SET UP SERIAL ==========
ser = serial.Serial()
ser.port = port
ser.baudrate = baud

try:
    ser.open()
except Exception as e:
    print("ERROR:", e)
    exit()
print(f"\nConnected to {port} at baud rate {baud}")
print("Press 'ctrl+c' to exit")

# ========== SET UP PLOT ==========
fig, axes = plt.subplots(num_features, 1, figsize=(10, 2 * num_features), sharex=True)
if num_features == 1:
    axes = [axes]
lines = []
for i, feature in enumerate(features):
    ax = axes[i]
    line, = ax.plot([], [], label=feature)
    ax.set_ylabel(feature[:4])
    ax.legend(loc='upper right')
    lines.append(line)
axes[-1].set_xlabel("Sample #")

def update_plot(frame):
    x_data = list(range(len(all_times)))
    for i, feature in enumerate(features):
        lines[i].set_data(x_data, feature_data[feature])
        axes[i].set_xlim(0, len(all_times) if len(all_times) > 10 else 10)
        axes[i].relim()
        axes[i].autoscale_view()
    return lines

# ========== START SERIAL READER THREAD ==========
serial_thread = threading.Thread(target=serial_read_loop, daemon=True)
serial_thread.start()

# ========== START PLOT ==========
header = "timestamp," + ",".join(features) + ",State\n"
write_csv(header, out_dir, label)

ani = animation.FuncAnimation(fig, update_plot, interval=1000)
plt.tight_layout()
plt.show()

# ========== CLEANUP ==========
print("Closing serial port")
ser.close()
