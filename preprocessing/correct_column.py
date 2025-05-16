import os
import pandas as pd

# Path to your dataset
data_dir = "/home/dewei/workspace/SmellNet/data/offline_testing"

# Define column name corrections
column_renames = {
    "C2H50H": "C2H5OH",
}

# Walk through all subdirectories
for dirpath, dirnames, filenames in os.walk(data_dir):
    for filename in filenames:
        full_path = os.path.join(dirpath, filename)

        # Step 1: Rename *.csv.csv files to *.csv
        if filename.endswith(".csv.csv"):
            corrected_name = filename.replace(".csv.csv", ".csv")
            corrected_path = os.path.join(dirpath, corrected_name)
            os.rename(full_path, corrected_path)
            print(f"Renamed file: {filename} -> {corrected_name}")
            full_path = corrected_path
            filename = corrected_name

        # Step 2: Process valid .csv files
        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(full_path)
                original_columns = df.columns.tolist()

                df.rename(columns=column_renames, inplace=True)

                if df.columns.tolist() != original_columns:
                    print(f"Fixed columns in: {full_path}")
                    df.to_csv(full_path, index=False)
                else:
                    print(f"No changes needed: {full_path}")

            except Exception as e:
                print(f"Failed to process {full_path}: {e}")