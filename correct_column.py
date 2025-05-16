import os
import pandas as pd

root_dir = "/Users/derre/Documents/workspace/smell-net/testing"

# Walk through the root directory
for root, dirs, files in os.walk(root_dir):
    for filename in files:
        if filename.endswith(".csv"):
            file_path = os.path.join(root, filename)
            df = pd.read_csv(file_path)

            # Rename column if necessary
            if "C2H50H" in df.columns:
                df = df.rename(columns={"C2H50H": "C2H5OH"})
                
                # Save back to the same file (overwrite)
                df.to_csv(file_path, index=False)
                print(f"Fixed and saved: {file_path}")
