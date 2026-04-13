#!/usr/bin/env python3
"""Create SmellNet-Base layout under base_data/: same structure as data/ but CSVs only have NO2,C2H5OH,VOC,CO,Alcohol,LPG."""

import csv
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "data"
DST = REPO_ROOT / "base_data"
KEEP_COLUMNS = ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG"]


def main():
    DST.mkdir(exist_ok=True)

    for path in sorted(SRC.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(SRC)
        out_path = DST / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() == ".csv":
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    shutil.copy2(path, out_path)
                    continue
                # Check if this is a sensor CSV (has our 6 columns)
                try:
                    indices = [header.index(c) for c in KEEP_COLUMNS]
                except ValueError:
                    # Different structure (e.g. gcms_dataframe.csv), copy as-is
                    shutil.copy2(path, out_path)
                    continue
                with open(out_path, "w", newline="", encoding="utf-8") as out:
                    writer = csv.writer(out)
                    writer.writerow(KEEP_COLUMNS)
                    for row in reader:
                        if len(row) > max(indices):
                            writer.writerow([row[i] for i in indices])
                        else:
                            writer.writerow([row[i] if i < len(row) else "" for i in indices])
        else:
            shutil.copy2(path, out_path)

    print(f"Created {DST}/ with filtered CSVs (columns: {', '.join(KEEP_COLUMNS)}) and copied other files.")


if __name__ == "__main__":
    main()
