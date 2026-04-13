from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Directory that contains all the ingredient folders (edit if your tree differs)
ROOT = REPO_ROOT / "data" / "offline_testing"

for csv_path in ROOT.rglob("*.csv"):
    # csv_path.name  -> e.g. "allspice.3262415c70ba.csv"
    # csv_path.stem  -> "allspice.3262415c70ba"
    ingredient = csv_path.stem.split(".")[0]   # "allspice"

    new_name = f"{ingredient}_6.csv"           # "allspice_6.csv"
    new_path = csv_path.with_name(new_name)

    print(f"Renaming {csv_path} -> {new_path}")
    csv_path.rename(new_path)