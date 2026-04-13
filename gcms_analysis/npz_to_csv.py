from pathlib import Path

import numpy as np
import pandas as pd

# ---------- mapping from pretty names to your internal snake_case ----------
pretty_to_internal = {
    "Peanut": "peanuts",
    "Cashew nut": "cashew",
    "Chestnut": "chestnuts",
    "Pistachio": "pistachios",
    "Almond": "almond",
    "Hazelnut": "hazelnut",
    "Common walnut": "walnuts",
    "Pecan nut": "pecans",
    "Brazil nut": "brazil_nut",
    "Pili nut": "pili_nut",
    "Cumin": "cumin",
    "Star anise": "star_anise",
    "Nutmeg": "nutmeg",
    "Cloves": "cloves",
    "Ginger": "ginger",
    "Allspice": "allspice",
    "Chervil": "chervil",
    "White mustard": "mustard",
    "Cinnamon": "cinnamon",
    "Saffron": "saffron",
    "Angelica": "angelica",
    "Garlic": "garlic",
    "Chives": "chives",
    "Turnip": "turnip",
    "Dill": "dill",
    "Mugwort": "mugwort",
    "Roman camomile": "chamomile",
    "Coriander": "coriander",
    "Mexican oregano": "oregano",
    "Spearmint": "mint",
    "Kiwi": "kiwi",
    "Pineapple": "pineapple",
    "Banana": "banana",
    "Lemon": "lemon",
    "Mandarin orange (Clementine, Tangerine)": "mandarin_orange",
    "Strawberry": "strawberry",
    "Apple": "apple",
    "Mango": "mango",
    "Peach": "peach",
    "Pear": "pear",
    "Cauliflower": "cauliflower",
    "Brussel sprouts": "brussel_sprouts",
    "Broccoli": "broccoli",
    "Sweet potato": "sweet_potato",
    "Asparagus": "asparagus",
    "Avocado": "avocado",
    "Radish": "radish",
    "Garden tomato": "tomato",
    "Potato": "potato",
    "Common cabbage": "cabbage",
}

# ---------- paths (repo-root gcms_processed/) ----------
_REPO_ROOT = Path(__file__).resolve().parent.parent
GCMS_PROCESSED = _REPO_ROOT / "gcms_processed"
npz_path = GCMS_PROCESSED / "gcms_food_vectors.npz"
out_csv = GCMS_PROCESSED / "gcms_food_vectors.csv"


# ---------- load NPZ ----------
data = np.load(npz_path, allow_pickle=True)
print("NPZ keys:", data.files)

X = data["vectors"]              # (N, D)
food_labels = data["food_labels"]  # (N,)

N, D = X.shape
print("Loaded", N, "foods with vector dim", D)


# ---------- map names to your internal IDs ----------
internal_names = []
for name in food_labels:
    name_str = str(name)
    internal = pretty_to_internal.get(name_str)
    if internal is None:
        # simple fallback so it doesn't crash if something unexpected appears
        internal = (
            name_str.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
        )
        print(f"[WARN] No mapping for '{name_str}', using '{internal}'")
    internal_names.append(internal)


# ---------- build very simple column names: f0, f1, ..., f{D-1} ----------
col_names = [f"f{i}" for i in range(D)]

# ---------- make DataFrame and save ----------
df = pd.DataFrame(X, columns=col_names)
df.insert(0, "food_name", internal_names)

df.to_csv(out_csv, index=False)
print("Wrote CSV to", out_csv)
