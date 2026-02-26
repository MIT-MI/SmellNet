import pandas as pd
import plotly.express as px
from collections import defaultdict

ingredient_to_category = {
    # Nuts
    "peanuts": "Nuts",
    "cashew": "Nuts",
    "chestnuts": "Nuts",
    "pistachios": "Nuts",
    "almond": "Nuts",
    "hazelnut": "Nuts",
    "walnuts": "Nuts",
    "pecans": "Nuts",
    "brazil_nut": "Nuts",
    "pili_nut": "Nuts",
    
    # Spices
    "cumin": "Spices",
    "star_anise": "Spices",
    "nutmeg": "Spices",
    "cloves": "Spices",
    "ginger": "Spices",
    "allspice": "Spices",
    "chervil": "Spices",
    "mustard": "Spices",
    "cinnamon": "Spices",
    "saffron": "Spices",
    
    # Herbs
    "angelica": "Herbs",
    "garlic": "Herbs",
    "chives": "Herbs",
    "turnip": "Herbs",
    "dill": "Herbs",
    "mugwort": "Herbs",
    "chamomile": "Herbs",
    "coriander": "Herbs",
    "oregano": "Herbs",
    "mint": "Herbs",
    
    # Fruits
    "kiwi": "Fruits",
    "pineapple": "Fruits",
    "banana": "Fruits",
    "lemon": "Fruits",
    "orange": "Fruits",
    "strawberry": "Fruits",
    "apple": "Fruits",
    "mango": "Fruits",
    "peach": "Fruits",
    "pear": "Fruits",
    
    # Vegetables
    "cauliflower": "Vegetables",
    "brussel_sprouts": "Vegetables",
    "broccoli": "Vegetables",
    "sweet_potato": "Vegetables",
    "asparagus": "Vegetables",
    "avocado": "Vegetables",
    "radish": "Vegetables",
    "tomato": "Vegetables",
    "potato": "Vegetables",
    "cabbage": "Vegetables",
}

# Format ingredient names nicely
ingredient_categories = defaultdict(list)
for food, category in ingredient_to_category.items():
    formatted_food = food.replace("_", " ").title()
    ingredient_categories[category].append(formatted_food)

def plot_smellnet_sunburst():
    # Prepare DataFrame
    rows = []
    for category, ingredients in ingredient_categories.items():
        for ingredient in ingredients:
            rows.append({
                "Format": "Sensor, GC-MS, Text",
                "Category": category,
                "Ingredient": ingredient,
                "Samples": 100  # Can keep this for equal segment sizes
            })

    df = pd.DataFrame(rows)

    # Plot with prettier colors
    fig = px.sunburst(
        df,
        path=["Format", "Category", "Ingredient"],
        values="Samples",
        color="Category",
        color_discrete_sequence=px.colors.qualitative.Set2,  # More vibrant color set
        height=850
    )

    # Remove percentage for cleaner display
    fig.update_traces(textinfo='label')

    # Aesthetic improvements
    fig.update_layout(
        margin=dict(t=50, l=0, r=0, b=0),
        uniformtext=dict(minsize=10, mode='show'),
        font=dict(size=14),
    )
    fig.update_layout(
        margin=dict(t=50, l=0, r=0, b=0),
        uniformtext=dict(minsize=35, mode='show'),  # larger label text
        font=dict(size=25),  # larger title/legend text
        width=1100,
        height=1100
    )

    # Then save the image
    fig.write_image("smellnet_sunburst.png", scale=2)  # square high-res PNG

if __name__ == "__main__":
    plot_smellnet_sunburst()