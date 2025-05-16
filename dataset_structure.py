from graphviz import Digraph
import os

def build_dataset_tree(root_dir, label, color="lightblue", max_ingredients=None, compact=False):
    dot = Digraph(comment=label)
    dot.attr(rankdir='LR', bgcolor='white')
    dot.attr('node', shape='box', style='filled,rounded', fontname='Helvetica')

    # Root node
    dot.node(root_dir, label=label.upper(), fillcolor=color)

    ingredient_folders = sorted(os.listdir(root_dir))
    if max_ingredients:
        ingredient_folders = ingredient_folders[:max_ingredients]

    for ingredient in ingredient_folders:
        ing_path = os.path.join(root_dir, ingredient)
        if not os.path.isdir(ing_path):
            continue

        ing_node = f"{label}_{ingredient}"
        dot.node(ing_node, ingredient.title(), fillcolor="white")
        dot.edge(root_dir, ing_node)

        csv_count = len([f for f in os.listdir(ing_path) if f.endswith(".csv")])

        if compact:
            label_text = f"{csv_count} CSV file{'s' if csv_count > 1 else ''}"
            dot.node(f"{ing_node}_files", label_text, shape="note", fillcolor="#f3f3f3")
            dot.edge(ing_node, f"{ing_node}_files")
        else:
            for i, csv_file in enumerate(sorted(os.listdir(ing_path))):
                file_label = csv_file.replace(".csv", "")
                file_node = f"{ing_node}_{i}"
                dot.node(file_node, file_label, shape="note", fillcolor="#f3f3f3")
                dot.edge(ing_node, file_node)

    return dot


# === Create training (compact) and testing (full) trees ===
train_tree = build_dataset_tree("/home/dewei/workspace/smell-net/training", "Training Set", color="lightblue", max_ingredients=5, compact=True)
test_tree = build_dataset_tree("/home/dewei/workspace/smell-net/testing", "Testing Set", color="lightgreen", max_ingredients=5, compact=True)

# Combine both into a single figure
final_dot = Digraph(name="SmellNet Dataset Structure")
final_dot.subgraph(train_tree)
final_dot.subgraph(test_tree)
final_dot.attr(label="SmellNet Dataset Structure", fontsize="20", fontname="Helvetica", labelloc="top")

# Save
final_dot.render("smellnet_dataset_tree_compact", format="png", cleanup=True)
