import json
import torch
from transformers import CLIPTokenizer, CLIPTextModel
import numpy as np

# Load your JSON file
with open("/home/dewei/workspace/SmellNet/data/text_description.json", "r") as f:
    descriptions = json.load(f)

# Initialize CLIP tokenizer and model
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Encode all descriptions into embeddings
text_embeddings = {}

for name, description in descriptions.items():
    inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token embedding (first token)
        embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, hidden_dim)
        text_embeddings[name] = embedding.squeeze().cpu().numpy()  # save as NumPy array

np.save("clip_text_embeddings.npy", text_embeddings)