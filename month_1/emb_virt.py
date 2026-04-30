import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

words = ["king", "queen", "man", "woman", "apple"]
inputs = tokenizer(words, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

# Calculate similarities
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(embeddings.numpy())

print("Cosine similarities:")
for i, word1 in enumerate(words):
    for j, word2 in enumerate(words):
        if i < j:
            print(f"{word1} vs {word2}: {similarities[i][j]:.3f}")
