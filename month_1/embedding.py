from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")  

words = ["king", "queen", "man", "woman", "apple"]
embeddings = model.encode(words)

similarities = cosine_similarity(embeddings)

print("Cosine similarities:")
for i, word1 in enumerate(words):
    for j, word2 in enumerate(words):
        if i < j:
            print(f"{word1} vs {word2}: {similarities[i][j]:.3f}")



