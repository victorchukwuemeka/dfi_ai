from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")
w = ["king","queen","apple","watermelon", "man","woman","goat","TV","cat","teacher","student","pen","ink"]
embeddings = model.encode(w)
similarities = cosine_similarity(embeddings)

print(similarities.shape)

print("Cosine similarities:")
for i, word1 in enumerate(w):
    for j, word2 in enumerate(w):
        if i < j:
            print(f"{word1} vs {word2}: {similarities[i][j]:.3f}")
