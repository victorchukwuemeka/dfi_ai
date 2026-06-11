import numpy as np
from sentence_transformers import CrossEncoder

rerank = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


# retrieve and rerank 
def re_rerank(query, vector_store,  embedder, top_k=20, rerank_top_n=3):
    
    results = vector_store.query(
        query_embeddings=embedder.encode([query]).tolist(),
        n_results=top_k
    )

    candidates = results["documents"][0]
    candidate_ids = results["ids"][0]
    candidate_metadatas = results["metadatas"][0]


    pairs = [(query,doc)for doc in candidates]

    scores = rerank.predict(pairs)

    ranked_indices = np.argsort(scores)[::-1]

    results = []

    for i in range(rerank_top_n):
        idx = ranked_indices[i]
        results.append({
            "rank": i + 1,
            "score": float(scores[idx]),
            "document": candidates[idx],
            "id": candidate_ids[idx],
            "metadata": candidate_metadatas[idx]
        })
    
    return results