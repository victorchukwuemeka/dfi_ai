def fixed_chunks(text, chunk_size=512, overlap=50):
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append({
            "text": text[start:end],
            "start_char": start,
            "end_char": end
        })
        start += chunk_size - overlap
    return chunks

# Pros: Simple, predictable chunk count
# Cons: May split in the middle of a sentence or thought