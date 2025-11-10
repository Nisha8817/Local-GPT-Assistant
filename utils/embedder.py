from sentence_transformers import SentenceTransformer

def get_embedder():
    """Return a pre-loaded embedding model."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
