import faiss
import numpy as np

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def search_index(index, query_vector, top_k=3):
    return index.search(np.array(query_vector), top_k)
