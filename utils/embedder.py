from sentence_transformers import SentenceTransformer
import torch

def get_embedder():
    # Force CPU for Streamlit Cloud to avoid NotImplementedError
    device = "cpu"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name, device=device)
    return model

