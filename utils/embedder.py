from sentence_transformers import SentenceTransformer
import os

def get_embedder():
    """
    Loads a lightweight sentence-transformer model safely for CPU-only environments
    (Streamlit Cloud compatible).
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Use a lighter, CPU-friendly model
    model_name = "sentence-transformers/all-MiniLM-L12-v2"

    # Explicitly avoid device placement to prevent NotImplementedError
    model = SentenceTransformer(model_name)
    
    return model


