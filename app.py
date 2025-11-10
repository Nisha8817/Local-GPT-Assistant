import streamlit as st
import numpy as np
from transformers import pipeline
from utils.file_loader import read_file
from utils.chunker import chunk_text
from utils.embedder import get_embedder
from utils.retriever import create_faiss_index, search_index

# -----------------------
# 🔧 Page setup
# -----------------------
st.set_page_config(page_title="Local GPT Assistant (Free RAG)", layout="centered")
st.title("📚 Local GPT Document Assistant (Free Model)")
st.write("Upload documents and ask questions from them — no API key, no internet, 100% local!")

# -----------------------
# 🧠 Initialize session state
# -----------------------
if "texts" not in st.session_state:
    st.session_state.texts = []
if "index" not in st.session_state:
    st.session_state.index = None

embedder = get_embedder()

# -----------------------
# 📂 Upload files
# -----------------------
files = st.file_uploader(
    "Upload PDF, TXT, DOCX, or CSV files",
    type=["pdf", "txt", "docx", "csv"],
    accept_multiple_files=True
)

if files:
    all_chunks = []
    for f in files:
        text = read_file(f)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    st.session_state.texts = all_chunks
    st.success("✅ Files processed successfully!")

    with st.spinner("Creating embeddings..."):
        embeddings = embedder.encode(all_chunks)
        st.session_state.index = create_faiss_index(embeddings)
    st.success("📦 Vector database created!")

# -----------------------
# 💬 Question answering
# -----------------------
query = st.text_input("Ask a question from your documents")

if query and st.session_state.index:
    q_vec = embedder.encode([query])
    D, I = search_index(st.session_state.index, q_vec, top_k=3)

    if len(I[0]) == 0 or D[0][0] > 1.5:
        st.warning("I don’t have enough information in the uploaded documents.")
    else:
        context = "\n\n".join([st.session_state.texts[i] for i in I[0]])
        st.info("📄 Context retrieved from your files.")

        # ✅ Try loading models safely with fallback
        try:
            llm = pipeline("text2text-generation", model="google/flan-t5-base")
            model_used = "google/flan-t5-base"
        except Exception as e:
            st.warning(f"⚠️ Flan-T5 not available, switching to smaller model (DistilGPT2): {e}")
            llm = pipeline("text-generation", model="distilgpt2")
            model_used = "distilgpt2"

        # ✅ Initialize answer before use
        answer = ""

        # ✂️ Truncate context to prevent model overflow
        if len(context) > 1500:
            context = context[:1500]

        with st.spinner(f"Generating answer using {model_used}..."):
            prompt = (
                f"Answer the following question only using the given context.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\nAnswer:"
            )

            try:
                output = llm(prompt, max_new_tokens=200, temperature=0.3)
                if output and len(output) > 0 and "generated_text" in output[0]:
                    answer = output[0]["generated_text"].strip()
                else:
                    answer = "⚠️ Model did not return any text."
            except Exception as e:
                st.error(f"⚠️ Model generation error: {e}")
                answer = "❌ Unable to generate a valid answer."

        # ✅ Display output neatly
        st.divider()
        st.info(f"🧩 Model used: {model_used}")
        st.subheader("🧠 Answer")
        if answer.strip():
            st.code(answer, language="markdown")
        else:
            st.warning("No response generated. Try asking a simpler or shorter question.")
else:
    st.caption("👆 Upload files and ask your question.")
