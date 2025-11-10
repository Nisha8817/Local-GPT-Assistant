# 📚 Local GPT Assistant (RAG-based Mini Project)

### 🚀 Overview
A **local AI assistant** that lets users upload documents (TXT, PDF, DOCX, CSV), indexes them using embeddings (SentenceTransformers + FAISS), and answers context-based questions — 100% offline and OpenAI-free.

If no answer is found in the data, it responds:
> “I don’t have enough information in the uploaded documents.”

### 🧠 Features
- Upload multiple files (`.pdf`, `.txt`, `.csv`, `.docx`)
- Extracts and chunks text (~300–500 words each)
- Embeds chunks with SentenceTransformers (`all-MiniLM-L6-v2`)
- Stores and retrieves context using FAISS
- Generates answers locally using a HuggingFace model (`distilgpt2`)
- No hallucinations — answers strictly from user-provided documents
- Streamlit-based interactive interface

### 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Libraries:** Streamlit, FAISS, SentenceTransformers, Transformers, PyTorch, NumPy, Pandas, PyPDF, python-docx
- **Framework:** Streamlit UI
- **Storage:** Local FAISS vector index

### ⚙️ Setup Instructions

```bash
# 1️⃣ Clone the repo
git clone https://github.com/Nisha8817/Local-GPT-Assistant.git
cd Local-GPT-Assistant

# 2️⃣ Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Windows

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run Streamlit app
streamlit run app.py
