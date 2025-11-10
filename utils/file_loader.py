from pypdf import PdfReader
from docx import Document
import pandas as pd

def read_file(file):
    """Reads content from PDF, TXT, DOCX, or CSV."""
    ext = file.name.split(".")[-1].lower()
    text = ""
    if ext == "txt":
        text = file.read().decode("utf-8", errors="ignore")
    elif ext == "pdf":
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif ext == "docx":
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif ext == "csv":
        df = pd.read_csv(file)
        text = df.to_string()
    return text
