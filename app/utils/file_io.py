import pdfplumber
from pathlib import Path
from docx import Document
import uuid
import os

def extract_text(file: UploadFile):
    ext = Path(file.filename).suffix.lower()
    content = file.file.read()
    if ext == ".txt":
        return content.decode("utf-8")
    elif ext == ".pdf":
        temp_path = f"/tmp/temp_extract_{uuid.uuid4()}.pdf"
        with open(temp_path, "wb") as f:
            f.write(content)
        with pdfplumber.open(temp_path) as pdf:
            text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
        os.remove(temp_path)
        return text
    elif ext in [".docx", ".doc"]:
        temp_path = f"/tmp/temp_extract_{uuid.uuid4()}.docx"
        with open(temp_path, "wb") as f:
            f.write(content)
        doc = Document(temp_path)
        text = "\n\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
        os.remove(temp_path)
        return text
    else:
        raise ValueError("Unsupported file type")
