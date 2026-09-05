import os
from typing import List, Optional
from PyPDF2 import PdfReader

class DocumentParser:
    """Base document parser interface."""
    def parse(self, filepath: str) -> str:
        raise NotImplementedError

class TextParser(DocumentParser):
    def parse(self, filepath: str) -> str:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

class PDFParser(DocumentParser):
    def parse(self, filepath: str) -> str:
        text = ""
        try:
            reader = PdfReader(filepath)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            print(f"Error parsing PDF {filepath}: {e}")
        return text

def parse_document(filepath: str) -> str:
    """Helper to route to correct parser based on extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.txt' or ext == '.md':
        return TextParser().parse(filepath)
    elif ext == '.pdf':
        return PDFParser().parse(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
