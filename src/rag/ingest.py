from pathlib import Path
import re
from .models import DocumentChunk


def extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF extraction requires PyMuPDF. Install requirements.txt.") from exc
    pages = []
    with fitz.open(path) as pdf:
        for number, page in enumerate(pdf, 1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append((number, text))
    return pages


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def chunk_pages(pages, document_name: str, chunk_chars: int = 1000,
                overlap: int = 160) -> list[DocumentChunk]:
    if chunk_chars < 100 or overlap < 0 or overlap >= chunk_chars:
        raise ValueError("chunk_chars must be >=100; overlap must be >=0 and less than chunk_chars")
    result = []
    for page, text in pages:
        text = clean_text(text)
        offset = 0
        while offset < len(text):
            end = min(offset + chunk_chars, len(text))
            if end < len(text):
                boundary = text.rfind(" ", offset + chunk_chars // 2, end)
                if boundary > offset:
                    end = boundary
            body = text[offset:end].strip()
            if body:
                heading = next((line.strip(" #:") for line in text.splitlines()
                                if line.strip() and len(line.strip()) < 100
                                and not line.strip().lower().startswith("edgefix ai")), "General")
                result.append(DocumentChunk(body, document_name, int(page), heading or "General"))
            if end >= len(text):
                break
            offset = max(offset + 1, end - overlap)
    return result
