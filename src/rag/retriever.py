from collections import Counter
from pathlib import Path
import json
import math
import re
from .ingest import chunk_pages, extract_pdf_pages
from .models import DocumentChunk, DocumentRetriever

TOKEN = re.compile(r"[a-z0-9][a-z0-9'-]*", re.I)


class LocalDocumentRetriever(DocumentRetriever):
    """Transparent, offline TF-IDF retrieval with a durable JSON index."""
    def __init__(self, settings):
        self.settings = settings
        self.index_path = settings.root / "data" / "processed" / "chunks.json"
        self.chunks = self._load()

    def _load(self):
        try:
            records = json.loads(self.index_path.read_text(encoding="utf-8"))
            return [DocumentChunk(**record) for record in records]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return []

    def ingest_directory(self, directory: Path | None = None):
        directory = directory or self.settings.root / "data" / "manuals"
        chunks = []
        for path in sorted(directory.glob("*.pdf")):
            chunks.extend(chunk_pages(extract_pdf_pages(path), path.stem.replace("_", " "),
                                      self.settings.chunk_chars, self.settings.chunk_overlap))
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps([c.__dict__ for c in chunks], ensure_ascii=False, indent=2), encoding="utf-8")
        self.chunks = chunks
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        terms = TOKEN.findall(query.lower())
        if not terms or not self.chunks:
            return []
        docs = [Counter(TOKEN.findall(c.text.lower())) for c in self.chunks]
        df = Counter(term for term in set(terms) for doc in docs if doc[term])
        scored = []
        for chunk, counts in zip(self.chunks, docs):
            score = sum((1 + math.log(counts[t])) * math.log(1 + len(docs) / (1 + df[t]))
                        for t in set(terms) if counts[t])
            if score:
                scored.append(DocumentChunk(**{**chunk.__dict__, "score": round(score, 4)}))
        return sorted(scored, key=lambda c: c.score, reverse=True)[:top_k]
