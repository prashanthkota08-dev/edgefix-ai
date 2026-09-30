"""Optional semantic retrieval; transparently falls back to local TF-IDF."""
from __future__ import annotations
import hashlib
import json
import os
from .retriever import LocalDocumentRetriever
from .models import DocumentChunk

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticRetriever(LocalDocumentRetriever):
    def __init__(self, settings, model_id=MODEL_ID, model_loader=None):
        super().__init__(settings)
        self.model_id, self._model_loader = model_id, model_loader
        self.active_backend = "Semantic"; self.fallback_reason = None; self._model = None; self._embeddings = None

    def _load_model(self):
        if self._model is None:
            if self._model_loader:
                self._model = self._model_loader()
            else:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_id, device="cpu",
                    local_files_only=os.environ.get("EDGEFIX_OFFLINE", "0").lower() in {"1", "true", "yes"})
        return self._model

    def _document_embeddings(self, model):
        fingerprint = hashlib.sha256((self.model_id + "\n" + "\n".join(c.text for c in self.chunks)).encode()).hexdigest()
        path = self.settings.root / "data" / "processed" / "semantic_embeddings.json"
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint:
                return cached["embeddings"]
        except (OSError, ValueError, TypeError):
            pass
        vectors = model.encode([c.text for c in self.chunks], normalize_embeddings=True,
                               show_progress_bar=False).tolist()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fingerprint": fingerprint, "embeddings": vectors}), encoding="utf-8")
        return vectors

    def retrieve(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        if not query.strip() or not self.chunks:
            return []
        try:
            model = self._load_model()
            embeddings = self._document_embeddings(model)
            q = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
            scored = []
            for chunk, vector in zip(self.chunks, embeddings):
                score = sum(float(a) * float(b) for a, b in zip(q, vector))
                scored.append(DocumentChunk(**{**chunk.__dict__, "score": round(score, 4)}))
            self.active_backend = "Semantic"; self.fallback_reason = None
            return sorted(scored, key=lambda c: c.score, reverse=True)[:top_k]
        except Exception as exc:
            self.active_backend = "TF-IDF fallback"
            self.fallback_reason = f"Semantic model unavailable: {exc}"
            return super().retrieve(query, top_k)
