from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    document_name: str
    page_number: int
    section: str = "General"
    score: float = 0.0


class DocumentRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        raise NotImplementedError
