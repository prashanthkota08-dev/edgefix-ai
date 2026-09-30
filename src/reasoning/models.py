from dataclasses import dataclass, field
from ..rag.models import DocumentChunk


@dataclass
class Answer:
    asked: str
    found: str
    recommended_checks: list[str] = field(default_factory=list)
    safety: str = "Follow site-specific safety procedures and lockout/tagout requirements before physical inspection."
    sources: list[DocumentChunk] = field(default_factory=list)
    uncertainty: str = "Recommendations are limited to the cited local evidence."
    mode: str = "LOCAL RULE-BASED"
    observations: list[dict] = field(default_factory=list)
    evidence_strength: str = "insufficient"
