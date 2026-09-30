from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class EvidenceContext:
    visual_observations: list[dict[str, Any]]
    question: str
    retrieved_sources: list[Any]
    evidence_strength: str
    uncertainties: list[str]

    def to_dict(self):
        return {**asdict(self), "retrieved_sources": [asdict(x) if hasattr(x, "__dataclass_fields__") else x for x in self.retrieved_sources]}


class EvidenceFusion:
    def combine(self, visual_result, question: str, sources) -> EvidenceContext:
        details = getattr(visual_result, "details", {}) or {}
        observations = details.get("observations", [])
        if isinstance(observations, list):
            observations = [x if isinstance(x, dict) else {"label": str(x)} for x in observations]
        uncertainties = ["Visual similarity is not a calibrated probability or confirmed diagnosis."] if observations else ["No visual model observation is available."]
        if not sources:
            uncertainties.append("No matching local documentation was retrieved.")
        if not observations:
            uncertainties.append("The image provides no model-supported visual observation.")
        strength = "stronger combined signal" if observations and sources else "documentation only" if sources else "insufficient"
        return EvidenceContext(observations, question, list(sources), strength, uncertainties)
