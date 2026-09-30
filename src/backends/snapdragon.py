from .base import InferenceBackend


class SnapdragonBackend(InferenceBackend):
    """Target integration placeholder. No Snapdragon execution is implemented."""
    name = "Snapdragon (target integration)"
    def status(self):
        return {"name": self.name, "available": False,
                "detail": "Target integration only; no Snapdragon hardware/runtime execution is implemented."}
