from .base import InferenceBackend


class CPUBackend(InferenceBackend):
    name = "CPU"
    def status(self):
        return {"name": self.name, "available": True, "detail": "Local CPU inference and retrieval"}
