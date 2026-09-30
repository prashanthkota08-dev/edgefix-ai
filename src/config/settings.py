from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    root: Path
    chunk_chars: int = 1000
    chunk_overlap: int = 160
    top_k: int = 5
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    @classmethod
    def from_env(cls):
        root = Path(__file__).resolve().parents[2]
        return cls(root=root, chunk_chars=int(os.getenv("EDGEFIX_CHUNK_CHARS", "1000")),
                   chunk_overlap=int(os.getenv("EDGEFIX_CHUNK_OVERLAP", "160")),
                   top_k=int(os.getenv("EDGEFIX_TOP_K", "5")),
                   ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
                   ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
