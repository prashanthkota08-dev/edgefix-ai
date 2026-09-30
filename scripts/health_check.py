from pathlib import Path
import platform, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.config.settings import Settings
from src.rag.retriever import LocalDocumentRetriever
from src.backends.cpu import CPUBackend
from src.vision.clip_engine import clip_environment_status
import importlib.util, shutil

def main():
    settings=Settings.from_env(); retriever=LocalDocumentRetriever(settings)
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"OS: {platform.platform()}")
    print(f"Backend: {CPUBackend().name}")
    print(f"Ollama endpoint: {settings.ollama_url} (optional; app falls back if unavailable)")
    print(f"Knowledge base: {len(retriever.chunks)} chunks")
    print(f"PDF manuals: {len(list((ROOT/'data'/'manuals').glob('*.pdf')))}")
    clip=clip_environment_status(ROOT)
    print(f"CLIP installed: {'YES' if clip['installed'] else 'NO'}")
    print(f"CLIP model: {clip['model']} | weights cached: {'YES' if clip['weights_cached'] else 'NO'} | device: {clip['device']}")
    print(f"Ollama installed: {'YES' if shutil.which('ollama') else 'NO'}")
    print(f"Semantic retrieval installed: {'YES' if importlib.util.find_spec('sentence_transformers') else 'NO (TF-IDF fallback)'}")
    print(f"Vector index: {'READY' if (ROOT/'data'/'processed'/'chunks.json').is_file() else 'NOT READY'}")
    print("Snapdragon target: NOT VALIDATED")
    print("Health check complete.")

if __name__=="__main__": main()
