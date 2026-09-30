from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.config.settings import Settings
from src.rag.retriever import LocalDocumentRetriever

if __name__=="__main__":
    try:
        retriever=LocalDocumentRetriever(Settings.from_env())
        count=retriever.ingest_directory()
        print(f"Indexed {count} chunks from {len({c.document_name for c in retriever.chunks})} PDFs.")
        if not count: print("No PDF manuals found in data/manuals.")
    except Exception as exc:
        print(f"Ingestion failed: {exc}",file=sys.stderr); raise SystemExit(1)
