from pathlib import Path
import json
import pytest
from src.config.settings import Settings
from src.rag.ingest import chunk_pages, extract_pdf_pages
from src.rag.retriever import LocalDocumentRetriever
from src.reasoning.engine import RuleBasedFallbackReasoningEngine, OllamaReasoningEngine
from src.rag.models import DocumentChunk
from src.backends.cpu import CPUBackend
from src.backends.snapdragon import SnapdragonBackend
from src.vision.engine import BasicVisionEngine


def settings(tmp_path): return Settings(root=tmp_path)

def test_chunking_metadata_and_retrieval(tmp_path):
    chunks=chunk_pages([(4,"Cooling passages: inspect blocked fins and verify fan condition. "*15)],"sample",180,30)
    assert chunks and all(c.document_name=="sample" and c.page_number==4 for c in chunks)
    r=LocalDocumentRetriever(settings(tmp_path)); r.chunks=chunks
    result=r.retrieve("inspect blocked cooling fins")
    assert result and result[0].page_number==4 and result[0].score>0

def test_pdf_extraction(tmp_path):
    fitz=pytest.importorskip("fitz")
    path=tmp_path/"test.pdf"; doc=fitz.open(); p=doc.new_page(); p.insert_text((50,50),"Bearing inspection guidance"); doc.save(path); doc.close()
    assert "Bearing" in extract_pdf_pages(path)[0][1]

def test_backend_selection_and_target_disclosure():
    assert CPUBackend().status()["available"]
    assert not SnapdragonBackend().status()["available"]
    assert "placeholder" in SnapdragonBackend.__doc__.lower()

def test_empty_knowledge_base_and_fallback_safety():
    answer=RuleBasedFallbackReasoningEngine().answer("unknown",[])
    assert answer.found=="Insufficient evidence in the local knowledge base or visual input."
    assert "lockout/tagout" in answer.safety.lower()

def test_fallback_uses_retrieved_passage():
    chunk=DocumentChunk("Inspect blocked cooling fins and verify airflow.","Manual",2,"Cooling",1.3)
    answer=RuleBasedFallbackReasoningEngine().answer("overheating",[chunk])
    assert answer.sources[0].page_number==2 and "Inspect blocked cooling fins" in answer.recommended_checks[0]

def test_invalid_image():
    result=BasicVisionEngine().analyze(b"not an image")
    assert result.status=="Invalid image"

def test_missing_ollama_falls_back(monkeypatch):
    monkeypatch.setattr(OllamaReasoningEngine,"_check",lambda self:False)
    engine=OllamaReasoningEngine("http://127.0.0.1:1","missing")
    chunk=DocumentChunk("Inspect cooling fins.","Manual",1)
    assert engine.answer("cooling",[chunk]).sources

def test_retriever_missing_index_and_empty_query(tmp_path):
    r=LocalDocumentRetriever(settings(tmp_path))
    assert r.retrieve("anything")==[]
    assert r.retrieve(" ")==[]
