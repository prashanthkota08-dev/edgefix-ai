from pathlib import Path
from io import BytesIO
import json
import sys
import pytest

from src.config.settings import Settings
from src.rag.models import DocumentChunk
from src.rag.semantic import SemanticRetriever
from src.vision.clip_engine import ClipVisionEngine, ALL_LABELS, rank_concepts, decode_image, preprocess_image, prepare_cpu_inputs, clip_environment_status
from src.multimodal.evidence_fusion import EvidenceFusion
from src.reasoning.engine import RuleBasedFallbackReasoningEngine


def test_clip_initialization_is_lazy_and_cpu_engine():
    engine=ClipVisionEngine(local_files_only=True)
    assert engine.model_id=="openai/clip-vit-base-patch32"
    assert engine._load_error is None

def test_diagnostic_reports_cached_pytorch_checkpoint(tmp_path,monkeypatch):
    import huggingface_hub
    cached=tmp_path/"hub"/"pytorch_model.bin";cached.parent.mkdir(parents=True);cached.write_bytes(b"weights")
    monkeypatch.setattr(huggingface_hub,"try_to_load_from_cache",lambda model,filename,cache_dir=None:str(cached) if filename=="pytorch_model.bin" else None)
    result=clip_environment_status(tmp_path)
    assert result["device"]=="CPU" and result["model"]=="openai/clip-vit-base-patch32" and result["weights_cached"]

def test_cpu_inference_engine_selects_cpu_and_returns_image_and_text_signals(tmp_path,monkeypatch):
    monkeypatch.setenv("HF_HOME",str(Path(__file__).resolve().parents[1]/"models"/"huggingface"))
    engine=ClipVisionEngine(local_files_only=True)
    from PIL import Image
    stream=BytesIO(); Image.new("RGB",(16,12),color=(80,100,120)).save(stream,format="PNG")
    result=engine.analyze(stream.getvalue())
    if result.status=="Vision model unavailable": pytest.skip(f"CLIP stack/weights are not installed locally: {result.summary}")
    assert result.details["device"]=="CPU"
    assert len(result.details["observations"])==5
    assert result.details["timings"]["inference_seconds"]>=0
    assert len(result.details["labels"])==len(ALL_LABELS)

def test_clip_ranks_labels_by_similarity():
    scores=list(range(len(ALL_LABELS)))
    ranked=rank_concepts(scores)
    assert ranked[0]["label"]==ALL_LABELS[-1]
    assert ranked[-1]["label"]==ALL_LABELS[0]
    assert ranked[0]["similarity"]>ranked[-1]["similarity"]

def test_clip_cpu_input_selection_and_image_preprocessing():
    from PIL import Image
    calls=[]
    class Tensor:
        def to(self,device): calls.append(device); return self
    image=Image.new("RGB",(12,9)); prompts,inputs=preprocess_image(image,lambda **kw: {"pixel_values":Tensor(),"input_ids":Tensor()})
    ready=prepare_cpu_inputs(inputs)
    assert image.mode=="RGB" and len(prompts)==len(ALL_LABELS) and calls==["cpu","cpu"]
    assert set(ready)=={"pixel_values","input_ids"}
    stream=BytesIO(); image.save(stream,format="PNG")
    assert decode_image(stream.getvalue()).size==(12,9)

def test_clip_missing_model_graceful_fallback(monkeypatch):
    import src.vision.clip_engine as module
    monkeypatch.setattr(module,"_load",lambda *a,**k: (_ for _ in ()).throw(ImportError("missing model")))
    from PIL import Image
    image=Image.new("RGB",(8,8)); stream=BytesIO(); image.save(stream,format="PNG")
    result=ClipVisionEngine().analyze(stream.getvalue())
    assert result.status=="Vision model unavailable" and result.details["observations"]==[]

def test_semantic_retriever_falls_back_to_tfidf(tmp_path):
    settings=Settings(root=tmp_path)
    retriever=SemanticRetriever(settings,model_loader=lambda: (_ for _ in ()).throw(RuntimeError("offline model absent")))
    retriever.chunks=[DocumentChunk("Inspect blocked cooling fins and verify fan airflow.","Manual",3,"Cooling")]
    results=retriever.retrieve("inspect cooling airflow")
    assert results and results[0].page_number==3
    assert retriever.active_backend=="TF-IDF fallback" and "offline model absent" in retriever.fallback_reason

def test_evidence_fusion_and_citation_preservation():
    source=DocumentChunk("Inspect cooling passages after isolation.","Motor Manual",7,"Cooling",1.2)
    vision=type("Result",(),{"details":{"observations":[{"label":"industrial motor","similarity":0.31}]}})()
    context=EvidenceFusion().combine(vision,"motor overheating",[source])
    answer=RuleBasedFallbackReasoningEngine().answer(context)
    assert context.question=="motor overheating" and context.evidence_strength=="stronger combined signal"
    assert answer.sources[0] is source and answer.observations[0]["label"]=="industrial motor"
    assert answer.safety and "lockout/tagout" in answer.safety

def test_grounding_insufficient_input_exact_text():
    answer=RuleBasedFallbackReasoningEngine().answer(EvidenceFusion().combine(None,"unknown",[]))
    assert answer.found=="Insufficient evidence in the local knowledge base or visual input."

def test_ai_hub_token_missing_is_not_success(monkeypatch,capsys):
    import snapdragon.qai_hub_profile as profile
    monkeypatch.delenv("QAI_HUB_API_TOKEN",raising=False)
    monkeypatch.setattr(sys,"argv",["qai_hub_profile.py"])
    assert profile.main()==2
    assert "Qualcomm AI Hub validation is optional and has not been configured." in capsys.readouterr().err

def test_benchmark_writes_json_and_csv(tmp_path):
    from scripts.benchmark_vision import save_result
    output=tmp_path/"bench.json"
    result={"benchmark_type":"LOCAL INTEL CPU BENCHMARK","inference_seconds":0.25}
    csv_path=save_result(result,output)
    assert json.loads(output.read_text())["benchmark_type"]=="LOCAL INTEL CPU BENCHMARK"
    assert "inference_seconds" in csv_path.read_text()
