"""Exercise CLIP -> local retrieval -> evidence fusion -> grounded reasoning."""
from pathlib import Path
import json, sys, datetime, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.config.settings import Settings
from src.rag.retriever import LocalDocumentRetriever
from src.vision.clip_engine import ClipVisionEngine
from src.multimodal.evidence_fusion import EvidenceFusion
from src.reasoning.engine import RuleBasedFallbackReasoningEngine

def main():
    import os
    os.environ.setdefault("HF_HOME",str(ROOT/"models"/"huggingface"))
    image=ROOT/"data"/"sample_images"/"sample_motor.png"
    if not image.is_file(): raise SystemExit(f"Missing smoke image: {image}")
    question="What should I inspect if this motor is overheating?"
    vision=ClipVisionEngine(local_files_only=True).analyze(image.read_bytes())
    if vision.status!="Visual similarity signal": raise SystemExit(f"CLIP unavailable: {vision.summary}")
    retriever=LocalDocumentRetriever(Settings.from_env()); sources=retriever.retrieve(question,5)
    if not sources: raise SystemExit("No local manual evidence was retrieved; run scripts/ingest_documents.py")
    context=EvidenceFusion().combine(vision,question,sources)
    answer=RuleBasedFallbackReasoningEngine().answer(context)
    result={"pipeline":"IMAGE → CLIP → VISUAL OBSERVATIONS → QUESTION → LOCAL RAG → EVIDENCE FUSION → GROUNDED ANSWER","completed_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"device":"CPU","gpu":"NOT USED","model":vision.details["model"],"image_path":str(image.relative_to(ROOT)),"question":question,"OBSERVATION":{"visual_signal":vision.summary,"top_concepts":vision.details["observations"]},"EVIDENCE":{"strength":context.evidence_strength,"retrieved_passages":[{"document":s.document_name,"page":s.page_number,"section":s.section,"score":s.score,"excerpt":s.text} for s in answer.sources]},"RECOMMENDED CHECKS":answer.recommended_checks,"SAFETY":answer.safety,"UNCERTAINTY":answer.uncertainty,"SOURCES":[{"document":s.document_name,"page":s.page_number,"section":s.section} for s in answer.sources],"grounding_note":"CLIP similarity does not diagnose overheating; checks below are extracted from the cited local documentation."}
    output=ROOT/"artifacts"/"end_to_end_multimodal_result.json";output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2)); print(f"Saved {output}")

if __name__=="__main__": main()
