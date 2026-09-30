import platform
import hashlib
import json
import importlib.util
import shutil
import streamlit as st
from pathlib import Path
from src.rag.semantic import SemanticRetriever
from src.multimodal.evidence_fusion import EvidenceFusion
from src.vision.clip_engine import clip_environment_status


def style():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    .stApp{background:#0b1117;color:#e8eef1;font-family:Manrope,sans-serif}.block-container{max-width:1220px;padding-top:1.5rem}
    [data-testid="stSidebar"]{background:#101820;border-right:1px solid #26333b}.hero{padding:30px 34px;border:1px solid #273a3b;border-radius:18px;background:linear-gradient(120deg,#11231f,#121c25 65%,#17222b);margin:10px 0 24px}
    .eyebrow{font:500 11px 'DM Mono',monospace;letter-spacing:2px;color:#59d9a5;text-transform:uppercase}.hero h1{font-size:40px;margin:7px 0 2px;letter-spacing:-1.5px}.hero p{color:#a9b8c1;margin:0}
    .metric{background:#111a21;border:1px solid #26343c;border-radius:13px;padding:18px 19px;min-height:100px}.metric small{font:11px 'DM Mono',monospace;color:#8da0aa;letter-spacing:1px}.metric strong{display:block;font-size:22px;margin-top:8px}.section{font:500 12px 'DM Mono',monospace;color:#69dcae;letter-spacing:1.5px;margin:24px 0 10px}
    .source{border-left:2px solid #52c99b;padding:10px 14px;margin:8px 0;background:#111a21;border-radius:0 8px 8px 0}.source small{color:#96a7af}.stButton>button{border-radius:9px;font-weight:700}
    </style>""", unsafe_allow_html=True)


def render_app(settings, retriever, reasoner, vision, backend):
    st.set_page_config(page_title="EdgeFix AI", page_icon="⚙️", layout="wide")
    style()
    with st.sidebar:
        st.markdown("<div class='eyebrow'>FIELD SYSTEMS / 01</div><h2>⚙ EDGEFIX <span style='color:#59d9a5'>AI</span></h2><p style='color:#91a3ad'>See it. Ask it. Fix it — offline.</p>", unsafe_allow_html=True)
        page = st.radio("Navigation", ["Dashboard", "Inspection", "Ask EdgeFix", "Knowledge Base", "Diagnostics", "Settings"], label_visibility="collapsed")
        st.markdown("<div class='section'>RETRIEVAL BACKEND</div>", unsafe_allow_html=True)
        retrieval_choice = st.radio("Retrieval backend", ["Semantic", "TF-IDF fallback"], index=1, key="retrieval_backend")
        st.divider()
        st.caption("LOCAL-FIRST · CPU / INTEL")
        st.caption("Target deployment · Snapdragon X Series")
    if "semantic_retriever" not in st.session_state:
        st.session_state.semantic_retriever = SemanticRetriever(settings)
    active_retriever = st.session_state.semantic_retriever if retrieval_choice == "Semantic" else retriever
    count_docs = len({c.document_name for c in active_retriever.chunks})
    if page == "Dashboard":
        st.markdown("<div class='hero'><div class='eyebrow'>OFFLINE FIELD COPILOT</div><h1>EDGEFIX AI</h1><p>See it. Ask it. Fix it — offline.</p></div>", unsafe_allow_html=True)
        cols = st.columns(5)
        dashboard_metrics = [("SYSTEM", "READY" if active_retriever.chunks else "INDEX NEEDED"), ("AI MODE", "LOCAL"), ("NETWORK", "OFFLINE"), ("VISION", "CLIP — CPU VERIFIED" if clip_environment_status(settings.root)["weights_cached"] else "CLIP — NOT CACHED"), ("SNAPDRAGON TARGET", "NOT YET VALIDATED")]
        for col, (title, value) in zip(cols, dashboard_metrics):
            col.markdown(f"<div class='metric'><small>{title}</small><strong>{value}</strong></div>", unsafe_allow_html=True)
        st.markdown("<div class='section'>KNOWLEDGE BASE</div>", unsafe_allow_html=True)
        a,b,c=st.columns(3)
        for col,title,value in [(a,"DOCUMENTS",count_docs),(b,"INDEXED CHUNKS",len(retriever.chunks)),(c,"KNOWLEDGE BASE",f"{count_docs} documents / {len(retriever.chunks)} chunks")]:
            col.markdown(f"<div class='metric'><small>{title}</small><strong>{value}</strong></div>",unsafe_allow_html=True)
        st.markdown("<div class='section'>QUICK START</div>",unsafe_allow_html=True)
        st.markdown("<div class='section'>SHOW ME AND TELL ME</div>", unsafe_allow_html=True)
        st.markdown("**IMAGE + QUESTION → CLIP VISUAL SIGNAL → LOCAL MANUAL RETRIEVAL → EVIDENCE FUSION → GROUNDED RESPONSE**")
        st.info("Open the Show me and tell me workflow to add an equipment photo and ask a maintenance question.")
        render_local_benchmark(settings.root)
        render_snapdragon_target(settings.root)
    elif page in ("Inspection", "Ask EdgeFix"):
        st.markdown("<div class='hero'><div class='eyebrow'>SHOW ME AND TELL ME</div><h1>Show me and tell me</h1><p>Image → CLIP visual signals → local manual evidence → grounded response.</p></div>", unsafe_allow_html=True)
        st.markdown("**AI PROCESSING PIPELINE**  \nIMAGE　→　VISION　→　RETRIEVAL　→　REASONING　→　ANSWER")
        upload = st.file_uploader("Upload equipment image (optional)", type=["png","jpg","jpeg","webp","bmp"])
        vr = None
        if upload:
            left,right=st.columns([1,1])
            left.image(upload, caption=upload.name, use_container_width=True)
            digest = hashlib.sha256(upload.getvalue()).hexdigest()
            cached = st.session_state.get("vision_result_cache")
            if cached and cached[0] == digest:
                vr = cached[1]
            else:
                with st.spinner("Running CLIP visual similarity on CPU…"):
                    vr=vision.analyze(upload.getvalue())
                st.session_state.vision_result_cache = (digest, vr)
            with right:
                st.markdown(f"<div class='section'>{vr.status.upper()}</div>",unsafe_allow_html=True)
                st.write(vr.summary)
                ranked = vr.details.get("observations", []) if vr.details else []
                for item in ranked:
                    st.write(f"**{item['label']}** · CLIP similarity {item['similarity']:.2f}")
                if vr.status == "Vision model unavailable": st.caption("Metadata may be shown by image preview only; no AI visual analysis was performed.")
        with st.form("ask"):
            query=st.text_area("What would you like to inspect?", placeholder="What should I inspect if this motor is overheating?", height=90)
            submitted=st.form_submit_button("Search evidence and answer", type="primary", use_container_width=True)
        if submitted:
            if not query.strip(): st.error("Enter a question to continue.")
            else:
                with st.spinner(f"Retrieving with {retrieval_choice} backend…"):
                    evidence=active_retriever.retrieve(query, settings.top_k)
                fusion=EvidenceFusion().combine(vr, query.strip(), evidence) if vr else EvidenceFusion().combine(type("NoVision",(),{"details":{}})(), query.strip(), evidence)
                answer=reasoner.answer(fusion)
                pipeline = [("1 · IMAGE RECEIVED", "✓ complete" if upload else "⚠ image not provided"),
                            ("2 · VISION ANALYSIS", "✓ complete" if vr and vr.status == "Visual similarity signal" else "⚠ unavailable" if upload else "● not run"),
                            ("3 · DOCUMENT RETRIEVAL", "✓ complete" if evidence else "⚠ no matching evidence"),
                            ("4 · EVIDENCE FUSION", "✓ complete"),
                            ("5 · GROUNDED ANSWER", "✓ complete" if evidence else "⚠ insufficient evidence")]
                stage_cols=st.columns(5)
                for col,(label,state) in zip(stage_cols,pipeline):
                    col.markdown(f"<div class='metric' style='min-height:85px'><small>{label}</small><strong style='font-size:14px'>{state}</strong></div>",unsafe_allow_html=True)
                st.markdown("<div class='section'>VISUAL OBSERVATION</div>",unsafe_allow_html=True)
                if answer.observations:
                    for item in answer.observations:
                        st.write(f"{item.get('label','Visual concept')}　{item.get('similarity',0):.2f} relative similarity")
                else: st.caption(vr.status if vr else "No image supplied.")
                st.markdown("<div class='section'>MODEL-SUPPORTED VISUAL OBSERVATION</div>",unsafe_allow_html=True); st.write(answer.found)
                st.markdown("<div class='section'>DOCUMENT EVIDENCE</div>",unsafe_allow_html=True)
                st.write(f"Evidence strength: {fusion.evidence_strength} · Retrieval: {getattr(active_retriever,'active_backend','TF-IDF')}")
                if getattr(active_retriever,"fallback_reason",None): st.warning(active_retriever.fallback_reason)
                st.markdown("<div class='section'>RECOMMENDED CHECKS</div>",unsafe_allow_html=True)
                for check in answer.recommended_checks: st.markdown(f"- {check}")
                st.markdown("<div class='section'>SAFETY</div>",unsafe_allow_html=True); st.warning(answer.safety)
                st.markdown("<div class='section'>SOURCES</div>",unsafe_allow_html=True)
                if not answer.sources: st.caption("No sources returned.")
                for source in answer.sources:
                    st.markdown(f"<div class='source'><b>{source.document_name}</b><br><small>Page {source.page_number} · Section: {source.section} · Match {source.score:.2f}</small><br>{source.text}</div>",unsafe_allow_html=True)
                st.markdown("<div class='section'>UNCERTAINTY</div>",unsafe_allow_html=True); st.caption(answer.uncertainty)
                st.caption(f"Reasoning mode: {answer.mode}")
    elif page == "Knowledge Base":
        st.markdown("<div class='hero'><div class='eyebrow'>LOCAL LIBRARY</div><h1>Knowledge Base</h1><p>Original synthetic training material, indexed locally.</p></div>",unsafe_allow_html=True)
        st.metric("Documents",count_docs); st.metric("Indexed chunks",len(retriever.chunks))
        if st.button("Re-index data/manuals PDFs"):
            try:
                n=retriever.ingest_directory()
                st.session_state.semantic_retriever = SemanticRetriever(settings)
                st.success(f"Indexed {n} chunks."); st.rerun()
            except Exception as exc: st.error(f"Indexing failed: {exc}")
        st.caption(f"Selected: {retrieval_choice} · Active: {getattr(active_retriever,'active_backend', retrieval_choice)}")
        for name in sorted({c.document_name for c in active_retriever.chunks}): st.write("📄",name)
        if not active_retriever.chunks: st.info("No indexed documents yet. Run scripts/ingest_documents.py or use the button above.")
    elif page == "Diagnostics":
        st.markdown("<div class='hero'><div class='eyebrow'>SYSTEM TELEMETRY</div><h1>Diagnostics</h1><p>Current development mode: CPU / INTEL · Target deployment: SNAPDRAGON</p></div>",unsafe_allow_html=True)
        gpu="Not detected by lightweight system probe"
        try:
            import torch
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else gpu
        except ImportError: pass
        clip_status=clip_environment_status(settings.root)
        ollama_installed=shutil.which("ollama") is not None
        semantic_installed=importlib.util.find_spec("sentence_transformers") is not None
        values={"Python":platform.python_version(),"Operating system":platform.platform(),"CPU":platform.processor() or platform.machine(),"GPU":gpu,"CLIP installed":"YES" if clip_status["installed"] else "NO","CLIP model":clip_status["model"],"CLIP weights cached":"YES" if clip_status["weights_cached"] else "NO","CLIP device":"CPU","Ollama installed":"YES" if ollama_installed else "NO","Ollama model available": "YES" if getattr(reasoner,"available",False) else "NO — local fallback active","Semantic retrieval installed":"YES" if semantic_installed else "NO — TF-IDF fallback","Vector DB":"READY (local JSON + TF-IDF)" if (settings.root/"data"/"processed"/"chunks.json").is_file() else "NOT READY","Knowledge base":"READY" if active_retriever.chunks else "EMPTY — run ingestion","Active retrieval":getattr(active_retriever,'active_backend',retrieval_choice),"Active inference backend":backend.name,"Snapdragon target status":"NOT VALIDATED"}
        for k,v in values.items(): st.markdown(f"<div class='metric' style='margin:8px 0;min-height:0'><small>{k.upper()}</small><strong style='font-size:16px'>{v}</strong></div>",unsafe_allow_html=True)
        render_snapdragon_target(settings.root)
    elif page == "Settings":
        st.markdown("<div class='hero'><div class='eyebrow'>PREFERENCES</div><h1>Settings</h1><p>Local inference configuration.</p></div>",unsafe_allow_html=True)
        st.write("Ollama endpoint:",settings.ollama_url); st.write("Model:",settings.ollama_model); st.write("Retrieval results:",settings.top_k)
        st.info("Ollama is optional. When unavailable, EdgeFix uses deterministic evidence extraction. No cloud API key is used.")


def render_snapdragon_target(root: Path):
    st.markdown("<div class='section'>SNAPDRAGON OPTIMIZATION PATH</div>",unsafe_allow_html=True)
    path=root/"snapdragon"/"ai_hub_profile_result.json"
    result=None
    try: result=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,ValueError): pass
    status=result.get("status") if result else "Target profile not yet validated"
    cols=st.columns(5)
    entries=[("Local development","Intel CPU"),("Target","Snapdragon X Series"),("Model","OpenAI CLIP"),("Target runtime","Qualcomm AI Runtime / supported ONNX Runtime path"),("Validation","Not yet performed" if not result else status)]
    for col,(label,value) in zip(cols,entries):
        col.markdown(f"<div class='metric'><small>{label.upper()}</small><strong style='font-size:15px'>{value}</strong></div>",unsafe_allow_html=True)
    st.caption("Target devices: Snapdragon X Elite · Snapdragon X Plus · Snapdragon X2 Elite. AI Hub profile data appears only after a real job result is saved.")
    if result:
        st.write({k:result.get(k) for k in ("job_id","device","job_url","submitted_at","job_status") if result.get(k)})
    elif not __import__('os').environ.get("QAI_HUB_API_TOKEN"):
        st.info("Qualcomm AI Hub validation is optional and has not been configured.")

def render_local_benchmark(root: Path):
    st.markdown("<div class='section'>LOCAL INTEL CPU BENCHMARK</div>", unsafe_allow_html=True)
    path = root / "artifacts" / "local_clip_benchmark.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("label") != "LOCAL INTEL CPU BENCHMARK" or data.get("execution_device") != "CPU":
            raise ValueError("Benchmark artifact is not labeled as a local CPU run.")
        average_ms = float(data["average_inference_time_seconds"]) * 1000
        a, b, c = st.columns(3)
        a.metric("Execution", "LOCAL INTEL CPU")
        b.metric("Model", data["model"])
        c.metric("Average inference", f"~{average_ms:.1f} ms")
        st.caption(f"Measured artifact: {data.get('measured_passes', '—')} passes after {data.get('warmup_passes', '—')} warmup · GPU {data.get('gpu', 'not used')} · {data.get('measured_at', '')}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.info(f"Local CPU benchmark artifact unavailable: {exc}")

