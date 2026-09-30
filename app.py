from __future__ import annotations

import streamlit as st

from src.config.settings import Settings
from src.rag.retriever import LocalDocumentRetriever
from src.reasoning.engine import build_reasoner
from src.vision.clip_engine import ClipVisionEngine
from src.backends.cpu import CPUBackend
from src.ui.pages import render_app


@st.cache_resource
def services():
    settings = Settings.from_env()
    retriever = LocalDocumentRetriever(settings)
    return settings, retriever, build_reasoner(settings), ClipVisionEngine()


if __name__ == "__main__":
    settings, retriever, reasoner, vision = services()
    render_app(settings, retriever, reasoner, vision, CPUBackend())
