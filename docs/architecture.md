# Architecture

Streamlit UI calls four independent interfaces: `DocumentRetriever`, `VisionEngine`, `ReasoningEngine`, and `InferenceBackend`. The current retriever extracts PDF pages through PyMuPDF, creates overlapping chunks with page provenance, persists JSON, then ranks candidates using offline TF-IDF. No database daemon is necessary.

`ClipVisionEngine` lazily loads `openai/clip-vit-base-patch32` through Transformers on CPU. A process cache avoids duplicate model construction. It embeds the image and a fixed text candidate set, ranks cosine similarities, and reports relative scores without labeling them probabilities or diagnoses. Missing optional packages, cached weights, or model execution return a clear unavailable result without stopping app startup.

`SemanticRetriever` optionally embeds page-aware chunks with `sentence-transformers/all-MiniLM-L6-v2` on CPU and persists embeddings keyed by content/model fingerprint. Any failure falls back to the existing local TF-IDF retriever. `EvidenceFusion` packages visual observations, question, and retrieved chunks into a structured context for reasoning.

`OllamaReasoningEngine` uses a local HTTP endpoint only when its configured model is already available. The deterministic fallback runs without Ollama and returns evidence-grounded passage excerpts. Missing manual evidence produces the explicit insufficiency statement. CLIP model observations never establish a fault.

`SnapdragonBackend` is only an interface placeholder. A future runtime adapter should implement the same backend contract and be selected only after supported devices and actual execution are verified. Model and retrieval abstractions should remain independent of the UI.

## Model provenance and validation boundary

The vision model is OpenAI CLIP ViT-B/32 loaded from `openai/clip-vit-base-patch32` through Transformers. It was not trained by EdgeFix. The model card presents it as a research model and warns about deployment and task-specific limitations; industrial condition labels have not been validated against a representative dataset. The reference software license and downloadable weight/repository terms must be reviewed separately before redistribution. See the model card at https://huggingface.co/openai/clip-vit-base-patch32.

The optional semantic model is `sentence-transformers/all-MiniLM-L6-v2`. Both models download only when explicitly selected/used, can then run from local cache, and are not required for app startup. AI Hub profile output is considered present only when created by the actual profile command.
