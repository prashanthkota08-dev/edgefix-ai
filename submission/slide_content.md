# Slide 1 — EdgeFix AI
**See it. Ask it. Fix it — offline.** A multimodal field technician copilot designed for unreliable-connectivity environments.

# Slide 2 — Field Technician Problem
Technicians need to interpret equipment context and find useful maintenance guidance in the field. Cloud access may be unreliable, while answers without traceable source material are difficult to trust.

# Slide 3 — Why Cloud-Only AI Is Not Ideal
Connectivity can be unavailable at the point of work. Uploading site images and records may create privacy concerns, latency, or operational dependencies. A local workflow keeps the core demo available offline.

# Slide 4 — EdgeFix Solution
A technician provides a photo and a question. EdgeFix returns model-supported visual observations, locally retrieved manual evidence, recommended checks, safety guidance, uncertainty, and sources.

# Slide 5 — Multimodal Architecture
**Image + question → OpenAI CLIP visual similarity → local PDF RAG → evidence fusion → local reasoning → grounded response.** TF-IDF and deterministic reasoning provide lightweight CPU fallback paths.

# Slide 6 — Evidence-Grounded AI
Responses show document, page, section, and excerpts. Visual similarity is a signal, not a diagnosis. The prototype states uncertainty and presents safety guidance. The four included manuals are synthetic demonstration material.

# Slide 7 — Snapdragon Optimization Path
Development: Intel CPU, locally verified. Target: Snapdragon X Series. Model: OpenAI CLIP. Planned runtime path: Qualcomm AI Runtime or supported ONNX Runtime configuration. Qualcomm AI Hub integration is implemented; a real profile job and target execution have not yet been validated.

# Slide 8 — Live Demo + Results + Future
Locally verified: CLIP CPU inference, four synthetic documents, 19 indexed chunks, and end-to-end multimodal retrieval/fusion/reasoning. Local benchmark: see the measured artifact for the exact model and timing. Next: validate ONNX compatibility and runtime on target hardware, then evaluate on representative industrial data. No Snapdragon, NPU, or industrial accuracy claim is made.
