# EDGEFIX AI — Offline Multimodal Field Technician Copilot

## PROBLEM
Field technicians need equipment understanding and maintenance guidance in environments where cloud connectivity is unreliable. Sending site images or maintenance records to cloud services can also be unsuitable for privacy or operational reasons.

## SOLUTION
EdgeFix AI is an offline multimodal prototype that combines an equipment image, a technician question, local technical-document retrieval, and evidence-grounded reasoning. Its primary workflow is **Show me and tell me**: image + question → CLIP visual similarity signal → local manual retrieval → evidence fusion → grounded response.

## DIFFERENTIATION
- **Offline-first:** core retrieval and response flow uses local resources.
- **Multimodal:** image observations are considered alongside a natural-language question and technical documents.
- **Evidence-grounded:** returned excerpts retain document, page, and section provenance.
- **Privacy-aware:** the prototype does not require cloud inference for its local workflow.
- **Snapdragon-targeted:** export and AI Hub integration support a planned target validation path.

## TECHNICAL IMPLEMENTATION
OpenAI CLIP ViT-B/32 provides relative image/text similarity observations; these are not calibrated probabilities, confirmed component identification, or industrial diagnostic accuracy. A local RAG pipeline indexes PDF text and retrieves page-aware passages. Evidence fusion combines retrieved material with the query and optional visual observations. Local reasoning extracts checks from the source material, with uncertainty and safety statements. TF-IDF retrieval and deterministic reasoning provide CPU-friendly fallbacks.

The model was verified locally on an Intel CPU. The end-to-end multimodal workflow was verified locally against four original synthetic manuals and 19 indexed chunks. Those manuals are demonstration content, not manufacturer instructions. Qualcomm AI Hub integration is implemented but not yet validated with a real job. Snapdragon X Series execution, Qualcomm runtime support for this exported configuration, NPU acceleration, and industrial accuracy have not been validated.
