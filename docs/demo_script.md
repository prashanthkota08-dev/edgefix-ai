# Three-minute demo script

## 0:00 — Problem (20 sec)
“Field technicians need equipment guidance where connectivity can be unreliable. EdgeFix connects a photo and a question to local evidence.”

## 0:20 — Open EdgeFix (15 sec)
Open the Dashboard. Point out the EDGEFIX AI name, the tagline “See it. Ask it. Fix it — offline,” and the ready state.

## 0:35 — Show Offline status (10 sec)
Point to AI MODE LOCAL, NETWORK OFFLINE, CLIP CPU VERIFIED, four documents / 19 chunks, and Snapdragon target NOT YET VALIDATED.

## 0:45 — Upload motor image (15 sec)
Open **Show me and tell me** and upload `data/sample_images/sample_motor.png`. Explain that this is a deterministic synthetic demo image.

## 1:00 — Ask (15 sec)
Ask: “What should I inspect if this motor is overheating?” Submit the question.

## 1:15 — Show CLIP visual signal (15 sec)
Show **VISUAL OBSERVATION**. Describe CLIP output as a visual similarity signal, not a diagnosis or proof of a fault.

## 1:30 — Show manual retrieval (15 sec)
Show **DOCUMENT EVIDENCE** and retrieved manual passages with page and section provenance. Note that the example manuals are synthetic.

## 1:45 — Show grounded answer (25 sec)
Walk through **RECOMMENDED CHECKS** and explain that the suggestions come from retrieved local passages.

## 2:10 — Show sources, safety, uncertainty (20 sec)
Point out **SOURCES**, **SAFETY**, and **UNCERTAINTY**. Emphasize that technicians must follow site procedures and official equipment guidance.

## 2:30 — Show Snapdragon target architecture (20 sec)
Show **SNAPDRAGON OPTIMIZATION PATH**: Intel CPU local development, Snapdragon X Series target, OpenAI CLIP, and Qualcomm AI Runtime / supported ONNX Runtime path.

## 2:50 — CPU verified / validation path (10 sec)
“CLIP and the multimodal workflow are verified locally on Intel CPU. Qualcomm AI Hub integration is implemented, but a real job and Snapdragon execution have not yet been validated.”

## 3:00 — Closing
“EdgeFix: See it. Ask it. Fix it — offline.”
