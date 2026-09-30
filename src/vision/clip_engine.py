"""Lazy, CPU-only OpenAI CLIP image/text similarity adapter."""
from __future__ import annotations

from dataclasses import dataclass
import os
import time
from threading import Lock
from typing import Any

from .engine import VisionEngine, VisionResult

MODEL_ID = "openai/clip-vit-base-patch32"
EQUIPMENT_LABELS = ["electric motor", "industrial motor", "pump", "fan", "gearbox",
                    "electrical panel", "belt drive", "bearing housing", "terminal box",
                    "industrial electric motor"]
CONDITION_LABELS = ["normal equipment appearance", "visible corrosion", "possible overheating indication",
                    "damaged component", "obstructed ventilation", "belt misalignment",
                    "oil leakage", "worn component", "cooling system", "ventilation opening",
                    "corroded metal"]
ALL_LABELS = EQUIPMENT_LABELS + CONDITION_LABELS
_CACHE: dict[str, tuple[Any, Any, Any]] = {}
_LOCK = Lock()


def rank_concepts(scores):
    return sorted(({"label": label, "similarity": round(float(score), 4)}
                   for label, score in zip(ALL_LABELS, scores)), key=lambda item: item["similarity"], reverse=True)


def _load(model_id: str, local_files_only: bool):
    key = f"{model_id}|{local_files_only}"
    with _LOCK:
        if key not in _CACHE:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            os.environ.setdefault("HF_HOME", os.path.join(project_root, "models", "huggingface"))
            import torch
            from transformers import CLIPModel, CLIPProcessor
            processor = CLIPProcessor.from_pretrained(model_id, local_files_only=local_files_only)
            model = CLIPModel.from_pretrained(model_id, local_files_only=local_files_only).to("cpu").eval()
            _CACHE[key] = (torch, processor, model)
        return _CACHE[key]


def decode_image(image_bytes: bytes):
    from io import BytesIO
    from PIL import Image
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def preprocess_image(image, processor):
    prompts = [f"A photo of {label} in an industrial maintenance setting." for label in ALL_LABELS]
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True, truncation=True)
    return prompts, inputs


def prepare_cpu_inputs(inputs):
    return {key: value.to("cpu") for key, value in inputs.items()}


def clip_environment_status(project_root):
    import importlib.util
    from pathlib import Path
    installed = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
    cache_root = Path(os.environ.get("HF_HOME", Path(project_root)/"models"/"huggingface"))
    os.environ.setdefault("HF_HOME", str(cache_root))
    cached = False
    if installed:
        try:
            from huggingface_hub import try_to_load_from_cache
            for filename in ("model.safetensors", "pytorch_model.bin"):
                cached_path = try_to_load_from_cache(MODEL_ID, filename, cache_dir=str(cache_root/"hub"))
                if isinstance(cached_path, str) and Path(cached_path).is_file() and Path(cached_path).stat().st_size > 0:
                    cached = True
                    break
        except Exception:
            cached = False
    return {"installed": installed, "model": MODEL_ID, "weights_cached": cached, "device": "CPU"}


class ClipVisionEngine(VisionEngine):
    """Produces relative cosine similarity signals, never a fault diagnosis."""
    def __init__(self, model_id: str = MODEL_ID, local_files_only: bool | None = None):
        offline = os.environ.get("EDGEFIX_OFFLINE", "0").lower() in {"1", "true", "yes"}
        self.model_id, self.local_files_only = model_id, offline if local_files_only is None else local_files_only
        self._load_error: str | None = None

    def analyze(self, image_bytes: bytes) -> VisionResult:
        try:
            image = decode_image(image_bytes)
        except Exception as exc:
            return VisionResult("Invalid image", f"The uploaded file could not be decoded: {exc}", {})
        try:
            load_started = time.perf_counter()
            torch, processor, model = _load(self.model_id, self.local_files_only)
            model_load_seconds = time.perf_counter() - load_started
            preprocessing_started = time.perf_counter()
            prompts, inputs = preprocess_image(image, processor)
            inputs = prepare_cpu_inputs(inputs)
            preprocessing_seconds = time.perf_counter() - preprocessing_started
            inference_started = time.perf_counter()
            with torch.inference_mode():
                output = model(**inputs)
                image_embedding = output.image_embeds / output.image_embeds.norm(dim=-1, keepdim=True)
                text_embedding = output.text_embeds / output.text_embeds.norm(dim=-1, keepdim=True)
                scores = (image_embedding @ text_embedding.T).squeeze(0).tolist()
            inference_seconds = time.perf_counter() - inference_started
            ranked = rank_concepts(scores)
            best = ranked[:5]
            observations = [f"{item['label']} — relative similarity {item['similarity']:.2f}" for item in best]
            return VisionResult("Visual similarity signal", "CLIP ranked text concepts by image/text similarity. Similarity is relative to this candidate list, is not a calibrated probability, and does not establish a fault.",
                                {"model": self.model_id, "device": "CPU", "observations": best,
                                 "timings": {"model_load_seconds": round(model_load_seconds, 4), "preprocessing_seconds": round(preprocessing_seconds, 4), "inference_seconds": round(inference_seconds, 4)},
                                 "labels": ALL_LABELS,
                                 "equipment": [x for x in ranked if x["label"] in EQUIPMENT_LABELS][:4],
                                 "conditions": [x for x in ranked if x["label"] in CONDITION_LABELS][:4]})
        except Exception as exc:
            self._load_error = str(exc)
            return VisionResult("Vision model unavailable", f"CLIP could not be loaded or run on CPU. Install the optional vision dependencies and download the model, then retry. Details: {exc}",
                                {"model": self.model_id, "device": "CPU", "observations": []})
