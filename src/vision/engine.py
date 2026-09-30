from dataclasses import dataclass
from io import BytesIO


@dataclass
class VisionResult:
    status: str
    summary: str
    details: dict


class VisionEngine:
    def analyze(self, image_bytes: bytes) -> VisionResult:
        raise NotImplementedError


class BasicVisionEngine(VisionEngine):
    """Image validity and metadata only; this is not visual diagnosis."""
    def analyze(self, image_bytes: bytes) -> VisionResult:
        try:
            from PIL import Image, ImageStat
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            stats = ImageStat.Stat(image)
            return VisionResult("Visual analysis unavailable", "Image received; no vision model is configured. Metadata and basic image statistics only.",
                                {"width": image.width, "height": image.height, "format": image.format or "decoded image",
                                 "mean_rgb": [round(v, 1) for v in stats.mean]})
        except Exception as exc:
            return VisionResult("Invalid image", f"The uploaded file could not be decoded as an image: {exc}", {})
