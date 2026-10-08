"""CLIP image tone analysis with ONNX Runtime.

The model (Xenova/clip-vit-base-patch32, quantized) compares an image against
nine fixed descriptions and sums the softmax similarities per sentiment class.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 36_000_000
MAX_ASPECT = 20.0
MAX_EDGE = 1600
CROP = 224

REPO = "Xenova/clip-vit-base-patch32"
REVISION = "d15189d7028b43f1d3e65039190477f6af591c2a"
_MODEL_FILES = [
    "onnx/model_quantized.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
]

PROMPTS: list[tuple[str, str]] = [
    ("positive", "a joyful and cheerful scene"),
    ("positive", "a bright and happy moment"),
    ("positive", "a warm and uplifting atmosphere"),
    ("neutral", "an ordinary everyday scene"),
    ("neutral", "a plain and uneventful moment"),
    ("neutral", "a calm and factual setting"),
    ("negative", "a sad and gloomy scene"),
    ("negative", "a harsh and unpleasant moment"),
    ("negative", "a dark and troubling atmosphere"),
]
CLASSES = ["positive", "neutral", "negative"]


class ImageValidationError(ValueError):
    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


class ImageModelUnavailable(RuntimeError):
    """Raised when CLIP model files or ONNX Runtime are unavailable."""


_CACHE: dict[str, Any] = {}


def _model_dir() -> Path:
    override = os.environ.get("CLIP_MODEL_DIR")
    if override:
        path = Path(override)
        if not (path / "onnx" / "model_quantized.onnx").exists():
            raise ImageModelUnavailable(
                "CLIP_MODEL_DIR does not contain onnx/model_quantized.onnx."
            )
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise ImageModelUnavailable("huggingface_hub is not installed.") from exc
    try:
        cached = Path(
            snapshot_download(
                REPO,
                revision=REVISION,
                allow_patterns=_MODEL_FILES,
            )
        )
    except Exception as exc:
        raise ImageModelUnavailable(
            f"Model files could not be downloaded from Hugging Face: {exc}"
        ) from exc
    return cached


def decode_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        raise ImageValidationError("The image bytes could not be decoded.") from exc
    width, height = image.size
    if width * height > MAX_PIXELS:
        raise ImageValidationError("The image exceeds the 36 megapixel limit.")
    aspect = max(width, height) / max(1, min(width, height))
    if aspect > MAX_ASPECT:
        raise ImageValidationError("The image aspect ratio is greater than 20:1.")
    return image


def prepare_image(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        rgba = image.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")
    if max(image.size) > MAX_EDGE:
        scale = MAX_EDGE / max(image.size)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return image


def _center_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    scale = CROP / min(width, height)
    resized = image.resize(
        (max(CROP, round(width * scale)), max(CROP, round(height * scale))),
        Image.Resampling.BICUBIC,
    )
    left = (resized.width - CROP) // 2
    top = (resized.height - CROP) // 2
    return resized.crop((left, top, left + CROP, top + CROP))


def _pixel_values(image: Image.Image, config: dict[str, Any]) -> np.ndarray:
    crop = _center_crop(image)
    array = np.asarray(crop, dtype=np.float32) / 255.0
    mean = np.array(config.get("mean", [0.48145466, 0.4578275, 0.40821073]), dtype=np.float32)
    std = np.array(config.get("std", [0.26862954, 0.26130258, 0.27577711]), dtype=np.float32)
    array = (array - mean) / std
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)


def _session(model_dir: Path) -> Any:
    key = str(model_dir)
    session = _CACHE.get("session")
    if session is not None and _CACHE.get("session_key") == key:
        return session
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise ImageModelUnavailable("onnxruntime is not installed.") from exc
    onnx_path = model_dir / "onnx" / "model_quantized.onnx"
    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ImageModelUnavailable(f"The ONNX model could not be loaded: {exc}") from exc
    _CACHE["session"] = session
    _CACHE["session_key"] = key
    return session


def _tokenize(texts: list[str], model_dir: Path) -> dict[str, np.ndarray]:
    key = f"tokenizer:{model_dir}"
    tokenizer = _CACHE.get(key)
    if tokenizer is None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - dependency issue
            raise ImageModelUnavailable("tokenizers is not installed.") from exc
        try:
            tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        except Exception as exc:
            raise ImageModelUnavailable(f"The tokenizer could not be loaded: {exc}") from exc
        tokenizer.enable_truncation(max_length=77)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=77)
        _CACHE[key] = tokenizer
    encodings = tokenizer.encode_batch(texts)
    ids = np.array([encoding.ids for encoding in encodings], dtype=np.int64)
    mask = np.array([encoding.attention_mask for encoding in encodings], dtype=np.int64)
    return {"input_ids": ids, "attention_mask": mask}


def _preprocessor_config(model_dir: Path) -> dict[str, Any]:
    key = f"preproc:{model_dir}"
    config = _CACHE.get(key)
    if config is None:
        try:
            with (model_dir / "preprocessor_config.json").open(encoding="utf-8") as handle:
                config = json.load(handle)
        except Exception as exc:
            raise ImageModelUnavailable(f"preprocessor_config.json could not be read: {exc}") from exc
        _CACHE[key] = config
    return config


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exps = np.exp(shifted)
    return exps / float(np.sum(exps))


def aggregate_scores(probs: list[float]) -> dict[str, float]:
    """Sum softmax description scores within each sentiment class."""
    probabilities = {label: 0.0 for label in CLASSES}
    for (label, _), score in zip(PROMPTS, probs):
        probabilities[label] += float(score)
    return probabilities


def is_unclear(probabilities: dict[str, float]) -> bool:
    """Mixed / unclear when the top class is below 0.5 or leads by under 0.1."""
    ranked = sorted(CLASSES, key=lambda label: probabilities[label], reverse=True)
    top, runner_up = ranked[0], ranked[1]
    return probabilities[top] < 0.5 or probabilities[top] - probabilities[runner_up] < 0.1


def analyze_image_bytes(data: bytes, content_type: str) -> dict[str, Any]:
    """Decode, prepare and classify raw image bytes. Runs in a worker thread."""
    if content_type not in SUPPORTED_TYPES:
        raise ImageValidationError("Use a JPG, PNG or WebP image.", status_code=415)
    if len(data) > MAX_BYTES:
        raise ImageValidationError("The image is larger than 10 MB.", status_code=413)

    image = prepare_image(decode_image(data))
    model_dir = _model_dir()
    session = _session(model_dir)
    config = _preprocessor_config(model_dir)

    inputs = _tokenize([text for _, text in PROMPTS], model_dir)
    inputs["pixel_values"] = _pixel_values(image, config)

    try:
        output_names = [output.name for output in session.get_outputs()]
        raw_outputs = session.run(None, inputs)
    except Exception as exc:
        raise ImageModelUnavailable(f"Image inference failed: {exc}") from exc
    outputs = dict(zip(output_names, raw_outputs))

    if "logits_per_image" in outputs:
        logits = np.asarray(outputs["logits_per_image"], dtype=np.float64).reshape(1, -1)[0]
        probs = _softmax(logits)
    elif "image_embeds" in outputs and "text_embeds" in outputs:
        image_embeds = np.asarray(outputs["image_embeds"], dtype=np.float64)
        text_embeds = np.asarray(outputs["text_embeds"], dtype=np.float64)
        image_embeds = image_embeds / np.linalg.norm(image_embeds, axis=-1, keepdims=True)
        text_embeds = text_embeds / np.linalg.norm(text_embeds, axis=-1, keepdims=True)
        probs = _softmax((image_embeds @ text_embeds.T).reshape(-1))
    else:
        raise ImageModelUnavailable("The model did not return usable similarity outputs.")

    probabilities = aggregate_scores(list(map(float, probs)))
    top = sorted(CLASSES, key=lambda label: probabilities[label], reverse=True)[0]
    unclear = is_unclear(probabilities)

    matches = sorted(
        (
            {"label": text, "score": float(score)}
            for (_, text), score in zip(PROMPTS, probs)
        ),
        key=lambda item: item["score"],
        reverse=True,
    )[:3]

    return {
        "label": top,
        "unclear": unclear,
        "probabilities": probabilities,
        "matches": matches,
        "model": f"{REPO}@{REVISION[:7]}",
        "engine": "python",
    }
