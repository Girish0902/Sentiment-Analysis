"""Unit tests for the image engine (no model download required)."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from backend import image_engine


def image_bytes(size=(16, 16), mode="RGB", fmt="PNG") -> bytes:
    image = Image.new(mode, size, (10, 120, 200) if mode != "RGBA" else (10, 120, 200, 128))
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


class DecodeTests(unittest.TestCase):
    def test_valid_image_decodes(self) -> None:
        image = image_engine.decode_image(image_bytes())
        self.assertEqual(image.size, (16, 16))

    def test_invalid_bytes_rejected(self) -> None:
        with self.assertRaises(image_engine.ImageValidationError):
            image_engine.decode_image(b"definitely not an image")

    def test_aspect_ratio_rejected(self) -> None:
        wide = image_bytes(size=(800, 20))
        with self.assertRaises(image_engine.ImageValidationError):
            image_engine.decode_image(wide)

    def test_megapixel_limit_rejected(self) -> None:
        payload = b"not-an-image"
        image = Image.new("RGB", (10, 10))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload = buffer.getvalue()
        original = image_engine.MAX_PIXELS
        image_engine.MAX_PIXELS = 50
        try:
            with self.assertRaises(image_engine.ImageValidationError):
                image_engine.decode_image(payload)
        finally:
            image_engine.MAX_PIXELS = original


class PrepareTests(unittest.TestCase):
    def test_rgba_flattened_to_rgb(self) -> None:
        prepared = image_engine.prepare_image(Image.open(io.BytesIO(image_bytes(mode="RGBA"))))
        self.assertEqual(prepared.mode, "RGB")

    def test_longest_edge_capped(self) -> None:
        prepared = image_engine.prepare_image(Image.new("RGB", (3200, 800), (1, 2, 3)))
        self.assertEqual(max(prepared.size), 1600)

    def test_center_crop_is_224(self) -> None:
        crop = image_engine._center_crop(Image.new("RGB", (640, 480), (1, 2, 3)))
        self.assertEqual(crop.size, (224, 224))


class RequestValidationTests(unittest.TestCase):
    def test_unsupported_type_status(self) -> None:
        with self.assertRaises(image_engine.ImageValidationError) as ctx:
            image_engine.analyze_image_bytes(b"x", "text/plain")
        self.assertEqual(ctx.exception.status_code, 415)

    def test_oversized_bytes_status(self) -> None:
        with self.assertRaises(image_engine.ImageValidationError) as ctx:
            image_engine.analyze_image_bytes(b"x" * (image_engine.MAX_BYTES + 1), "image/jpeg")
        self.assertEqual(ctx.exception.status_code, 413)

    def test_model_failure_raises(self) -> None:
        previous = os.environ.get("CLIP_MODEL_DIR")
        with tempfile.TemporaryDirectory() as empty:
            os.environ["CLIP_MODEL_DIR"] = empty
            try:
                with self.assertRaises(image_engine.ImageModelUnavailable):
                    image_engine.analyze_image_bytes(image_bytes(), "image/png")
            finally:
                if previous is None:
                    os.environ.pop("CLIP_MODEL_DIR", None)
                else:
                    os.environ["CLIP_MODEL_DIR"] = previous


class ScoringTests(unittest.TestCase):
    def test_aggregation_sums_per_class(self) -> None:
        uniform = [1.0 / len(image_engine.PROMPTS)] * len(image_engine.PROMPTS)
        probabilities = image_engine.aggregate_scores(uniform)
        self.assertEqual(set(probabilities), {"positive", "neutral", "negative"})
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=9)
        self.assertAlmostEqual(probabilities["positive"], 1.0 / 3, places=9)

    def test_softmax_sums_to_one(self) -> None:
        import numpy as np

        probs = image_engine._softmax(np.array([2.0, 1.0, 0.5]))
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=9)

    def test_unclear_rule(self) -> None:
        clear = {"positive": 0.7, "neutral": 0.2, "negative": 0.1}
        self.assertFalse(image_engine.is_unclear(clear))
        low = {"positive": 0.45, "neutral": 0.35, "negative": 0.2}
        self.assertTrue(image_engine.is_unclear(low))
        narrow = {"positive": 0.55, "neutral": 0.48, "negative": 0.0}
        self.assertTrue(image_engine.is_unclear(narrow))


if __name__ == "__main__":
    unittest.main()
