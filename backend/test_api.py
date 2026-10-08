"""API tests for the FastAPI backend."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from backend import engine
from backend.main import app

ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)


def png_bytes(width: int = 8, height: int = 8, mode: str = "RGB") -> bytes:
    image = Image.new(mode, (width, height), (120, 160, 200) if mode == "RGB" else (120, 160, 200, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class HealthTests(unittest.TestCase):
    def test_health(self) -> None:
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["engine"], "python")
        self.assertTrue(payload["modelVersion"])

    def test_model_info(self) -> None:
        response = client.get("/api/model")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["classes"], engine.CLASSES)
        self.assertIn("accuracy", payload["report"])
        self.assertEqual(payload["limits"]["maxTexts"], 200)


class AnalyzeTests(unittest.TestCase):
    def post(self, items: list[dict]) -> dict:
        response = client.post("/api/analyze", json={"items": items})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_all_three_classes(self) -> None:
        payload = self.post(
            [
                {"text": "This is great, I love it!"},
                {"text": "The package arrives on Tuesday."},
                {"text": "Terrible, the item broke and the seller refused a refund."},
            ]
        )
        labels = [item["label"] for item in payload["results"]]
        self.assertEqual(labels, ["positive", "neutral", "negative"])

    def test_probabilities_sum_to_one(self) -> None:
        payload = self.post([{"text": "Absolutely wonderful service!"}])
        probs = payload["results"][0]["probabilities"]
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)

    def test_negation_is_preserved(self) -> None:
        processed = engine.preprocess("I do not like this at all")
        self.assertNotIn("not", processed["removedStopWords"])
        self.assertIn("not", processed["tokens"])
        payload = self.post([{"text": "I do not like this at all."}])
        self.assertNotEqual(payload["results"][0]["label"], "positive")

    def test_processing_details_returned(self) -> None:
        payload = self.post([{"text": "The service was excellent and the staff were helpful."}])
        result = payload["results"][0]
        self.assertEqual(result["tokens"], ["service", "excellent", "staff", "helpful"])
        self.assertIn("the", result["removedStopWords"])
        self.assertTrue(result["features"])

    def test_unknown_words_flagged(self) -> None:
        payload = self.post([{"text": "zzz qqq vvv www jjj"}])
        result = payload["results"][0]
        self.assertTrue(result["insufficient"])
        self.assertTrue(result["warnings"])
        prior = engine.load_model()["prior"]
        for label in engine.CLASSES:
            self.assertAlmostEqual(result["probabilities"][label], prior[label], places=6)

    def test_blank_text_rejected(self) -> None:
        response = client.post("/api/analyze", json={"items": [{"text": "   "}]})
        self.assertEqual(response.status_code, 422)

    def test_over_limit_rejected(self) -> None:
        response = client.post("/api/analyze", json={"items": [{"text": "x" * 5001}]})
        self.assertEqual(response.status_code, 422)

    def test_too_many_items_rejected(self) -> None:
        items = [{"text": "fine"} for _ in range(201)]
        response = client.post("/api/analyze", json={"items": items})
        self.assertEqual(response.status_code, 422)

    def test_invalid_json_rejected(self) -> None:
        response = client.post(
            "/api/analyze", content=b"{not json", headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status_code, 422)


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rows = list(csv.DictReader((ROOT / "data" / "test.csv").open(encoding="utf-8")))
        cls.items = [{"text": row["text"], "label": row["label"]} for row in rows]
        response = client.post("/api/evaluate", json={"items": cls.items})
        assert response.status_code == 200, response.text
        cls.payload = response.json()

    def test_results_match_input(self) -> None:
        self.assertEqual(len(self.payload["results"]), len(self.items))

    def test_metrics_arithmetic(self) -> None:
        metrics = self.payload["metrics"]
        results = self.payload["results"]
        correct = sum(1 for item in results if item["correct"])
        self.assertAlmostEqual(metrics["accuracy"], correct / len(results), places=6)
        matrix = metrics["confusionMatrix"]["rows"]
        self.assertEqual(sum(sum(row) for row in matrix), len(results))
        f1_values = [metrics["perClass"][label]["f1"] for label in engine.CLASSES]
        self.assertAlmostEqual(metrics["macroF1"], sum(f1_values) / 3, places=6)
        precision_values = [metrics["perClass"][label]["precision"] for label in engine.CLASSES]
        self.assertAlmostEqual(metrics["macroPrecision"], sum(precision_values) / 3, places=6)

    def test_missing_label_rejected(self) -> None:
        response = client.post("/api/evaluate", json={"items": [{"text": "hello"}]})
        self.assertEqual(response.status_code, 422)

    def test_invalid_label_rejected(self) -> None:
        response = client.post(
            "/api/evaluate", json={"items": [{"text": "hello", "label": "great"}]}
        )
        self.assertEqual(response.status_code, 422)

    def test_train_and_test_texts_are_separate(self) -> None:
        train = {
            row["text"] for row in csv.DictReader((ROOT / "data" / "train.csv").open(encoding="utf-8"))
        }
        test = {row["text"] for row in csv.DictReader((ROOT / "data" / "test.csv").open(encoding="utf-8"))}
        self.assertEqual(train & test, set())


class ImageEndpointTests(unittest.TestCase):
    def test_unsupported_type(self) -> None:
        response = client.post(
            "/api/analyze-image", content=b"hello", headers={"Content-Type": "text/plain"}
        )
        self.assertEqual(response.status_code, 415)

    def test_oversized_upload(self) -> None:
        response = client.post(
            "/api/analyze-image",
            content=b"x" * (10 * 1024 * 1024 + 1),
            headers={"Content-Type": "image/jpeg"},
        )
        self.assertEqual(response.status_code, 413)

    def test_invalid_image_bytes(self) -> None:
        response = client.post(
            "/api/analyze-image",
            content=b"this is not an image",
            headers={"Content-Type": "image/png"},
        )
        self.assertEqual(response.status_code, 422)

    def test_model_failure_returns_503(self) -> None:
        previous = os.environ.get("CLIP_MODEL_DIR")
        with tempfile.TemporaryDirectory() as empty:
            os.environ["CLIP_MODEL_DIR"] = empty
            try:
                response = client.post(
                    "/api/analyze-image",
                    content=png_bytes(),
                    headers={"Content-Type": "image/png"},
                )
            finally:
                if previous is None:
                    os.environ.pop("CLIP_MODEL_DIR", None)
                else:
                    os.environ["CLIP_MODEL_DIR"] = previous
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
