"""Text sentiment engine: preprocessing, TF-IDF features, inference and evaluation.

This module is the single source of truth for the text pipeline. The training
script (backend.train) exports its artifacts to public/model.json, which is
consumed both by this engine and by the browser engine (lib/sentiment.ts).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

CLASSES: list[str] = ["positive", "neutral", "negative"]
MODEL_VERSION = "1.1.0"

MAX_ITEMS = 200
MAX_CHARS = 5000
MAX_CSV_BYTES = 1_048_576

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "public" / "model.json"


class InputError(ValueError):
    """Raised when a request payload violates the documented limits."""


class ModelUnavailableError(RuntimeError):
    """Raised when the exported model artifact cannot be loaded."""


_URL_RE = re.compile(r"(?:https?://|www\.)\S+")
_WORD_RE = re.compile(r"[a-z]+")

CONTRACTIONS: dict[str, str] = {
    "i'm": "i am",
    "i've": "i have",
    "i'd": "i would",
    "i'll": "i will",
    "you're": "you are",
    "you've": "you have",
    "you'd": "you would",
    "you'll": "you will",
    "he's": "he is",
    "he'd": "he would",
    "he'll": "he will",
    "she's": "she is",
    "she'd": "she would",
    "she'll": "she will",
    "it's": "it is",
    "it'd": "it would",
    "it'll": "it will",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "what's": "what is",
    "who's": "who is",
    "let's": "let us",
    "we're": "we are",
    "we've": "we have",
    "we'd": "we would",
    "we'll": "we will",
    "they're": "they are",
    "they've": "they have",
    "they'd": "they would",
    "they'll": "they will",
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "mustn't": "must not",
    "shan't": "shall not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "ain't": "is not",
}

_CONTRACTION_RULES = [
    (re.compile(r"\b" + re.escape(key) + r"\b"), value)
    for key, value in sorted(CONTRACTIONS.items(), key=lambda kv: -len(kv[0]))
]

# Negation words are deliberately absent: not, no, nor, never.
STOP_WORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because
    been before being below between both by can could did do does doing down
    during each few for from further had has have having he her here hers
    herself him himself his how i if in into is it its itself just me more
    most my myself of off on once only or other our ours ourselves out over
    own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until
    up very was we were what when where which while who whom why will with
    you your yours yourself yourselves
    """.split()
)

_VOWELS = frozenset("aeiou")


def _undouble(base: str) -> str:
    if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in _VOWELS:
        return base[:-1]
    return base


def stem(word: str) -> str:
    """Small deterministic suffix stemmer shared with the browser engine."""
    if len(word) <= 3:
        return word
    w = word
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith(("sses", "ches", "shes", "xes", "zes")) and len(w) > 5:
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if w.endswith("ingly") and len(w) > 6 and len(w[:-5]) >= 3:
        w = w[:-5]
    elif w.endswith("edly") and len(w) > 5 and len(w[:-4]) >= 3:
        w = w[:-4]
    elif w.endswith("ing") and len(w) > 4 and len(w[:-3]) >= 3:
        w = _undouble(w[:-3])
    elif w.endswith("ed") and len(w) > 4:
        base = w[:-2]
        if len(base) >= 3:
            if base[-1] == "t" and base[-2] in "kpsfxh":
                base = base[:-1]
            else:
                base = _undouble(base)
            w = base
    if w.endswith("ly") and len(w) > 4 and len(w[:-2]) >= 3:
        w = w[:-2]
    elif w.endswith("ness") and len(w) > 5:
        w = w[:-4]
    elif w.endswith("ment") and len(w) > 5:
        w = w[:-4]
    elif w.endswith("ation") and len(w) > 6 and len(w[:-5]) >= 3:
        w = w[:-5]
    return w


def preprocess(text: str) -> dict[str, Any]:
    """Clean, tokenize, de-stop-word and stem an English text."""
    lowered = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    without_urls = _URL_RE.sub(" ", lowered)
    expanded = without_urls
    for pattern, replacement in _CONTRACTION_RULES:
        expanded = pattern.sub(replacement, expanded)

    words = _WORD_RE.findall(expanded)
    removed = [word for word in words if word in STOP_WORDS]
    kept = [word for word in words if word not in STOP_WORDS]
    tokens = [stem(word) for word in kept]
    cleaned = " ".join(kept)
    return {
        "cleaned": cleaned,
        "removedStopWords": removed,
        "tokens": tokens,
    }


def extract_counts(tokens: list[str]) -> dict[str, int]:
    """Count stemmed unigrams and adjacent bigrams."""
    counts: dict[str, int] = {}
    for index, token in enumerate(tokens):
        counts[token] = counts.get(token, 0) + 1
        if index + 1 < len(tokens):
            bigram = token + " " + tokens[index + 1]
            counts[bigram] = counts.get(bigram, 0) + 1
    return counts


def load_model(force: bool = False) -> dict[str, Any]:
    """Load public/model.json, caching it until the file changes on disk."""
    if not MODEL_PATH.exists():
        raise ModelUnavailableError(
            "Model file public/model.json is missing. Run `python -m backend.train` first."
        )
    mtime = MODEL_PATH.stat().st_mtime_ns
    cached = _MODEL_CACHE.get("model")
    if cached is not None and _MODEL_CACHE.get("mtime") == mtime and not force:
        return cached
    try:
        with MODEL_PATH.open(encoding="utf-8") as handle:
            model = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelUnavailableError(f"Model file could not be read: {exc}") from exc
    if not model.get("terms") or not model.get("coefficients"):
        raise ModelUnavailableError("Model file is incomplete.")
    model["_index"] = {term: position for position, term in enumerate(model["terms"])}
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["mtime"] = mtime
    return model


_MODEL_CACHE: dict[str, Any] = {"model": None, "mtime": None}


def vectorize(counts: dict[str, int], model: dict[str, Any]) -> list[float]:
    """Sublinear TF x smoothed IDF with L2 normalization, matching TfidfVectorizer."""
    idf = model["idf"]
    index = model["_index"]
    vector = [0.0] * len(model["terms"])
    for term, count in counts.items():
        position = index.get(term)
        if position is None:
            continue
        vector[position] = (1.0 + math.log(count)) * idf[position]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm > 0:
        vector = [value / norm for value in vector]
    return vector


def _softmax(scores: list[float]) -> list[float]:
    peak = max(scores)
    exps = [math.exp(score - peak) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


def _validate_text(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        raise InputError("Text is empty. Enter a non-blank English text.")
    if len(text) > MAX_CHARS:
        raise InputError(f"Text is longer than the {MAX_CHARS} character limit.")
    return text


def analyze_text(text: str, model: dict[str, Any] | None = None) -> dict[str, Any]:
    """Classify one text and return scores plus processing details."""
    _validate_text(text)
    if model is None:
        model = load_model()

    processed = preprocess(text)
    tokens: list[str] = processed["tokens"]
    counts = extract_counts(tokens)
    known = [term for term in counts if term in model["_index"]]
    vector = vectorize(counts, model)

    warnings: list[str] = []
    insufficient = not known
    if insufficient:
        probabilities = {label: float(model["prior"][label]) for label in CLASSES}
        warnings.append(
            "No known model features were found in this text, so the scores "
            "reflect the training class prior."
        )
    else:
        scores = [
            intercept + sum(weight * value for weight, value in zip(row, vector))
            for row, intercept in zip(model["coefficients"], model["intercepts"])
        ]
        values = _softmax(scores)
        probabilities = dict(zip(CLASSES, values))
        if len(known) < len(counts):
            hidden = len(counts) - len(known)
            warnings.append(
                f"{hidden} of {len(counts)} text features are not in the model "
                "vocabulary and were ignored."
            )

    label = max(CLASSES, key=lambda name: probabilities[name])

    features = sorted(
        (
            {"term": term, "weight": vector[model["_index"][term]], "count": counts[term]}
            for term in known
        ),
        key=lambda item: (-item["weight"], item["term"]),
    )

    return {
        "text": text,
        "label": label,
        "probabilities": probabilities,
        "insufficient": insufficient,
        "warnings": warnings,
        "cleaned": processed["cleaned"],
        "removedStopWords": processed["removedStopWords"],
        "tokens": tokens,
        "features": features,
        "coverage": {"known": len(known), "total": len(counts)},
    }


def _prf(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy, macro averages, per-class metrics and a confusion matrix."""
    total = len(results)
    correct = sum(1 for item in results if item["label"] == item["actual"])
    matrix = [[0 for _ in CLASSES] for _ in CLASSES]
    index = {label: position for position, label in enumerate(CLASSES)}
    for item in results:
        matrix[index[item["actual"]]][index[item["label"]]] += 1

    per_class: dict[str, dict[str, float]] = {}
    for row_offset, label in enumerate(CLASSES):
        true_positive = matrix[row_offset][row_offset]
        false_positive = sum(matrix[row][row_offset] for row in range(len(CLASSES)) if row != row_offset)
        false_negative = sum(matrix[row_offset][col] for col in range(len(CLASSES)) if col != row_offset)
        metrics = _prf(true_positive, false_positive, false_negative)
        metrics["support"] = float(sum(matrix[row_offset]))
        per_class[label] = metrics

    return {
        "accuracy": correct / total if total else 0.0,
        "macroPrecision": sum(per_class[label]["precision"] for label in CLASSES) / len(CLASSES),
        "macroRecall": sum(per_class[label]["recall"] for label in CLASSES) / len(CLASSES),
        "macroF1": sum(per_class[label]["f1"] for label in CLASSES) / len(CLASSES),
        "perClass": per_class,
        "confusionMatrix": {"labels": list(CLASSES), "rows": matrix},
    }


def _validate_items(items: Any, require_labels: bool) -> list[dict[str, Any]]:
    if not isinstance(items, list) or not items:
        raise InputError("Provide at least one text item.")
    if len(items) > MAX_ITEMS:
        raise InputError(f"A maximum of {MAX_ITEMS} texts can be processed at once.")
    cleaned: list[dict[str, Any]] = []
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise InputError(f"Item {position + 1} is not a valid object.")
        text = _validate_text(item.get("text"))
        entry: dict[str, Any] = {"text": text}
        if "label" in item and item["label"] is not None:
            label = item["label"]
            if label not in CLASSES:
                raise InputError(
                    f"Item {position + 1} has an invalid label. Use positive, neutral or negative."
                )
            entry["label"] = label
        elif require_labels:
            raise InputError(
                f"Item {position + 1} is missing a label. Every row needs positive, "
                "neutral or negative."
            )
        cleaned.append(entry)
    return cleaned


def analyze_items(items: Any, model: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if model is None:
        model = load_model()
    return [analyze_text(item["text"], model) for item in _validate_items(items, False)]


def evaluate_items(items: Any, model: dict[str, Any] | None = None) -> dict[str, Any]:
    """Analyze labeled items and compare predictions against known labels."""
    if model is None:
        model = load_model()
    validated = _validate_items(items, True)
    results: list[dict[str, Any]] = []
    for item in validated:
        result = analyze_text(item["text"], model)
        result["actual"] = item["label"]
        result["correct"] = result["label"] == item["label"]
        results.append(result)
    return {"results": results, "metrics": compute_metrics(results)}
