"""Train the text model and export public/model.json.

Usage (from the project root, with backend/requirements-train.txt installed):

    python -m backend.train
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from backend.engine import CLASSES, MODEL_PATH, MODEL_VERSION, ROOT, preprocess

DATA_DIR = ROOT / "data"
LABELS = set(CLASSES)


def read_dataset(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing dataset: {path}")
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise SystemExit(f"{path} must have a header with text and label columns.")
        for line_number, row in enumerate(reader, start=2):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if not text:
                raise SystemExit(f"{path} line {line_number} has an empty text.")
            if label not in LABELS:
                raise SystemExit(f"{path} line {line_number} has an invalid label: {label!r}")
            rows.append((text, label))
    if not rows:
        raise SystemExit(f"{path} contains no data rows.")
    return rows


def stemmed(texts: list[str]) -> list[str]:
    return [" ".join(preprocess(text)["tokens"]) for text in texts]


def classification_report(
    y_true: list[str], y_pred: list[str], test_count: int
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, average="macro", zero_division=0
    )
    per_class = {
        label: {
            "precision": round(float(precision[index]), 6),
            "recall": round(float(recall[index]), 6),
            "f1": round(float(f1[index]), 6),
            "support": int(support[index]),
        }
        for index, label in enumerate(CLASSES)
    }
    return {
        "testExamples": test_count,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macroPrecision": round(float(macro_precision), 6),
        "macroRecall": round(float(macro_recall), 6),
        "macroF1": round(float(macro_f1), 6),
        "perClass": per_class,
        "confusionMatrix": {
            "labels": list(CLASSES),
            "rows": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
        },
    }


def main() -> None:
    train_rows = read_dataset(DATA_DIR / "train.csv")
    test_rows = read_dataset(DATA_DIR / "test.csv")

    train_labels = {label for _, label in train_rows}
    if train_labels != LABELS:
        raise SystemExit("Training data must contain all three classes: positive, neutral, negative.")

    train_texts = [text for text, _ in train_rows]
    train_targets = [label for _, label in train_rows]
    test_texts = [text for text, _ in test_rows]
    test_targets = [label for _, label in test_rows]

    train_docs = stemmed(train_texts)
    test_docs = stemmed(test_texts)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        smooth_idf=True,
        norm="l2",
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
    )
    x_train = vectorizer.fit_transform(train_docs)
    x_test = vectorizer.transform(test_docs)

    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(x_train, train_targets)
    predictions = classifier.predict(x_test).tolist()

    # Confirm that the runtime inference formula reproduces the training matrix.
    from backend.engine import extract_counts, vectorize  # noqa: PLC0415

    artifact: dict[str, Any] = {
        "version": MODEL_VERSION,
        "classes": list(CLASSES),
        "terms": [],
        "idf": [],
        "coefficients": [],
        "intercepts": [],
        "prior": {},
        "training": {},
        "report": {},
    }
    order = [list(classifier.classes_).index(label) for label in CLASSES]
    vocabulary = vectorizer.vocabulary_
    terms = [""] * len(vocabulary)
    for term, position in vocabulary.items():
        terms[position] = term

    artifact["terms"] = terms
    artifact["idf"] = [round(float(value), 10) for value in vectorizer.idf_]
    artifact["coefficients"] = [
        [round(float(value), 10) for value in classifier.coef_[row]] for row in order
    ]
    artifact["intercepts"] = [round(float(classifier.intercept_[row]), 10) for row in order]

    label_counts = {label: 0 for label in CLASSES}
    for label in train_targets:
        label_counts[label] += 1
    artifact["prior"] = {
        label: round(label_counts[label] / len(train_targets), 10) for label in CLASSES
    }
    artifact["training"] = {
        "examples": len(train_rows),
        "features": len(terms),
        "iterations": int(classifier.n_iter_[0]),
        "balanced": len({label_counts[label] for label in CLASSES}) == 1,
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    artifact["report"] = classification_report(test_targets, predictions, len(test_rows))

    artifact["_index"] = {term: position for position, term in enumerate(terms)}
    for row_position, doc in enumerate(test_docs):
        probe = vectorize(extract_counts(doc.split()), artifact)
        reference = x_test[row_position].toarray().ravel()
        for expected, actual in zip(reference, probe):
            if abs(expected - actual) > 1e-9:
                raise SystemExit(
                    "Runtime vectorization does not match the training vectorizer. "
                    "Check backend.engine.vectorize."
                )
    del artifact["_index"]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=True, separators=(",", ":"))

    report = artifact["report"]
    print(f"Exported {MODEL_PATH.relative_to(ROOT)}")
    print(f"  training examples : {artifact['training']['examples']}")
    print(f"  vocabulary terms  : {len(terms)}")
    print(f"  test examples     : {report['testExamples']}")
    print(f"  accuracy          : {report['accuracy']:.3f}")
    print(f"  macro F1          : {report['macroF1']:.3f}")


if __name__ == "__main__":
    main()
