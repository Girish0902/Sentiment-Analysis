"""Print model quality diagnostics after training.

Usage: python scripts/report.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.engine import analyze_text, extract_counts, load_model, preprocess  # noqa: E402


def main() -> None:
    model = load_model(force=True)
    report = model["report"]
    print(f"test accuracy     : {report['accuracy']:.4f}")
    print(f"macro precision   : {report['macroPrecision']:.4f}")
    print(f"macro recall      : {report['macroRecall']:.4f}")
    print(f"macro F1          : {report['macroF1']:.4f}")
    print(f"confusion (rows=actual, cols=predicted, order={report['confusionMatrix']['labels']}):")
    for label, row in zip(report["confusionMatrix"]["labels"], report["confusionMatrix"]["rows"]):
        print(f"  {label:9} {row}")

    rows = list(csv.DictReader(open(ROOT / "data" / "test.csv", encoding="utf-8")))
    coverages: list[float] = []
    wrong = 0
    for row in rows:
        result = analyze_text(row["text"], model)
        counts = extract_counts(preprocess(row["text"])["tokens"])
        coverages.append(result["coverage"]["known"] / max(1, result["coverage"]["total"]))
        if result["label"] != row["label"]:
            wrong += 1
            probs = {k: round(v, 2) for k, v in result["probabilities"].items()}
            print(
                f"  MISS true={row['label']:8} pred={result['label']:8} "
                f"p={probs} cov={result['coverage']} :: {row['text'][:90]}"
            )
    print(f"wrong: {wrong}/{len(rows)}")
    print(f"mean test feature coverage: {sum(coverages) / len(coverages):.2%}")


if __name__ == "__main__":
    main()
