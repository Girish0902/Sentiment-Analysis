"""Show which stemmed features of a text are known to the model.

Usage: python scripts/cov.py "Some text here"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.engine import extract_counts, load_model, preprocess  # noqa: E402


def main() -> None:
    model = load_model(force=True)
    for arg in sys.argv[1:]:
        processed = preprocess(arg)
        counts = extract_counts(processed["tokens"])
        known = [term for term in counts if term in model["_index"]]
        unknown = [term for term in counts if term not in model["_index"]]
        print(f"text     : {arg}")
        print(f"tokens   : {processed['tokens']}")
        print(f"known ({len(known)}/{len(counts)}): {known}")
        print(f"unknown  : {unknown}")
        print()


if __name__ == "__main__":
    main()
