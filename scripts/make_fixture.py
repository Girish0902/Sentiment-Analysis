"""Generate scripts/fixtures.json: Python engine predictions for 41 inputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.engine import analyze_text, load_model  # noqa: E402

TEXTS = [
    "This app is fantastic, it never crashes and I love the dark mode!",
    "Customer service was incredibly helpful and resolved my issue in minutes.",
    "The pasta at Roma Trattoria was delicious and the staff were so friendly.",
    "My order arrived early and everything was packed perfectly, I'm really impressed.",
    "This laptop is fast, lightweight, and the battery easily lasts all day.",
    "The movie had me on the edge of my seat from start to finish.",
    "I've read this book three times and it keeps getting better each time.",
    "Shipping was quick and you can follow every step at https://example.com, the product matched the description exactly.",
    "I can't believe how good this show is, the writing is sharp and witty.",
    "It's a great value for the price and I would happily buy it again.",
    "You won't regret downloading this app, the free version is already generous.",
    "Great value! I've recommended this service to all of my friends.",
    "The meeting starts at nine.",
    "The package is scheduled to arrive on Tuesday.",
    "The store opens at ten in the morning and closes at eight.",
    "Chapter four covers the history of the railway system.",
    "The flight departs from gate B12 and boards at 2:30.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The train to the airport runs every fifteen minutes.",
    "The camera on this phone has four lenses and a flash.",
    "Delivery usually takes between two and five business days.",
    "The warranty covers parts for twenty four months.",
    "You can pay by card, cash, or mobile wallet at the counter.",
    "The reception desk is located on the ground floor.",
    "The app crashes every time I open the settings menu, it's unusable.",
    "Customer service put me on hold for an hour and then hung up.",
    "Our meals arrived cold and the order was completely wrong.",
    "The delivery driver left my package in the rain and it was ruined.",
    "This laptop overheats constantly and the fan sounds like a jet engine.",
    "The film was boring, predictable, and far too long.",
    "I've wasted half the book waiting for something interesting to happen.",
    "The flight was delayed six hours with no explanation or compensation.",
    "I don't recommend this hotel, the walls are paper thin.",
    "This phone's battery dies before noon even with light use.",
    "My refund has not arrived after three weeks of waiting.",
    "I'm never flying with this airline again, they lost my luggage twice.",
    "Nobody warned me about the hidden fees, so the final bill shocked me.",
    "zzz qqq vvv jjj www",
    "ok.",
    "It's not worth the price, the material feels cheap and scratches easily.",
    "The service was excellent and the staff were helpful.",
]


def main() -> None:
    model = load_model(force=True)
    fixtures = []
    for text in TEXTS:
        result = analyze_text(text, model)
        fixtures.append(
            {
                "text": text,
                "tokens": result["tokens"],
                "label": result["label"],
                "insufficient": result["insufficient"],
                "probabilities": {key: round(value, 12) for key, value in result["probabilities"].items()},
            }
        )
    out = ROOT / "scripts" / "fixtures.json"
    out.write_text(json.dumps(fixtures, indent=1, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {out} with {len(fixtures)} fixtures")


if __name__ == "__main__":
    main()
