"""CLI entrypoint for training the Phase 3 FiLM MIDI generator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generator import DEFAULT_CHECKPOINT, train_music


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 3 FiLM MIDI generator.")
    parser.add_argument("--corpus", default="data/raw/classical_ancient_midifiles")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    metrics = train_music(args.corpus, args.checkpoint, args.limit, args.epochs)
    Path(args.checkpoint).with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
