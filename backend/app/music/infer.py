"""Deterministic music generation and audio rendering from a 12-D latent vector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io.wavfile import write

from .generator import DEFAULT_CHECKPOINT, generate_midi, midi_metrics


def infer_music(
    latent: np.ndarray | list[float],
    output_midi: str | Path,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    beats: int = 16,
    render_audio: bool = True,
) -> dict[str, Any]:
    """Generate MIDI and optional WAV audio preview from latent conditioning."""
    pm, controls = generate_midi(np.asarray(latent, dtype=np.float32), checkpoint_path, beats)
    out = Path(output_midi)
    out.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out))
    result: dict[str, Any] = {"midi": str(out), "controls": controls, "metrics": midi_metrics(pm)}

    if render_audio:
        try:
            audio = pm.synthesize(fs=16000)
            wav = out.with_suffix(".wav")
            write(str(wav), 16000, (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16))
            result["audio"] = str(wav)
        except Exception as error:
            result["audio_unavailable"] = str(error)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MIDI from a 12-D EEG latent.")
    parser.add_argument("--latent", nargs=12, type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--beats", type=int, default=16)
    parser.add_argument("--midi-only", action="store_true")
    args = parser.parse_args()

    result = infer_music(args.latent, args.output, args.checkpoint, args.beats, not args.midi_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
