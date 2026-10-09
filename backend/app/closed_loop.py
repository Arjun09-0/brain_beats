"""Real-time closed-loop biofeedback engine connecting EEG, encoder, and music generator."""
from __future__ import annotations

import base64
import io
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pretty_midi
from scipy.io import wavfile
import torch

from .eeg.mock_stream import BANDS, _STATE_WEIGHTS, MockEEGConfig, _band_limited_noise
from .eeg.preprocessing import bandpass_filter, extract_band_powers
from .encoder.infer import load_encoder
from .encoder.model import EEGStateEncoder
from .music.generator import FiLMMusicGenerator, _scale, load_music_model, music_controls
from .storage.session_store import SessionStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENCODER_PATH = PROJECT_ROOT / "models" / "encoder" / "synthetic_demo.pt"
DEFAULT_MUSIC_PATH = PROJECT_ROOT / "models" / "music" / "synthetic_demo.pt"

CHANNEL_NAMES = ("Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2")
KEY_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass
class TelemetryFrame:
    frame_id: int
    timestamp: float
    session_id: str
    preset: str
    target_goal: str
    band_powers: dict[str, float]
    channel_band_powers: list[dict[str, float]]
    valence: float
    arousal: float
    latent: list[float]
    tempo_bpm: float
    notes_per_beat: float
    key_root: int
    key_name: str
    mode: str
    biofeedback_score: float
    notes: list[dict[str, Any]]
    raw_eeg_preview: list[list[float]]
    audio_base64: str | None = None
    midi_filename: str | None = None
    wav_filename: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClosedLoopEngine:
    """Orchestrates continuous EEG ingestion, latent encoding, and adaptive music generation."""

    def __init__(
        self,
        encoder_checkpoint: str | Path = DEFAULT_ENCODER_PATH,
        music_checkpoint: str | Path = DEFAULT_MUSIC_PATH,
        storage_dir: str | Path | None = None,
        sample_rate: float = 128.0,
        channels: int = 8,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.channels = channels
        self.encoder_checkpoint = Path(encoder_checkpoint)
        self.music_checkpoint = Path(music_checkpoint)
        self.store = SessionStore(storage_dir) if storage_dir else SessionStore()

        # Load models
        self.encoder_model, self.encoder_meta = load_encoder(self.encoder_checkpoint)
        self.music_model = load_music_model(self.music_checkpoint)
        self.feature_mean = np.asarray(self.encoder_meta["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(self.encoder_meta["feature_std"], dtype=np.float32)

        # Session state
        self.session_id: str | None = None
        self.is_running = False
        self.preset = "calm"
        self.target_goal = "calm"  # 'calm' or 'focus'
        self.frame_counter = 0

        # Circular buffer for raw EEG: 4.0 seconds (512 samples)
        self.buffer_samples = int(self.sample_rate * 4.0)
        self.buffer = np.zeros((self.buffer_samples, self.channels), dtype=np.float64)
        self.latest_frame: TelemetryFrame | None = None

        # Synthetic generator stream continuity state
        self._synth_seed = 100
        self._synth_rng = np.random.default_rng(self._synth_seed)

    def start_session(
        self,
        preset: str = "calm",
        target_goal: str = "calm",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Initialize and start a new closed-loop biofeedback session."""
        self.session_id = session_id or str(uuid.uuid4())
        self.preset = preset if preset in _STATE_WEIGHTS else "calm"
        self.target_goal = target_goal
        self.frame_counter = 0
        self.is_running = True

        # Pre-seed buffer with initial synthetic EEG so first tick has full data
        initial_samples = self.buffer_samples
        eeg, _ = self._generate_synthetic_chunk(initial_samples, self.preset)
        self.buffer[:] = eeg

        metadata = {
            "session_id": self.session_id,
            "preset": self.preset,
            "target_goal": self.target_goal,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "encoder_checkpoint": str(self.encoder_checkpoint),
            "music_checkpoint": str(self.music_checkpoint),
        }
        return self.store.create_session(self.session_id, metadata)

    def stop_session(self) -> dict[str, Any]:
        """Stop the active session and persist completion summary."""
        if not self.session_id:
            return {"status": "no_active_session"}

        sess_id = self.session_id
        summary = {
            "total_frames": self.frame_counter,
            "final_preset": self.preset,
            "final_biofeedback_score": self.latest_frame.biofeedback_score if self.latest_frame else 0.0,
        }
        res = self.store.finish_session(sess_id, summary)
        self.is_running = False
        self.session_id = None
        return res

    def set_preset(self, preset: str) -> str:
        """Dynamically switch the EEG simulation preset."""
        if preset in _STATE_WEIGHTS:
            self.preset = preset
        return self.preset

    def set_target_goal(self, goal: str) -> str:
        """Set biofeedback target goal ('calm' or 'focus')."""
        if goal in ("calm", "focus"):
            self.target_goal = goal
        return self.target_goal

    def ingest_chunk(self, data: np.ndarray) -> None:
        """Ingest external raw EEG samples into circular buffer."""
        chunk = np.asarray(data, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != self.channels:
            raise ValueError(f"Expected chunk shape (samples, {self.channels})")
        n = min(len(chunk), self.buffer_samples)
        self.buffer = np.roll(self.buffer, -n, axis=0)
        self.buffer[-n:] = chunk[-n:]

    def step(self, render_audio: bool = True, beats: int = 16) -> TelemetryFrame:
        """Execute one complete cycle of the closed-loop biofeedback pipeline."""
        self.frame_counter += 1
        now = time.time()

        # 1. Ingest new chunk if running synthetic simulation
        chunk_samples = int(self.sample_rate * 2.0)  # 2-second chunk per tick
        synthetic_chunk, _ = self._generate_synthetic_chunk(chunk_samples, self.preset)
        self.buffer = np.roll(self.buffer, -chunk_samples, axis=0)
        self.buffer[-chunk_samples:] = synthetic_chunk

        # 2. Spectral feature extraction over latest 4 seconds (split into two 2-second epochs)
        epoch_width = int(round(self.sample_rate * 2.0))
        epoch1 = self.buffer[-2 * epoch_width : -epoch_width]
        epoch2 = self.buffer[-epoch_width:]

        feat1 = extract_band_powers(epoch1, self.sample_rate)
        feat2 = extract_band_powers(epoch2, self.sample_rate)
        features = np.stack([feat1.relative, feat2.relative], axis=0).astype(np.float32)  # (2, 8, 5)

        # Compute summary band powers
        mean_relative = feat2.relative.mean(axis=0)
        band_names = tuple(BANDS.keys())
        band_powers = {name: float(mean_relative[i]) for i, name in enumerate(band_names)}

        channel_band_powers = []
        for ch in range(self.channels):
            channel_band_powers.append(
                {name: float(feat2.relative[ch, i]) for i, name in enumerate(band_names)}
            )

        # 3. Phase 2: CNN+LSTM Encoder forward pass
        normalised = (features - self.feature_mean) / self.feature_std
        x_tensor = torch.from_numpy(normalised).unsqueeze(0)  # (1, 2, 8, 5)

        with torch.inference_mode():
            enc_out = self.encoder_model(x_tensor)
            latent = enc_out["latent"].squeeze(0).cpu().numpy()
            v_logits = enc_out["valence_logits"].item()
            a_logits = enc_out["arousal_logits"].item()

        # Valence and Arousal normalized into [-1.0, 1.0]
        valence = float(np.tanh(v_logits / 2.0))
        arousal = float(np.tanh(a_logits / 2.0))
        latent_list = [float(z) for z in latent]

        # 4. Phase 3: Music Macro Controls & Generation
        controls = music_controls(latent)
        tempo_bpm = float(controls["tempo_bpm"])
        notes_per_beat = float(controls["notes_per_beat"])
        key_root = int(controls["key_root"])
        mode = str(controls["mode"])
        key_name = f"{KEY_NAMES[key_root % 12]} {mode.capitalize()}"

        # Generate MIDI sequence conditioned on latent vector
        pm, note_dicts = self._generate_music_sequence(latent, controls, beats)

        # 5. Render Audio and serialize artifacts
        audio_base64 = None
        midi_bytes = None
        wav_bytes = None

        midi_io = io.BytesIO()
        pm.write(midi_io)
        midi_bytes = midi_io.getvalue()

        if render_audio:
            try:
                audio = pm.synthesize(fs=16000)
                audio_clipped = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                wav_io = io.BytesIO()
                wavfile.write(wav_io, 16000, audio_clipped)
                wav_bytes = wav_io.getvalue()
                audio_base64 = base64.b64encode(wav_bytes).decode("ascii")
            except Exception:
                audio_base64 = None

        # 6. Save artifacts if session is active
        midi_fn = None
        wav_fn = None
        if self.session_id:
            midi_fn, wav_fn = self.store.save_music_artifacts(
                self.session_id, self.frame_counter, midi_bytes, wav_bytes
            )

        # 7. Compute biofeedback score (0 to 100)
        # Higher alpha and balanced arousal indicates calm compliance
        alpha_val = band_powers.get("alpha", 0.2)
        beta_val = band_powers.get("beta", 0.2)
        if self.target_goal == "calm":
            # Calm target: rewards alpha dominance and lower arousal
            score = np.clip((alpha_val / (beta_val + 0.1) * 35.0) + (1.0 - arousal) * 25.0, 0, 100)
        else:
            # Focus target: rewards beta presence and moderate-high valence
            score = np.clip((beta_val / (alpha_val + 0.1) * 35.0) + (valence + 1.0) * 25.0, 0, 100)

        # 8. Downsampled raw EEG preview for live oscilloscope (64 points per channel)
        step_stride = max(1, epoch_width // 64)
        raw_preview = [
            [float(self.buffer[-epoch_width + s, ch]) for s in range(0, epoch_width, step_stride)]
            for ch in range(self.channels)
        ]

        frame = TelemetryFrame(
            frame_id=self.frame_counter,
            timestamp=now,
            session_id=self.session_id or "preview",
            preset=self.preset,
            target_goal=self.target_goal,
            band_powers=band_powers,
            channel_band_powers=channel_band_powers,
            valence=valence,
            arousal=arousal,
            latent=latent_list,
            tempo_bpm=tempo_bpm,
            notes_per_beat=notes_per_beat,
            key_root=key_root,
            key_name=key_name,
            mode=mode,
            biofeedback_score=float(score),
            notes=note_dicts,
            raw_eeg_preview=raw_preview,
            audio_base64=audio_base64,
            midi_filename=midi_fn,
            wav_filename=wav_fn,
        )

        self.latest_frame = frame
        if self.session_id:
            self.store.append_telemetry(self.session_id, frame.to_dict())

        return frame

    def _generate_synthetic_chunk(self, samples: int, preset: str) -> tuple[np.ndarray, np.ndarray]:
        """Generate contiguous synthetic chunk with random variation."""
        signal = np.zeros((samples, self.channels), dtype=np.float64)
        weights = _STATE_WEIGHTS.get(preset, _STATE_WEIGHTS["calm"])

        for channel in range(self.channels):
            for band, (low, high) in BANDS.items():
                var = self._synth_rng.uniform(0.9, 1.1)
                noise = _band_limited_noise(samples, self.sample_rate, low, high, self._synth_rng)
                signal[:, channel] += weights[band] * var * noise
            signal[:, channel] += 0.04 * self._synth_rng.standard_normal(samples)

        times = np.arange(samples, dtype=float) / self.sample_rate
        return signal, times

    def _generate_music_sequence(
        self,
        latent: np.ndarray,
        controls: dict[str, Any],
        beats: int,
    ) -> tuple[pretty_midi.PrettyMIDI, list[dict[str, Any]]]:
        """Condition FiLM GRU and produce notes."""
        count = max(4, round(beats * controls["notes_per_beat"]))
        spacing = 60.0 / controls["tempo_bpm"] / controls["notes_per_beat"]
        history = [60] * 32
        z = torch.tensor(np.asarray(latent, dtype=np.float32)).unsqueeze(0)

        pm = pretty_midi.PrettyMIDI(initial_tempo=controls["tempo_bpm"])
        inst = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
        notes_list = []

        with torch.inference_mode():
            for i in range(count):
                logits, _ = self.music_model(torch.tensor([history]), z)
                raw_pitch = int(logits[0, -1].argmax())
                pitch = _scale(raw_pitch, controls["key_root"], controls["mode"])
                start = i * spacing
                dur = spacing * 0.85
                inst.notes.append(
                    pretty_midi.Note(velocity=85, pitch=pitch, start=start, end=start + dur)
                )
                notes_list.append(
                    {
                        "pitch": pitch,
                        "name": pretty_midi.note_number_to_name(pitch),
                        "start": round(start, 3),
                        "duration": round(dur, 3),
                        "velocity": 85,
                    }
                )
                history = history[1:] + [pitch]

        pm.instruments.append(inst)
        return pm, notes_list
