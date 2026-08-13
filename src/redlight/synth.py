"""Renders the chant and the elimination buzzer as WAV audio.

`chant.json` holds the melody as data — `{"bpm": ..., "notes": [[midi,
beats], ...]}` — so it can be baked into the browser build (Task 11) as the
same source of truth, without either renderer depending on the other.
Everything here is pure arithmetic on that data: a sine oscillator per
note, shaped by a short attack / decay / sustain / release envelope so
notes don't click at their edges, laid back-to-back on the chant's own
clock. No randomness and no wall-clock reads mean the same input always
produces the same bytes.

The buzzer is a short square-wave burst built the same way, for the
"you're out" cue.
"""

from __future__ import annotations

import io
import json
import wave
from importlib import resources

import numpy as np

CHANT_RESOURCE = "chant.json"

BUZZER_FREQUENCY_HZ = 146.83
"""Pitch of the elimination buzzer — a low D, well outside the chant's
pentatonic scale so it reads as a distinct, unmusical alarm rather than a
wrong note in the melody."""

BUZZER_DURATION_S = 0.4


def load_chant() -> dict:
    """Load the chant melody from the package's bundled `chant.json`.

    Uses `importlib.resources` so this works whether `redlight` is
    installed as a package or just sitting on `PYTHONPATH`.

    Returns:
        `{"bpm": float, "notes": [[midi, beats], ...]}`.
    """
    text = resources.files("redlight").joinpath(CHANT_RESOURCE).read_text(encoding="utf-8")
    return json.loads(text)


def midi_to_hz(midi: float) -> float:
    """Convert a MIDI note number to frequency, A4 (69) = 440 Hz."""
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _envelope(
    n_samples: int,
    sample_rate: int,
    *,
    attack_s: float,
    decay_s: float,
    sustain_level: float,
    release_s: float,
) -> np.ndarray:
    """An attack/decay/sustain/release amplitude curve, `n_samples` long.

    Each stage is clamped to what actually fits: a note shorter than the
    requested attack+decay+release still gets a sensible (if compressed)
    shape rather than a negative-length array.
    """
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float64)

    attack_n = min(int(round(attack_s * sample_rate)), n_samples)
    decay_n = min(int(round(decay_s * sample_rate)), n_samples - attack_n)
    release_n = min(int(round(release_s * sample_rate)), n_samples - attack_n - decay_n)
    sustain_n = n_samples - attack_n - decay_n - release_n

    stages = []
    if attack_n:
        stages.append(np.linspace(0.0, 1.0, attack_n, endpoint=False))
    if decay_n:
        stages.append(np.linspace(1.0, sustain_level, decay_n, endpoint=False))
    if sustain_n:
        stages.append(np.full(sustain_n, sustain_level))
    if release_n:
        stages.append(np.linspace(sustain_level, 0.0, release_n))

    env = np.concatenate(stages) if stages else np.zeros(n_samples)
    return env[:n_samples]


def _sine_note(midi: int, duration_s: float, sample_rate: int) -> np.ndarray:
    """One enveloped sine-wave note, `duration_s` long."""
    n = max(int(round(duration_s * sample_rate)), 1)
    t = np.arange(n) / sample_rate
    tone = np.sin(2.0 * np.pi * midi_to_hz(midi) * t)
    env = _envelope(
        n,
        sample_rate,
        attack_s=min(0.01, duration_s * 0.25),
        decay_s=min(0.02, duration_s * 0.25),
        sustain_level=0.7,
        release_s=min(0.05, duration_s * 0.4),
    )
    return tone * env


def _synthesize_chant(chant: dict, sample_rate: int) -> np.ndarray:
    """Render every note of a chant back-to-back at the chant's own tempo."""
    beat_s = 60.0 / chant["bpm"]
    notes = [
        _sine_note(midi, beats * beat_s, sample_rate) for midi, beats in chant["notes"]
    ]
    return np.concatenate(notes) if notes else np.zeros(0, dtype=np.float64)


def _synthesize_buzzer(sample_rate: int) -> np.ndarray:
    """A short, harsh square-wave burst for the elimination cue."""
    n = max(int(round(BUZZER_DURATION_S * sample_rate)), 1)
    t = np.arange(n) / sample_rate
    tone = np.sign(np.sin(2.0 * np.pi * BUZZER_FREQUENCY_HZ * t))
    env = _envelope(
        n,
        sample_rate,
        attack_s=0.005,
        decay_s=0.05,
        sustain_level=0.6,
        release_s=0.15,
    )
    return tone * env


def _to_int16(samples: np.ndarray) -> np.ndarray:
    """Peak-normalise to 16-bit PCM with a little headroom, so it never clips."""
    if samples.size == 0:
        return samples.astype(np.int16)
    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        samples = samples / peak * 0.9
    return np.clip(np.round(samples * 32767.0), -32768, 32767).astype(np.int16)


def _wav_bytes(int16_samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono 16-bit PCM samples as WAV bytes via the stdlib `wave` module."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_samples.tobytes())
    return buf.getvalue()


def chant_wav_bytes(sample_rate: int = 22050) -> tuple[bytes, float]:
    """Render the packaged chant to in-memory WAV bytes.

    Returns:
        (wav_bytes, duration_s).
    """
    raw = _synthesize_chant(load_chant(), sample_rate)
    duration_s = raw.size / sample_rate
    return _wav_bytes(_to_int16(raw), sample_rate), duration_s


def buzzer_wav_bytes(sample_rate: int = 22050) -> tuple[bytes, float]:
    """Render the elimination buzzer to in-memory WAV bytes.

    Returns:
        (wav_bytes, duration_s).
    """
    raw = _synthesize_buzzer(sample_rate)
    duration_s = raw.size / sample_rate
    return _wav_bytes(_to_int16(raw), sample_rate), duration_s


def render_chant(path: str, sample_rate: int = 22050) -> float:
    """Write the chant to a WAV file at `path`.

    Deterministic: the same `chant.json` and `sample_rate` always produce
    byte-identical output, since nothing here reads the clock or a random
    source.

    Returns:
        Duration of the rendered audio, in seconds.
    """
    wav_bytes, duration_s = chant_wav_bytes(sample_rate)
    with open(path, "wb") as f:
        f.write(wav_bytes)
    return duration_s


def render_buzzer(path: str, sample_rate: int = 22050) -> float:
    """Write the elimination buzzer to a WAV file at `path`.

    Returns:
        Duration of the rendered audio, in seconds.
    """
    wav_bytes, duration_s = buzzer_wav_bytes(sample_rate)
    with open(path, "wb") as f:
        f.write(wav_bytes)
    return duration_s
