"""Tests for the chant synthesiser and optional audio playback.

The chant itself lives in `chant.json` as data, not code, so the browser
build (Task 11) can bake the exact same melody into the site — one source
of truth for both renderers. These tests pin the properties that source of
truth has to hold: it decodes to a valid, audible WAV; rendering it twice
produces the same bytes; and playback degrades to a silent no-op whenever
pygame isn't available, so the referee never depends on sound to run.
"""

import sys
import wave

import numpy as np
import pytest

from redlight import synth


def _read_wav(path):
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    samples = np.frombuffer(raw, dtype=np.int16)
    return n_channels, sample_width, framerate, samples


class TestRenderChant:
    def test_produces_valid_audible_wav(self, tmp_path):
        """The rendered file is mono 16-bit PCM at the requested rate, over
        a second long, and actually has sound in it (not silence)."""
        path = tmp_path / "chant.wav"
        duration_s = synth.render_chant(str(path), sample_rate=22050)

        n_channels, sample_width, framerate, samples = _read_wav(path)
        assert n_channels == 1
        assert sample_width == 2
        assert framerate == 22050
        assert duration_s > 1.0
        assert samples.size / framerate == pytest.approx(duration_s, abs=1e-3)
        assert np.any(samples != 0)

    def test_is_deterministic(self, tmp_path):
        """Same inputs, same bytes — no timestamps, no randomness."""
        path_a = tmp_path / "a.wav"
        path_b = tmp_path / "b.wav"
        synth.render_chant(str(path_a))
        synth.render_chant(str(path_b))

        assert path_a.read_bytes() == path_b.read_bytes()

    def test_respects_sample_rate(self, tmp_path):
        path = tmp_path / "chant_11025.wav"
        synth.render_chant(str(path), sample_rate=11025)
        _, _, framerate, _ = _read_wav(path)
        assert framerate == 11025


class TestRenderBuzzer:
    def test_produces_valid_audible_wav(self, tmp_path):
        path = tmp_path / "buzzer.wav"
        duration_s = synth.render_buzzer(str(path))

        n_channels, sample_width, framerate, samples = _read_wav(path)
        assert n_channels == 1
        assert sample_width == 2
        assert duration_s > 0.0
        assert np.any(samples != 0)

    def test_is_deterministic(self, tmp_path):
        path_a = tmp_path / "a.wav"
        path_b = tmp_path / "b.wav"
        synth.render_buzzer(str(path_a))
        synth.render_buzzer(str(path_b))

        assert path_a.read_bytes() == path_b.read_bytes()


class TestLoadChant:
    def test_schema(self):
        chant = synth.load_chant()

        assert isinstance(chant, dict)
        assert isinstance(chant["bpm"], (int, float))
        assert chant["bpm"] > 0

        notes = chant["notes"]
        assert isinstance(notes, list)
        assert 10 <= len(notes) <= 14

        pentatonic_pitch_classes = {0, 2, 4, 7, 9}  # C D E G A
        for note in notes:
            assert len(note) == 2
            midi, beats = note
            assert isinstance(midi, int)
            assert midi % 12 in pentatonic_pitch_classes
            assert beats > 0

    def test_matches_packaged_chant_json(self):
        """Loaded data is exactly what's in the package-data file, not a copy."""
        import json
        from importlib import resources

        on_disk = json.loads(
            resources.files("redlight").joinpath("chant.json").read_text()
        )
        assert synth.load_chant() == on_disk


class TestSpeakerWithoutPygame:
    """pygame is an optional extra; the referee must run fine without it."""

    def _block_pygame_import(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pygame", None)

    def test_construction_does_not_raise(self, monkeypatch):
        self._block_pygame_import(monkeypatch)
        from redlight import audio

        speaker = audio.Speaker()
        assert speaker is not None

    def test_play_chant_is_silent_noop(self, monkeypatch, capsys):
        self._block_pygame_import(monkeypatch)
        from redlight import audio

        speaker = audio.Speaker()
        speaker.play_chant()

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_play_buzzer_is_silent_noop(self, monkeypatch, capsys):
        self._block_pygame_import(monkeypatch)
        from redlight import audio

        speaker = audio.Speaker()
        speaker.play_buzzer()

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


class TestSpeakerMuted:
    """muted=True must short-circuit before ever touching pygame."""

    def test_muted_speaker_never_imports_pygame(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pygame", None)
        from redlight import audio

        speaker = audio.Speaker(muted=True)
        speaker.play_chant()
        speaker.play_buzzer()
        # No exception is the assertion: a muted speaker must not even try
        # the (here poisoned) pygame import.
