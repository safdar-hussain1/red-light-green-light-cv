"""Optional sound: the chant during green light, a buzzer on elimination.

Nothing about the referee's calls depends on sound, so `Speaker` is built
to disappear cleanly when it can't work: no pygame installed, no audio
device, `muted=True` — every case collapses to the same silent no-op,
never an exception and never console noise. The `pygame` import happens
lazily inside `__init__`, not at module load, so importing `redlight.audio`
never triggers pygame's own startup banner; only constructing an unmuted
`Speaker` does.
"""

from __future__ import annotations

import io

from redlight import synth


class Speaker:
    """Plays the chant and buzzer through pygame's mixer, or does nothing.

    Attributes:
        muted: If True, playback is disabled without ever importing pygame.
    """

    def __init__(self, muted: bool = False):
        self._pygame = None
        self._enabled = False
        if muted:
            return

        try:
            import pygame
        except ImportError:
            return

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
        except Exception:
            return

        self._pygame = pygame
        self._enabled = True

    def _play(self, wav_bytes: bytes) -> None:
        if not self._enabled:
            return
        try:
            self._pygame.mixer.Sound(file=io.BytesIO(wav_bytes)).play()
        except Exception:
            pass

    def play_chant(self) -> None:
        """Play the countdown melody. A silent no-op if audio is unavailable."""
        if not self._enabled:
            return
        wav_bytes, _ = synth.chant_wav_bytes()
        self._play(wav_bytes)

    def play_buzzer(self) -> None:
        """Play the elimination buzzer. A silent no-op if audio is unavailable."""
        if not self._enabled:
            return
        wav_bytes, _ = synth.buzzer_wav_bytes()
        self._play(wav_bytes)
