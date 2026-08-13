"""Game state machine: phases, timing, and the elimination rules.

The referee runs a fixed loop of light phases. Registration happens once,
at the start of the countdown: only players who signed up before the
countdown began can ever be eliminated or win, so anyone the camera picks
up afterward is not part of the match. During red light, a short grace
window right after the light turns forgives the motion needed to actually
come to a stop — a player already mid-stride when the light changes is not
punished for a reaction time nobody has zero of. Once that window closes
the phase is armed, and any confirmed motion counts. Losing the tracker's
lock on a registered player — walking out of frame, staying occluded too
long — eliminates them under either light, on the same footing as anyone
else who can no longer be seen well enough to judge.

This module is pure logic: no video, no wall clock, no sleeping. Every
method that cares about time is handed `now` by the caller as a plain
float, and the light schedule is drawn from a single `random.Random`
seeded once at construction, so the whole match is reproducible from that
seed and a sequence of `now` values.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto

from redlight.config import GameConfig


class Phase(Enum):
    """Where the match stands right now."""

    LOBBY = auto()
    COUNTDOWN = auto()
    GREEN = auto()
    RED = auto()
    VICTORY = auto()
    WIPEOUT = auto()


class EventType(Enum):
    """What kind of thing just happened."""

    PHASE_CHANGED = auto()
    PLAYER_ELIMINATED = auto()
    GAME_OVER = auto()


@dataclass(frozen=True)
class Event:
    """One thing that happened during `update`, for a caller to react to.

    `track_id` and `reason` are only ever set on PLAYER_ELIMINATED events.
    `reason` is `"moved"` — caught by the judge while the current red light
    was armed — or `"left_arena"` — the tracker lost the player entirely.
    """

    type: EventType
    phase: Phase
    track_id: int | None = None
    reason: str | None = None


@dataclass
class Player:
    """One registered player: their id, whether they are still in, and when they went out."""

    track_id: int
    alive: bool = True
    eliminated_at: float | None = None


class Game:
    """Runs the light schedule and applies the elimination rules.

    Nothing here reads a clock or a camera; every method that needs the
    current time takes `now` as an argument. The phase schedule comes from
    one `random.Random(config.seed)` built in `__init__` and consumed one
    draw at a time, on every light change — so two games built with the
    same seed and driven with the same sequence of `now` values produce
    the identical schedule of green/red lengths.
    """

    def __init__(self, config: GameConfig):
        self.config = config
        self.phase = Phase.LOBBY
        self.players: dict[int, Player] = {}
        self._rng = random.Random(config.seed)
        self._phase_started_at: float | None = None
        self._phase_end_at: float | None = None
        self._countdown_end_at: float | None = None
        self._match_started_at: float | None = None

    def start(self, now: float, player_track_ids: Iterable[int]) -> list[Event]:
        """Register players and open the countdown.

        Only the ids passed here are ever eligible to be eliminated or to
        win the match. Can only be called while the game is in LOBBY, and
        needs at least one player to register.

        Raises:
            RuntimeError: If the game is not in LOBBY (already started).
            ValueError: If `player_track_ids` is empty.
        """
        if self.phase != Phase.LOBBY:
            raise RuntimeError(
                f"start() requires phase LOBBY, game is in {self.phase.name}"
            )
        ids = list(player_track_ids)
        if not ids:
            raise ValueError("start() requires at least one player_track_id")

        self.players = {track_id: Player(track_id=track_id) for track_id in ids}
        self.phase = Phase.COUNTDOWN
        self._phase_started_at = now
        self._countdown_end_at = now + self.config.countdown_s
        return [Event(EventType.PHASE_CHANGED, self.phase)]

    def update(
        self,
        now: float,
        violations: Iterable[int] | None = None,
        missing: Iterable[int] | None = None,
    ) -> list[Event]:
        """Advance the match to `now`, applying eliminations and light changes.

        `violations` are the ids of players the judge has confirmed are
        moving on this tick; they only cost a player their spot while the
        current red light is armed (see `armed`) — during green, or in
        red's grace window, motion is forgiven. `missing` are registered
        players whose track has died; they are eliminated under either
        light, since a player the camera can no longer see cannot be
        judged fairly. Ids that are not registered, or already
        eliminated, are silently ignored in both lists.

        Before the countdown finishes, or after the match has ended, this
        is a no-op that returns no events.

        Returns:
            The events produced by this tick: eliminations first, then a
            match outcome if this tick ended the match, then a light
            change if the match is still running.
        """
        if self.phase == Phase.LOBBY or self.finished:
            return []
        if self.phase == Phase.COUNTDOWN:
            return self._update_countdown(now)
        return self._update_match(now, violations or (), missing or ())

    def _update_countdown(self, now: float) -> list[Event]:
        if now < self._countdown_end_at:
            return []
        self.phase = Phase.GREEN
        self._match_started_at = self._countdown_end_at
        self._phase_started_at = self._countdown_end_at
        self._schedule_next_phase_end()
        return [Event(EventType.PHASE_CHANGED, self.phase)]

    def _update_match(
        self, now: float, violations: Iterable[int], missing: Iterable[int]
    ) -> list[Event]:
        events = self._apply_eliminations(now, violations, missing)

        if self.alive_count == 0:
            events.extend(self._end_match(Phase.WIPEOUT))
            return events

        if now - self._match_started_at >= self.config.duration_s:
            events.extend(self._end_match(Phase.VICTORY))
            return events

        events.extend(self._flip_phase(now))
        return events

    def _apply_eliminations(
        self, now: float, violations: Iterable[int], missing: Iterable[int]
    ) -> list[Event]:
        events = []
        if self.armed(now):
            for track_id in violations:
                events.extend(self._eliminate(track_id, now, "moved"))
        for track_id in missing:
            events.extend(self._eliminate(track_id, now, "left_arena"))
        return events

    def _eliminate(self, track_id: int, now: float, reason: str) -> list[Event]:
        player = self.players.get(track_id)
        if player is None or not player.alive:
            return []
        player.alive = False
        player.eliminated_at = now
        return [Event(EventType.PLAYER_ELIMINATED, self.phase, track_id, reason)]

    def _flip_phase(self, now: float) -> list[Event]:
        events = []
        while now >= self._phase_end_at:
            self.phase = Phase.RED if self.phase == Phase.GREEN else Phase.GREEN
            self._phase_started_at = self._phase_end_at
            self._schedule_next_phase_end()
            events.append(Event(EventType.PHASE_CHANGED, self.phase))
        return events

    def _schedule_next_phase_end(self) -> None:
        length = self._rng.uniform(self.config.phase_min_s, self.config.phase_max_s)
        self._phase_end_at = self._phase_started_at + length

    def _end_match(self, phase: Phase) -> list[Event]:
        self.phase = phase
        return [
            Event(EventType.PHASE_CHANGED, phase),
            Event(EventType.GAME_OVER, phase),
        ]

    def armed(self, now: float) -> bool:
        """Whether a red light currently counts motion as an elimination.

        The light itself switches instantly, but a player mid-stride when
        it happens needs a moment to stop; for `grace_s` after the light
        turns red, motion is forgiven. After that the phase is armed, and
        any confirmed motion is grounds for elimination.
        """
        return self.phase == Phase.RED and self.phase_elapsed(now) >= self.config.grace_s

    def time_left(self, now: float) -> float:
        """Seconds remaining on the match clock.

        Outside green/red — before the match has started, or after it has
        ended — this is just the configured `duration_s`, since no match
        clock is running to count down from it.
        """
        if not self.in_match:
            return self.config.duration_s
        return max(self.config.duration_s - (now - self._match_started_at), 0.0)

    def countdown_left(self, now: float) -> float:
        """Seconds remaining before the countdown ends, or 0 outside it."""
        if self.phase != Phase.COUNTDOWN:
            return 0.0
        return max(self._countdown_end_at - now, 0.0)

    def phase_elapsed(self, now: float) -> float:
        """Seconds since the current phase began, or 0 before a match has started."""
        if self._phase_started_at is None:
            return 0.0
        return now - self._phase_started_at

    @property
    def alive_count(self) -> int:
        """How many registered players have not been eliminated."""
        return sum(1 for player in self.players.values() if player.alive)

    @property
    def in_match(self) -> bool:
        """Whether a light is currently live (green or red)."""
        return self.phase in (Phase.GREEN, Phase.RED)

    @property
    def finished(self) -> bool:
        """Whether the match has reached a final outcome (victory or wipeout)."""
        return self.phase in (Phase.VICTORY, Phase.WIPEOUT)
