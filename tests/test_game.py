"""Tests for the game state machine.

Every test drives `Game` with a plain float clock — no `time.sleep`, no
wall-clock reads. That is what lets the grace window and the seeded phase
schedule be pinned down exactly: a test can ask "what happens at
grace_s - 0.1" and "what happens at grace_s + 0.1" and get a real answer
instead of a flaky one.
"""

import random

import pytest

from redlight.config import GameConfig
from redlight.game import Event, EventType, Game, Phase


def make_config(**overrides) -> GameConfig:
    base = dict(
        seed=42,
        countdown_s=2.0,
        duration_s=10.0,
        phase_min_s=1.0,
        phase_max_s=1.0,
        grace_s=0.5,
    )
    base.update(overrides)
    return GameConfig(**base)


def test_full_happy_path_to_victory():
    """Countdown, into green, one light flip, then the match clock runs out survivors win."""
    cfg = make_config(countdown_s=2.0, duration_s=4.0, phase_min_s=1.0, phase_max_s=1.0)
    game = Game(cfg)

    events = game.start(0.0, [1, 2, 3])
    assert events == [Event(EventType.PHASE_CHANGED, Phase.COUNTDOWN)]
    assert game.phase == Phase.COUNTDOWN
    assert game.countdown_left(0.0) == pytest.approx(2.0)
    assert game.countdown_left(1.5) == pytest.approx(0.5)
    assert game.time_left(1.0) == cfg.duration_s  # no match clock running yet

    # Still counting down: no event.
    assert game.update(1.0) == []
    assert game.phase == Phase.COUNTDOWN

    # Countdown elapses -> GREEN, match clock starts.
    events = game.update(2.0)
    assert events == [Event(EventType.PHASE_CHANGED, Phase.GREEN)]
    assert game.phase == Phase.GREEN
    assert game.in_match
    assert game.countdown_left(2.0) == 0.0
    assert game.phase_elapsed(2.0) == 0.0
    assert game.time_left(2.0) == pytest.approx(4.0)
    assert game.time_left(3.0) == pytest.approx(3.0)

    # Green phase (length 1.0) ends at t=3.0 -> flips to RED.
    events = game.update(3.0)
    assert events == [Event(EventType.PHASE_CHANGED, Phase.RED)]
    assert game.phase == Phase.RED
    assert game.phase_elapsed(3.0) == 0.0

    # Match clock (duration_s=4.0, started at t=2.0) expires at t=6.0.
    events = game.update(6.0)
    assert events == [
        Event(EventType.PHASE_CHANGED, Phase.VICTORY),
        Event(EventType.GAME_OVER, Phase.VICTORY),
    ]
    assert game.phase == Phase.VICTORY
    assert game.finished
    assert not game.in_match
    assert game.alive_count == 3
    assert game.time_left(999.0) == cfg.duration_s  # sentinel once match is over

    # A finished match ignores further updates.
    assert game.update(7.0, violations=[1]) == []
    assert game.players[1].alive is True


def test_wipeout_when_all_players_eliminated():
    """Every player caught moving while armed ends the match in a wipeout."""
    cfg = make_config(countdown_s=1.0, duration_s=100.0, phase_min_s=1.0, phase_max_s=1.0, grace_s=0.0)
    game = Game(cfg)
    game.start(0.0, [1, 2])

    game.update(1.0)  # -> GREEN at t=1.0
    events = game.update(2.0)  # green (len 1.0) ends at t=2.0 -> RED
    assert events == [Event(EventType.PHASE_CHANGED, Phase.RED)]
    assert game.armed(2.0)  # grace_s == 0.0, armed the instant red starts

    events = game.update(2.0, violations=[1, 2])
    assert events == [
        Event(EventType.PLAYER_ELIMINATED, Phase.RED, 1, "moved"),
        Event(EventType.PLAYER_ELIMINATED, Phase.RED, 2, "moved"),
        Event(EventType.PHASE_CHANGED, Phase.WIPEOUT),
        Event(EventType.GAME_OVER, Phase.WIPEOUT),
    ]
    assert game.phase == Phase.WIPEOUT
    assert game.finished
    assert game.alive_count == 0
    assert game.players[1].alive is False
    assert game.players[1].eliminated_at == 2.0


def test_start_only_allowed_from_lobby():
    """The countdown gate: a match already under way cannot be re-registered."""
    game = Game(make_config())
    game.start(0.0, [1])

    with pytest.raises(RuntimeError):
        game.start(1.0, [1, 2])


def test_start_requires_at_least_one_player():
    game = Game(make_config())
    with pytest.raises(ValueError):
        game.start(0.0, [])


def test_violations_ignored_unless_armed_grace_window():
    """Motion in red's grace window is forgiven; the same motion after it is not."""
    cfg = make_config(countdown_s=1.0, duration_s=100.0, phase_min_s=1.0, phase_max_s=1.0, grace_s=0.5)
    game = Game(cfg)
    game.start(0.0, [1, 2])  # a second player keeps the match alive past the elimination
    game.update(1.0)  # -> GREEN at t=1.0
    game.update(2.0)  # green ends at t=2.0 -> RED, phase_started_at=2.0

    just_before = 2.0 + 0.4  # grace_s - 0.1
    assert not game.armed(just_before)
    events = game.update(just_before, violations=[1])
    assert events == []
    assert game.players[1].alive is True

    just_after = 2.0 + 0.6  # grace_s + 0.1
    assert game.armed(just_after)
    events = game.update(just_after, violations=[1])
    assert events == [Event(EventType.PLAYER_ELIMINATED, Phase.RED, 1, "moved")]
    assert game.players[1].alive is False


def test_violations_ignored_during_green():
    """Motion never eliminates while the light is green, regardless of arming."""
    cfg = make_config(countdown_s=1.0, duration_s=100.0)
    game = Game(cfg)
    game.start(0.0, [1])
    game.update(1.0)  # -> GREEN

    assert not game.armed(1.5)
    events = game.update(1.5, violations=[1])
    assert events == []
    assert game.players[1].alive is True


def test_unregistered_and_already_eliminated_ids_are_ignored():
    cfg = make_config(countdown_s=1.0, duration_s=100.0, phase_min_s=1.0, phase_max_s=1.0, grace_s=0.0)
    game = Game(cfg)
    game.start(0.0, [1, 2])  # a second player keeps the match alive past the elimination
    game.update(1.0)  # -> GREEN
    game.update(2.0)  # -> RED, armed immediately (grace_s=0.0)

    # Unregistered id: silently ignored, no event, no error.
    assert game.update(2.0, violations=[999]) == []

    # Registered id: eliminated exactly once.
    events = game.update(2.0, violations=[1])
    assert events == [Event(EventType.PLAYER_ELIMINATED, Phase.RED, 1, "moved")]

    # Already-eliminated id: silently ignored on a later tick.
    assert game.update(2.0, violations=[1]) == []
    assert game.update(2.0, missing=[1]) == []


def test_missing_player_eliminated_even_on_green():
    """A dead track eliminates its player under either light, unlike a motion violation."""
    cfg = make_config(countdown_s=1.0, duration_s=100.0)
    game = Game(cfg)
    game.start(0.0, [1, 2])
    game.update(1.0)  # -> GREEN
    assert game.phase == Phase.GREEN

    events = game.update(1.2, missing=[1])
    assert events == [Event(EventType.PLAYER_ELIMINATED, Phase.GREEN, 1, "left_arena")]
    assert game.players[1].alive is False
    assert game.players[1].eliminated_at == 1.2
    assert game.players[2].alive is True
    assert game.alive_count == 1

    # Unregistered/dead ids in `missing` are ignored too.
    assert game.update(1.3, missing=[1, 999]) == []


def test_seeded_schedule_is_reproducible_and_seed_dependent():
    """Same seed -> identical sequence of phase-flip times; a different seed diverges."""

    def flip_times(seed: int, flips: int = 3) -> list[float]:
        cfg = make_config(seed=seed, countdown_s=1.0, phase_min_s=1.0, phase_max_s=3.0, duration_s=10_000.0)
        game = Game(cfg)
        game.start(0.0, [1])
        game.update(1.0)  # -> GREEN; consumes the schedule's first draw

        rng = random.Random(seed)
        lengths = [rng.uniform(1.0, 3.0) for _ in range(flips)]

        times = []
        elapsed = 0.0
        for length in lengths:
            elapsed += length
            now = 1.0 + elapsed
            events = game.update(now)
            assert events and events[0].type == EventType.PHASE_CHANGED, (seed, now, events)
            times.append(now)
        return times

    same_a = flip_times(42)
    same_b = flip_times(42)
    different = flip_times(7)

    assert same_a == same_b
    assert same_a != different


def test_events_are_well_formed():
    cfg = make_config(countdown_s=1.0, duration_s=100.0, phase_min_s=1.0, phase_max_s=1.0, grace_s=0.0)
    game = Game(cfg)

    all_events = list(game.start(0.0, [1]))
    all_events += game.update(1.0)  # -> GREEN
    all_events += game.update(2.0)  # -> RED
    all_events += game.update(2.0, violations=[1])  # -> eliminated, wipeout, game over

    assert all_events, "expected at least one event across the sequence"
    for event in all_events:
        assert isinstance(event, Event)
        assert isinstance(event.type, EventType)
        assert isinstance(event.phase, Phase)
        assert event.track_id is None or isinstance(event.track_id, int)
        assert event.reason is None or event.reason in ("moved", "left_arena")
        if event.type != EventType.PLAYER_ELIMINATED:
            assert event.track_id is None
            assert event.reason is None

    # Events are frozen: no mutating a record after the fact.
    with pytest.raises(Exception):
        all_events[0].phase = Phase.WIPEOUT
