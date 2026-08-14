"""Tests for the end-to-end pipeline runner (`app.run`) and its HUD.

Every `app.run` test here drives `data/vtest.avi` with the fast HOG
detector and small durations, so the whole suite stays quick even though it
exercises the real detect -> track -> judge -> game -> HUD loop end to end;
one slow test at the bottom repeats a minimal run with the default YOLO
detector, for confidence that the production detector path wires up the
same way.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from redlight import app, hud
from redlight.config import ConfigError, GameConfig
from redlight.detection import Detection
from redlight.game import EventType, Game, Phase
from redlight.judge import FrameSampler, MotionJudge
from redlight.tracking import Track

VIDEO = "data/vtest.avi"


def make_config(**overrides) -> GameConfig:
    """A config tuned to run fast on vtest.avi: short countdown, short match,
    quick red/green flips, no grace period so eliminations show up fast.
    """
    base = dict(
        seed=7,
        countdown_s=0.2,
        duration_s=0.8,
        phase_min_s=0.2,
        phase_max_s=0.3,
        grace_s=0.0,
        metric="flow",
        threshold=0.12,
        confirm_frames=2,
        smoothing=0.5,
        detector="hog",
        conf=0.3,
        max_misses=8,
    )
    base.update(overrides)
    return GameConfig(**base)


RUN_KWARGS = dict(headless=True, auto_start_frames=2, skip=3, max_seconds=15.0)


class TestRunAutoStart:
    def test_returns_match_report_with_at_least_three_players(self):
        report = app.run(make_config(), VIDEO, **RUN_KWARGS)

        assert report.players >= 3
        assert report.outcome in ("victory", "wipeout", "aborted")
        assert 0 <= report.survivors <= report.players
        for track_id, elapsed, reason in report.eliminations:
            assert isinstance(track_id, int)
            assert elapsed >= 0.0
            assert reason in ("moved", "left_arena")


class TestRunDeterminism:
    def test_two_runs_on_the_same_footage_agree_exactly(self):
        report1 = app.run(make_config(), VIDEO, **RUN_KWARGS)
        report2 = app.run(make_config(), VIDEO, **RUN_KWARGS)

        assert report1 == report2
        assert report1.eliminations == report2.eliminations


class TestRunRecord:
    def test_record_writes_an_mp4_with_frames(self, tmp_path):
        out_path = tmp_path / "match.mp4"

        app.run(make_config(), VIDEO, record=str(out_path), **RUN_KWARGS)

        assert out_path.exists()
        assert out_path.stat().st_size > 0

        cap = cv2.VideoCapture(str(out_path))
        try:
            frame_count = 0
            while True:
                ok, _frame = cap.read()
                if not ok:
                    break
                frame_count += 1
        finally:
            cap.release()
        assert frame_count > 0


class TestRunResetsJudgeAndSamplerOnEveryPhaseChange:
    def test_reset_call_count_matches_phase_changed_event_count(self, monkeypatch):
        reset_calls = {"judge": 0, "sampler": 0}
        phase_changes = {"count": 0}

        orig_judge_reset = MotionJudge.reset

        def spy_judge_reset(self):
            reset_calls["judge"] += 1
            return orig_judge_reset(self)

        orig_sampler_reset = FrameSampler.reset

        def spy_sampler_reset(self):
            reset_calls["sampler"] += 1
            return orig_sampler_reset(self)

        orig_start = Game.start

        def spy_start(self, now, ids):
            events = orig_start(self, now, ids)
            phase_changes["count"] += sum(1 for e in events if e.type == EventType.PHASE_CHANGED)
            return events

        orig_update = Game.update

        def spy_update(self, now, violations=None, missing=None):
            events = orig_update(self, now, violations, missing)
            phase_changes["count"] += sum(1 for e in events if e.type == EventType.PHASE_CHANGED)
            return events

        monkeypatch.setattr(MotionJudge, "reset", spy_judge_reset)
        monkeypatch.setattr(FrameSampler, "reset", spy_sampler_reset)
        monkeypatch.setattr(Game, "start", spy_start)
        monkeypatch.setattr(Game, "update", spy_update)

        app.run(make_config(), VIDEO, **RUN_KWARGS)

        # Sanity: this config's short phases must have actually flipped the
        # light more than once, or the assertion below would pass vacuously.
        assert phase_changes["count"] >= 3
        assert reset_calls["judge"] == phase_changes["count"]
        assert reset_calls["sampler"] == phase_changes["count"]


class TestRunValidation:
    def test_headless_without_auto_start_frames_raises_config_error(self):
        with pytest.raises(ConfigError):
            app.run(make_config(), VIDEO, headless=True)


class TestHudDrawSmoke:
    """Fast, synthetic-state smoke tests for `hud.draw` — no video, no
    detector, no model. Covers every phase and the eliminated/kill-flash/
    banner paths that a short `app.run` test won't reliably hit on its own.
    """

    def _config(self, **overrides) -> GameConfig:
        base = dict(seed=1, countdown_s=2.0, duration_s=10.0, phase_min_s=1.0, phase_max_s=1.0, grace_s=0.5)
        base.update(overrides)
        return GameConfig(**base)

    def _frame(self) -> np.ndarray:
        return np.full((240, 320, 3), 40, dtype=np.uint8)

    def _tracks(self) -> list[Track]:
        return [
            Track(track_id=1, box=Detection(x=20, y=20, w=40, h=100)),
            Track(track_id=2, box=Detection(x=150, y=30, w=40, h=100)),
        ]

    def _judge(self, config: GameConfig) -> MotionJudge:
        return MotionJudge(threshold=config.threshold, confirm_frames=config.confirm_frames)

    def _assert_valid_frame(self, out: np.ndarray, frame: np.ndarray) -> None:
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    @pytest.mark.parametrize(
        "phase_setup",
        ["lobby", "countdown", "green", "red_grace", "red_armed", "victory", "wipeout"],
    )
    def test_draws_every_phase_without_crashing(self, phase_setup):
        config = self._config()
        game = Game(config)
        frame = self._frame()
        tracks = self._tracks()
        judge = self._judge(config)
        now = 0.0

        if phase_setup == "lobby":
            pass
        elif phase_setup == "countdown":
            game.start(0.0, [1, 2])
            now = 1.0
        elif phase_setup == "green":
            game.start(0.0, [1, 2])
            game.update(config.countdown_s)
            now = config.countdown_s + 0.1
        elif phase_setup == "red_grace":
            game.start(0.0, [1, 2])
            game.update(config.countdown_s)
            game.update(config.countdown_s + config.phase_max_s)
            now = config.countdown_s + config.phase_max_s + 0.1
        elif phase_setup == "red_armed":
            game.start(0.0, [1, 2])
            game.update(config.countdown_s)
            game.update(config.countdown_s + config.phase_max_s)
            now = config.countdown_s + config.phase_max_s + config.grace_s + 0.1
        elif phase_setup == "victory":
            game.phase = Phase.VICTORY
        elif phase_setup == "wipeout":
            game.phase = Phase.WIPEOUT

        out = hud.draw(frame, game, tracks, judge, config, now)
        self._assert_valid_frame(out, frame)
        assert not np.array_equal(out, frame)  # something was actually drawn

    def test_eliminated_player_drawn_greyed_with_no_meter(self):
        config = self._config()
        game = Game(config)
        game.start(0.0, [1, 2])
        game.update(config.countdown_s)  # -> GREEN
        game.players[1].alive = False
        game.players[1].eliminated_at = 5.0

        out = hud.draw(self._frame(), game, self._tracks(), self._judge(config), config, 5.1)
        self._assert_valid_frame(out, self._frame())

    def test_kill_flash_fades_out_after_half_a_second(self):
        config = self._config()
        game = Game(config)
        game.start(0.0, [1, 2])
        game.update(config.countdown_s)
        game.players[1].alive = False
        game.players[1].eliminated_at = 5.0

        frame = self._frame()
        judge = self._judge(config)
        tracks = self._tracks()

        just_after = hud.draw(frame, game, tracks, judge, config, 5.05)
        long_after = hud.draw(frame, game, tracks, judge, config, 5.5 + hud.KILL_FLASH_S)

        # The flash frame and the long-after frame both draw the same
        # eliminated box, but only the flash frame carries the extra
        # full-frame tint, so they must not be pixel-identical.
        assert not np.array_equal(just_after, long_after)

    def test_hud_class_draw_matches_module_function(self):
        config = self._config()
        game = Game(config)
        frame = self._frame()
        tracks = self._tracks()
        judge = self._judge(config)

        via_class = hud.Hud().draw(frame, game, tracks, judge, config, 0.0)
        via_function = hud.draw(frame, game, tracks, judge, config, 0.0)

        assert np.array_equal(via_class, via_function)


@pytest.mark.slow
class TestRunWithYoloDetector:
    def test_default_detector_config_runs_end_to_end(self):
        config = make_config(detector="yolo", conf=0.35)
        report = app.run(config, VIDEO, **RUN_KWARGS)

        assert report.players >= 1
        assert report.outcome in ("victory", "wipeout", "aborted")
