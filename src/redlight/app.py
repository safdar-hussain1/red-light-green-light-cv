"""End-to-end pipeline: read frames, detect and track players, judge motion,
run the game clock, and draw the HUD — wired together the way `redlight
play` runs a match.

One `FrameSampler` paces motion scoring for the whole run on a fixed real-time
clock, independent of the light schedule. Every tick still detects and tracks
(the tracker needs to see every frame to keep ids stable across misses), but
a pair is only scored — and only fed to the judge — when the sampler
releases one and the current red light is armed; during green light and
red's grace window, players keep being tracked but are simply not judged.
Both the judge and the sampler are reset on every phase change, so no
baseline survives from one light into the next.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from redlight.audio import Speaker
from redlight.config import ConfigError, GameConfig
from redlight.detection import Detection, make_detector
from redlight.game import EventType, Game, Phase
from redlight.hud import Hud
from redlight.judge import FrameSampler, MotionJudge, crop_window, diff_score_window, flow_score
from redlight.sources import FrameSource
from redlight.tracking import Tracker

_QUIT_KEYS = (ord("q"), ord("Q"))
_START_KEYS = (ord("s"), ord("S"))
_WINDOW_NAME = "Red Light, Green Light"


@dataclass(frozen=True)
class MatchReport:
    """How a run of the referee ended.

    `outcome` is `"victory"` (the clock ran out with survivors left),
    `"wipeout"` (every registered player was eliminated), or `"aborted"`
    (the run stopped — source exhausted, `max_seconds`, or a quit keypress —
    before the game reached either). `eliminations` are in the order they
    happened, as `(track_id, match-relative seconds, reason)`; `reason`
    matches `game.Event.reason`: `"moved"` or `"left_arena"`.
    """

    outcome: str
    players: int
    survivors: int
    eliminations: list[tuple[int, float, str]]


def run(
    config: GameConfig,
    source: str | int,
    *,
    headless: bool = False,
    record: str | None = None,
    auto_start_frames: int | None = None,
    skip: int = 0,
    max_seconds: float | None = None,
) -> MatchReport:
    """Run one match end to end and report how it finished.

    `source` is anything `sources.FrameSource` accepts (a video path or a
    camera index); this function owns opening and closing it. `skip`
    discards that many frames before the loop starts, so a warm-up section
    of footage can be skipped without shifting the match clock. Timestamps
    come straight from the source, so a file run is fully deterministic:
    given the same config, source, and `skip`/`auto_start_frames`, two runs
    produce the same eliminations.

    `auto_start_frames` registers whichever tracks are live once that many
    consecutive frames have had at least one — for unattended runs (a demo,
    a benchmark) where nobody is at the keyboard to start the match. In
    interactive mode (`headless=False`), a window is also shown and the
    match can be started with the 'S' key, or the run abandoned with 'Q'
    (outcome `"aborted"`). A headless run given no `auto_start_frames` has no
    way to ever start a match, so it is rejected up front rather than
    hanging forever in LOBBY.

    `record`, if given, writes every HUD-annotated frame to that path as an
    mp4 (via OpenCV's `mp4v` writer) — this works in headless mode too, so a
    match can be captured without a display attached. `max_seconds` is a
    safety stop, measured on the source's own timestamps: if the match
    hasn't finished by then, the run stops and reports `"aborted"`.

    `config` is used as given — call `config.validate()` yourself first if
    you want bad settings rejected with a readable message; this function
    does not do it for you, matching every other module here (`Game`,
    `Tracker`, ... all take a `GameConfig` on trust too).

    Raises:
        ConfigError: If `headless` is set with no `auto_start_frames`.
    """
    if headless and auto_start_frames is None:
        raise ConfigError(
            "headless runs need auto_start_frames set - with no window and "
            "no auto-start, the match could never be started"
        )

    # OpenCV's default thread pool reduces across worker threads in whatever
    # order they finish, which is fine for a single detection but leaves
    # HOG/Farneback results a sub-pixel-scale coin flip between runs. That
    # is nothing on its own, but it is occasionally enough to flip which of
    # two near-identically-stale tracks a match considers lost first —
    # which breaks the promise that a fixed seed on the same footage always
    # plays out the same match. Pinning to one thread makes every run bit
    # for bit identical, at the cost of not using extra cores.
    cv2.setNumThreads(1)

    detector = make_detector(config.detector, config.conf)
    tracker = Tracker(max_misses=config.max_misses)
    threshold = config.threshold if config.metric == "flow" else config.diff_threshold
    judge = MotionJudge(threshold=threshold, confirm_frames=config.confirm_frames, smoothing=config.smoothing)
    sampler = FrameSampler()
    game = Game(config)
    speaker = Speaker(muted=headless)
    hud = Hud()

    eliminations: list[tuple[int, float, str]] = []
    match_start_ts: float | None = None
    stable_frames = 0
    writer: cv2.VideoWriter | None = None
    loop_start_ts: float | None = None
    outcome = "aborted"

    def handle_events(events) -> None:
        for event in events:
            if event.type == EventType.PHASE_CHANGED:
                judge.reset()
                sampler.reset()
                if event.phase == Phase.GREEN:
                    speaker.play_chant()
            elif event.type == EventType.PLAYER_ELIMINATED:
                speaker.play_buzzer()
                base = match_start_ts if match_start_ts is not None else ts
                eliminations.append((event.track_id, ts - base, event.reason))

    try:
        with FrameSource(source) as src:
            for _ in range(skip):
                if src.read() is None:
                    break

            while True:
                result = src.read()
                if result is None:
                    break
                frame, ts = result
                if loop_start_ts is None:
                    loop_start_ts = ts
                if max_seconds is not None and (ts - loop_start_ts) > max_seconds:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = detector.detect(frame)
                tracks = tracker.update(detections)
                dead_ids = tracker.pop_dead()

                if game.phase == Phase.LOBBY and auto_start_frames is not None:
                    stable_frames = stable_frames + 1 if tracks else 0
                    if stable_frames >= auto_start_frames:
                        match_start_ts = ts + config.countdown_s
                        handle_events(game.start(ts, [t.track_id for t in tracks]))

                sample = sampler.offer(gray, ts)
                violations: list[int] = []
                if sample is not None and game.armed(ts):
                    prev_gray, cur_gray, dt = sample
                    for track in tracks:
                        player = game.players.get(track.track_id)
                        if player is None or not player.alive:
                            continue
                        score = _score(config.metric, prev_gray, cur_gray, track.box, dt)
                        if score is None:
                            continue
                        if judge.update(track.track_id, score):
                            violations.append(track.track_id)

                missing = [
                    track_id
                    for track_id in dead_ids
                    if track_id in game.players and game.players[track_id].alive
                ]

                handle_events(game.update(ts, violations=violations, missing=missing))

                if record is not None or not headless:
                    hud_frame = hud.draw(frame, game, tracks, judge, config, ts)
                    if record is not None:
                        if writer is None:
                            h, w = hud_frame.shape[:2]
                            fps = src.fps or 30.0
                            writer = cv2.VideoWriter(record, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        writer.write(hud_frame)

                    if not headless:
                        cv2.imshow(_WINDOW_NAME, hud_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in _QUIT_KEYS:
                            outcome = "aborted"
                            break
                        if game.phase == Phase.LOBBY and key in _START_KEYS and tracks:
                            match_start_ts = ts + config.countdown_s
                            handle_events(game.start(ts, [t.track_id for t in tracks]))

                if game.finished:
                    break
    finally:
        if writer is not None:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()

    if game.finished:
        outcome = "victory" if game.phase == Phase.VICTORY else "wipeout"

    return MatchReport(
        outcome=outcome,
        players=len(game.players),
        survivors=game.alive_count,
        eliminations=eliminations,
    )


def _score(
    metric: str, prev_gray: np.ndarray, cur_gray: np.ndarray, box: Detection, dt: float
) -> float | None:
    """One player's motion score for the configured metric, on one sampled pair."""
    if metric == "flow":
        return flow_score(prev_gray, cur_gray, box, dt)
    prev_win = crop_window(prev_gray, box)
    cur_win = crop_window(cur_gray, box)
    if prev_win is None or cur_win is None:
        return None
    return diff_score_window(prev_win, cur_win, dt)
