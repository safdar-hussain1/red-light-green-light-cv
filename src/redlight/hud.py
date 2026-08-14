"""On-screen referee overlay: phase, players, and match outcome.

Every visual here is derived fresh from the state handed to `draw` on each
call — phase and countdown from `game`, boxes from `tracks`, motion levels
from `judge`. Even the kill flash needs nothing of its own:
`game.players[id].eliminated_at` already carries the timestamp an
elimination happened, so there is no separate clock to keep in step with
the referee's own state.

cv2's drawing calls ignore an image's alpha channel — filling a rectangle on
a BGRA frame paints the RGB and leaves alpha untouched, so drawing straight
onto the video frame one call at a time, each at some intended opacity, does
not composite the way it looks like it should. Every element here is
instead drawn once onto a floating-point BGR canvas and a matching alpha
canvas, and the whole overlay is alpha-blended onto the frame in a single
pass at the end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from redlight.config import GameConfig
from redlight.game import Game, Phase
from redlight.judge import MotionJudge
from redlight.tracking import Track

KILL_FLASH_S = 0.5
"""How long the kill flash stays visible after an elimination, in seconds."""

METER_HEADROOM = 1.5
"""Meter bars read full at this multiple of the active threshold, not at 1x,
so a confirmed violation doesn't look like it pinned the gauge."""

_FONT = cv2.FONT_HERSHEY_SIMPLEX

_PHASE_COLOR: dict[Phase, tuple[int, int, int]] = {
    Phase.LOBBY: (150, 150, 150),
    Phase.COUNTDOWN: (60, 200, 230),
    Phase.GREEN: (60, 200, 60),
    Phase.RED: (50, 50, 220),
    Phase.VICTORY: (0, 200, 255),
    Phase.WIPEOUT: (40, 40, 160),
}
_GREY = (130, 130, 130)
_WHITE = (235, 235, 235)


def draw(
    frame: np.ndarray,
    game: Game,
    tracks: list[Track],
    judge: MotionJudge,
    config: GameConfig,
    now: float,
) -> np.ndarray:
    """Render the referee's overlay on top of `frame`.

    Pure and stateless: the same `game`/`tracks`/`judge`/`config`/`now`
    always render the same output frame.

    Returns:
        A new BGR frame the same size as `frame`; the input is untouched.
    """
    h, w = frame.shape[:2]
    color = np.zeros((h, w, 3), dtype=np.float32)
    alpha = np.zeros((h, w), dtype=np.float32)

    _draw_kill_flash(color, alpha, game, now, w, h)
    _draw_players(color, alpha, game, tracks, judge, config, now)
    _draw_phase_pill(color, alpha, game, config, now, w)
    if game.finished:
        _draw_banner(color, alpha, game, w, h)

    return _blend(frame, color, alpha)


@dataclass
class Hud:
    """Callable wrapper around `draw`, for callers that want one HUD object
    per run rather than the bare function.

    Holds no per-frame state of its own: `draw` already reconstructs
    everything it shows — the kill flash included — from what it's handed
    each call, so there is nothing here that could fall out of sync with
    the match.
    """

    def draw(
        self,
        frame: np.ndarray,
        game: Game,
        tracks: list[Track],
        judge: MotionJudge,
        config: GameConfig,
        now: float,
    ) -> np.ndarray:
        return draw(frame, game, tracks, judge, config, now)


def _blend(frame: np.ndarray, color: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Alpha-composite the accumulated overlay onto `frame` in one pass."""
    a = alpha[..., None]
    out = frame.astype(np.float32) * (1.0 - a) + color * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _fill_rect(color, alpha, pt1, pt2, bgr, opacity) -> None:
    cv2.rectangle(color, pt1, pt2, bgr, -1)
    cv2.rectangle(alpha, pt1, pt2, opacity, -1)


def _outline_rect(color, alpha, pt1, pt2, bgr, opacity, thickness=2) -> None:
    cv2.rectangle(color, pt1, pt2, bgr, thickness)
    cv2.rectangle(alpha, pt1, pt2, opacity, thickness)


def _line(color, alpha, pt1, pt2, bgr, opacity, thickness=2) -> None:
    cv2.line(color, pt1, pt2, bgr, thickness, cv2.LINE_AA)
    cv2.line(alpha, pt1, pt2, opacity, thickness, cv2.LINE_AA)


def _text(color, alpha, text, org, bgr, opacity, scale=0.6, thickness=1) -> None:
    cv2.putText(color, text, org, _FONT, scale, bgr, thickness, cv2.LINE_AA)
    cv2.putText(alpha, text, org, _FONT, scale, opacity, thickness, cv2.LINE_AA)


def _ellipse(color, alpha, center, axes, rot, start, end, bgr, opacity, thickness=3) -> None:
    cv2.ellipse(color, center, axes, rot, start, end, bgr, thickness, cv2.LINE_AA)
    cv2.ellipse(alpha, center, axes, rot, start, end, opacity, thickness, cv2.LINE_AA)


def _text_size(text: str, scale: float = 0.6, thickness: int = 1) -> tuple[int, int]:
    (w, h), _baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    return w, h


def _meter_color(ratio: float) -> tuple[int, int, int]:
    """BGR color for a motion ratio: green when calm, red at/above threshold."""
    t = min(max(ratio, 0.0), 1.0)
    g = int(210 * (1 - t) + 60 * t)
    r = int(60 * (1 - t) + 220 * t)
    return (60, g, r)


def _draw_kill_flash(color, alpha, game: Game, now: float, w: int, h: int) -> None:
    peak = 0.0
    for player in game.players.values():
        if player.eliminated_at is None:
            continue
        age = now - player.eliminated_at
        if 0.0 <= age < KILL_FLASH_S:
            peak = max(peak, 0.45 * (1.0 - age / KILL_FLASH_S))
    if peak <= 0.0:
        return
    _fill_rect(color, alpha, (0, 0), (w - 1, h - 1), (40, 40, 220), peak)


def _draw_players(
    color,
    alpha,
    game: Game,
    tracks: list[Track],
    judge: MotionJudge,
    config: GameConfig,
    now: float,
) -> None:
    threshold = config.threshold if config.metric == "flow" else config.diff_threshold
    armed = game.armed(now)

    for track in tracks:
        box = track.box
        pt1 = (int(box.x), int(box.y))
        pt2 = (int(box.x + box.w), int(box.y + box.h))
        player = game.players.get(track.track_id)

        if player is not None and not player.alive:
            _outline_rect(color, alpha, pt1, pt2, _GREY, 0.85, thickness=2)
            _line(color, alpha, pt1, pt2, _GREY, 0.85, thickness=2)
            _line(color, alpha, (pt1[0], pt2[1]), (pt2[0], pt1[1]), _GREY, 0.85, thickness=2)
            _text(color, alpha, f"P{track.track_id} OUT", (pt1[0], max(pt1[1] - 8, 12)), _GREY, 1.0)
            continue

        ratio = 0.0
        if player is not None:
            ratio = judge.smoothed(track.track_id) / threshold if threshold > 0 else 0.0
            box_bgr = _meter_color(ratio) if armed else _WHITE
        else:
            box_bgr = (200, 200, 200)

        _outline_rect(color, alpha, pt1, pt2, box_bgr, 1.0, thickness=2)
        _text(color, alpha, f"P{track.track_id}", (pt1[0], max(pt1[1] - 8, 12)), box_bgr, 1.0)

        if player is not None:
            _draw_meter(color, alpha, pt1, pt2, ratio)


def _draw_meter(color, alpha, pt1, pt2, ratio: float) -> None:
    bar_x1, bar_y1 = pt1[0], pt2[1] + 6
    bar_x2 = pt2[0]
    bar_y2 = bar_y1 + 8
    if bar_x2 <= bar_x1:
        return

    _fill_rect(color, alpha, (bar_x1, bar_y1), (bar_x2, bar_y2), (60, 60, 60), 0.6)
    frac = min(max(ratio, 0.0), METER_HEADROOM) / METER_HEADROOM
    fill_x2 = bar_x1 + int((bar_x2 - bar_x1) * frac)
    if fill_x2 > bar_x1:
        _fill_rect(color, alpha, (bar_x1, bar_y1), (fill_x2, bar_y2), _meter_color(ratio), 0.9)


def _phase_label(game: Game, config: GameConfig, now: float) -> str:
    if game.phase == Phase.LOBBY:
        return "LOBBY - PRESS S TO START"
    if game.phase == Phase.COUNTDOWN:
        return f"COUNTDOWN {math.ceil(game.countdown_left(now))}"
    if game.phase == Phase.GREEN:
        return "GREEN LIGHT"
    if game.phase == Phase.RED:
        grace_left = config.grace_s - game.phase_elapsed(now)
        if grace_left > 0:
            return f"RED LIGHT - GRACE {grace_left:.1f}s"
        return "RED LIGHT"
    if game.phase == Phase.VICTORY:
        return "VICTORY"
    return "WIPEOUT"


def _draw_phase_pill(color, alpha, game: Game, config: GameConfig, now: float, w: int) -> None:
    label = _phase_label(game, config, now)
    bgr = _PHASE_COLOR[game.phase]
    text_w, text_h = _text_size(label, scale=0.8, thickness=2)
    pad_x, pad_y = 24, 14
    pill_w = text_w + pad_x * 2
    pill_h = text_h + pad_y * 2
    x1 = w // 2 - pill_w // 2
    y1 = 12
    x2 = x1 + pill_w
    y2 = y1 + pill_h

    _fill_rect(color, alpha, (x1, y1), (x2, y2), (25, 25, 25), 0.55)
    _outline_rect(color, alpha, (x1, y1), (x2, y2), bgr, 1.0, thickness=2)
    _text(color, alpha, label, (x1 + pad_x, y1 + pad_y + text_h), bgr, 1.0, scale=0.8, thickness=2)

    if game.phase == Phase.COUNTDOWN and config.countdown_s > 0:
        frac = game.countdown_left(now) / config.countdown_s
        center = (x2 + 26, (y1 + y2) // 2)
        _ellipse(color, alpha, center, (16, 16), -90, 0, 360, (70, 70, 70), 1.0, thickness=4)
        _ellipse(color, alpha, center, (16, 16), -90, 0, 360.0 * frac, bgr, 1.0, thickness=4)


def _draw_banner(color, alpha, game: Game, w: int, h: int) -> None:
    victory = game.phase == Phase.VICTORY
    label = "VICTORY" if victory else "WIPEOUT"
    survivors = game.alive_count
    sub = f"{survivors} SURVIVOR{'S' if survivors != 1 else ''}" if victory else "NO SURVIVORS"
    bgr = _PHASE_COLOR[game.phase]

    band_h = 120
    y1 = h // 2 - band_h // 2
    y2 = y1 + band_h
    _fill_rect(color, alpha, (0, y1), (w, y2), (15, 15, 15), 0.75)

    label_w, label_h = _text_size(label, scale=1.6, thickness=4)
    _text(color, alpha, label, (w // 2 - label_w // 2, y1 + 20 + label_h), bgr, 1.0, scale=1.6, thickness=4)

    sub_w, _sub_h = _text_size(sub, scale=0.8, thickness=2)
    _text(color, alpha, sub, (w // 2 - sub_w // 2, y2 - 20), _WHITE, 1.0, scale=0.8, thickness=2)
