"""The recorded replay's light schedule agrees with the recorded match.

The page replays the seeded demo match (`docs/demo_match.mp4`) with a light,
a doll and a ring beside the video. The browser cannot draw that match's
schedule again from its seed, because its random number generator is not
Python's, so `site/arena.js` carries the schedule as data:
`REPLAY_LIGHT_STARTS_S`. These tests hold that data to the match it
describes — the published `demo_match` block — so an edit that would put the
sidebar on red while the footage is on green fails here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARENA_JS = REPO_ROOT / "site" / "arena.js"
RESULTS = REPO_ROOT / "reports" / "benchmark_results.json"


@pytest.fixture(scope="module")
def demo() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))["demo_match"]


@pytest.fixture(scope="module")
def slots(demo) -> list[tuple[str, float, float]]:
    """The schedule as `buildReplaySchedule` lays it out: green from 0, alternating."""
    match = re.search(
        r"const REPLAY_LIGHT_STARTS_S = Object\.freeze\(\[([^\]]*)\]\);",
        ARENA_JS.read_text(encoding="utf-8"),
    )
    assert match, "REPLAY_LIGHT_STARTS_S not found in site/arena.js"
    starts = [float(value) for value in match.group(1).split(",")]
    duration = demo["config"]["duration_s"]
    edges = [0.0, *[t for t in starts if t < duration], duration]
    return [
        ("GREEN" if i % 2 == 0 else "RED", edges[i], edges[i + 1])
        for i in range(len(edges) - 1)
    ]


def test_lights_start_in_order_inside_the_match(slots, demo):
    assert len(slots) >= 3
    for phase, start, end in slots:
        assert 0.0 <= start < end <= demo["config"]["duration_s"], (phase, start, end)


def test_every_full_light_is_as_long_as_the_match_allowed(slots, demo):
    """Each light but the last, which the final whistle cuts short, lasts 1.5-3.0 s."""
    low, high = demo["config"]["phase_min_s"], demo["config"]["phase_max_s"]
    for phase, start, end in slots[:-1]:
        assert low <= end - start <= high, (phase, start, end)


def test_every_movement_call_lands_on_an_armed_red_light(slots, demo):
    grace = demo["config"]["grace_s"]
    moved = [at for _, at, reason in demo["eliminations"] if reason == "moved"]
    assert moved, "the demo match should contain calls for movement"
    for at in moved:
        phase, start, _ = next(slot for slot in slots if slot[1] <= at < slot[2])
        assert phase == "RED", f"a call at {at} s lands on {phase}"
        assert at - start >= grace, f"a call at {at} s lands inside the grace window"
