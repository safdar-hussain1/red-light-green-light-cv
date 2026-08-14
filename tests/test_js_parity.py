"""The browser judge and the Python judge must agree, and be proven to.

The arena page ships its own JavaScript referee so a visitor can play in a
browser with no install. That only means anything if it scores the same way:
two 96x96 windows and a dt have to produce the same number on both sides,
and the same stream of scores has to produce the same eliminations.

The boundary of the claim matters. What is pinned here is the scoring
kernel — `diff_score_window`, `MotionJudge`, `FrameSampler` — and the game
rules. What is *not* pinned is how a camera frame becomes a 96x96 window:
`judge.crop_window` uses OpenCV's INTER_AREA resize and a 3x3 Gaussian
blur, and the browser gets there through canvas. So these tests prove
identical scoring of identical windows, not identical verdicts from an
identical frame, and no copy anywhere should claim the latter.

These tests hold the port to that claim rather than trusting it. Golden
fixtures — real 96x96 uint8 window pairs, with the Python score recorded at
full float precision — are replayed through `site/judge.js` under node, and
the two sets of numbers are required to be *bit-identical*, not merely
close. That is a fair demand here: the score is an integer count divided by
9216 and then by dt, and IEEE-754 division is exactly rounded, so both
languages are doing the identical arithmetic on the identical doubles.
`PARITY_TOLERANCE` exists only as an outer bound on the claim.

The game rules are pinned the same way. Every scenario below is driven with
`phase_min_s == phase_max_s`, which makes `uniform(a, a) == a` in both
languages and takes the two different PRNGs out of the comparison — so the
event streams can be compared move for move, and any divergence is a real
rules divergence rather than a seeding artefact.
"""

import base64
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from redlight.config import GameConfig
from redlight.game import Game
from redlight.judge import FrameSampler, MotionJudge, diff_score_window

REPO_ROOT = Path(__file__).resolve().parents[1]
JUDGE_JS = REPO_ROOT / "site" / "judge.js"
GAME_JS = REPO_ROOT / "site" / "game.js"
FIXTURES_JSON = REPO_ROOT / "tests" / "fixtures" / "judge_fixtures.json"

PARITY_TOLERANCE = 1e-12
"""Outer bound on browser/Python disagreement. The tests assert exact equality."""

NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(
    NODE is None,
    reason=(
        "node is not installed, so the browser judge cannot be executed here. "
        "Install Node.js (v18+) to run the Python/JavaScript parity suite."
    ),
)


def run_node(script: str, tmp_path: Path) -> dict:
    """Execute a JS snippet under node and return the JSON it prints.

    The snippet is handed the absolute paths it needs through `PATHS`, so
    nothing depends on node's working directory.
    """
    prelude = "const PATHS = " + json.dumps(
        {
            "judge": str(JUDGE_JS),
            "game": str(GAME_JS),
            "fixtures": str(FIXTURES_JSON),
        }
    ) + ";\n"
    script_path = tmp_path / "runner.js"
    script_path.write_text(prelude + script, encoding="utf-8")

    result = subprocess.run(
        [NODE, str(script_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"node exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return json.loads(result.stdout)


def load_fixtures() -> dict:
    assert FIXTURES_JSON.exists(), (
        f"golden fixtures missing at {FIXTURES_JSON}; "
        "generate them with redlight.export.write_judge_fixtures"
    )
    return json.loads(FIXTURES_JSON.read_text(encoding="utf-8"))


def decode_window(b64: str) -> np.ndarray:
    """Turn a fixture's base64 payload back into a 96x96 uint8 window."""
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.uint8).reshape(96, 96)


# --------------------------------------------------------------------------
# The fixtures themselves
# --------------------------------------------------------------------------


def test_fixture_file_is_well_formed():
    """The golden file carries what both sides need to agree on."""
    data = load_fixtures()

    assert data["window"] == 96
    assert data["diff_pixel_delta"] == 25
    cases = data["cases"]
    assert len(cases) >= 24, f"expected at least 24 golden cases, got {len(cases)}"

    names = [case["name"] for case in cases]
    assert len(names) == len(set(names)), "fixture names must be unique"

    kinds = {case["kind"] for case in cases}
    for required in ("identical", "noise", "boundary", "shift", "texture"):
        assert required in kinds, f"fixtures must cover the {required!r} case"

    assert {case["dt"] for case in cases} == {0.033, 0.1}

    for case in cases:
        assert len(base64.b64decode(case["prev"])) == 96 * 96
        assert len(base64.b64decode(case["cur"])) == 96 * 96
        assert isinstance(case["expected"], float)


def test_fixtures_cover_the_wrap_risk_boundary():
    """A naive uint8 subtraction wraps; the fixtures catch that both ways round.

    `prev=100, cur=126` and `prev=126, cur=100` are the same 26-step change
    in opposite directions. Subtract them as unsigned bytes and one of the
    two comes out as 230, which sails past the delta and scores a pixel that
    should not have scored. Both directions are pinned so a port cannot pass
    by getting only the easy sign right.
    """
    data = load_fixtures()
    by_name = {case["name"]: case for case in data["cases"]}

    for name in ("wrap_risk_100_to_126", "wrap_risk_126_to_100"):
        assert name in by_name, f"missing wrap-risk fixture {name!r}"
        case = by_name[name]
        # Exactly one pixel differs, by 26 — one step past the delta.
        assert case["expected"] == pytest.approx(1 / 9216 / case["dt"], rel=1e-15)


def test_fixtures_reproduce_python_exactly():
    """The recorded values still match what the Python judge computes today.

    This is the guard against silent fixture drift: if `diff_score_window`
    ever changes, this fails here rather than quietly re-baselining the
    browser against a stale file.
    """
    data = load_fixtures()

    for case in data["cases"]:
        prev = decode_window(case["prev"])
        cur = decode_window(case["cur"])
        recomputed = diff_score_window(prev, cur, case["dt"])
        assert recomputed == case["expected"], (
            f"fixture {case['name']!r} drifted: "
            f"recorded {case['expected']!r}, Python now says {recomputed!r}"
        )


# --------------------------------------------------------------------------
# diffScoreWindow parity
# --------------------------------------------------------------------------


def test_js_diff_score_matches_python_on_every_fixture(tmp_path):
    """The headline claim: same windows, same dt, same number — bit for bit."""
    data = load_fixtures()

    results = run_node(
        """
        const { diffScoreWindow } = require(PATHS.judge);
        const fixtures = require(PATHS.fixtures);

        const scores = fixtures.cases.map((c) => {
          const prev = new Uint8Array(Buffer.from(c.prev, "base64"));
          const cur = new Uint8Array(Buffer.from(c.cur, "base64"));
          return { name: c.name, score: diffScoreWindow(prev, cur, c.dt) };
        });
        process.stdout.write(JSON.stringify({ scores }));
        """,
        tmp_path,
    )

    js_by_name = {entry["name"]: entry["score"] for entry in results["scores"]}
    assert len(js_by_name) == len(data["cases"])

    worst = 0.0
    for case in data["cases"]:
        expected = case["expected"]
        actual = js_by_name[case["name"]]
        worst = max(worst, abs(actual - expected))
        assert actual == expected, (
            f"fixture {case['name']!r}: browser said {actual!r}, Python said {expected!r}"
        )
    assert worst < PARITY_TOLERANCE
    assert worst == 0.0, f"expected bit-exact agreement, worst absolute difference was {worst!r}"


def test_js_diff_score_rejects_bad_input(tmp_path):
    """Wrong-shaped or untyped input is a bug to surface, exactly as in Python.

    Python raises on a window that is not 96x96 uint8 rather than scoring it.
    The browser has to be just as loud, or a silent shape bug in the arena
    would show up as a wrong verdict instead of an error.
    """
    results = run_node(
        """
        const { diffScoreWindow } = require(PATHS.judge);
        const ok = new Uint8Array(9216);

        function threw(fn) {
          try { fn(); return false; } catch (e) { return true; }
        }

        process.stdout.write(JSON.stringify({
          zero_dt: threw(() => diffScoreWindow(ok, ok, 0)),
          negative_dt: threw(() => diffScoreWindow(ok, ok, -0.1)),
          nan_dt: threw(() => diffScoreWindow(ok, ok, NaN)),
          short_prev: threw(() => diffScoreWindow(new Uint8Array(9215), ok, 0.1)),
          long_cur: threw(() => diffScoreWindow(ok, new Uint8Array(9217), 0.1)),
          plain_array: threw(() => diffScoreWindow(new Array(9216).fill(0), ok, 0.1)),
          float_array: threw(() => diffScoreWindow(new Float32Array(9216), ok, 0.1)),
          null_input: threw(() => diffScoreWindow(null, ok, 0.1)),
          clamped_ok: !threw(() => diffScoreWindow(new Uint8ClampedArray(9216), ok, 0.1)),
        }));
        """,
        tmp_path,
    )

    for name, threw in results.items():
        assert threw is True, f"expected {name} to be rejected by the browser judge"

    # And Python agrees about the shape rule.
    good = np.zeros((96, 96), dtype=np.uint8)
    with pytest.raises(ValueError):
        diff_score_window(good, good, 0.0)
    with pytest.raises(ValueError):
        diff_score_window(good, np.zeros((95, 96), dtype=np.uint8), 0.1)
    with pytest.raises(ValueError):
        diff_score_window(good, np.zeros((96, 96), dtype=np.float32), 0.1)


# --------------------------------------------------------------------------
# MotionJudge parity
# --------------------------------------------------------------------------

JUDGE_SCORE_SEQUENCE = [
    0.0,
    0.30,  # first score seeds the average outright
    0.30,
    0.30,
    0.30,
    0.05,  # one quiet frame puts the streak back to zero
    0.30,
    0.30,
    0.1455,  # exactly at the threshold: not above it, so no streak
    0.30,
    0.30,
    0.30,
    0.30,
]


def test_js_motion_judge_matches_python_step_for_step(tmp_path):
    """Seeding, smoothing, the strict `>`, and the streak reset all line up."""
    threshold = 0.1455

    python_judge = MotionJudge(threshold=threshold, confirm_frames=3, smoothing=0.5)
    expected = [
        {"fired": python_judge.update(7, score), "ema": python_judge.smoothed(7)}
        for score in JUDGE_SCORE_SEQUENCE
    ]

    results = run_node(
        """
        const { MotionJudge } = require(PATHS.judge);
        const scores = %s;
        const judge = new MotionJudge(%r, 3, 0.5);
        const steps = scores.map((s) => ({ fired: judge.update(7, s), ema: judge.smoothed(7) }));
        const before = judge.smoothed(7);
        judge.reset();
        process.stdout.write(JSON.stringify({
          steps,
          ema_before_reset: before,
          ema_after_reset: judge.smoothed(7),
          unseen_player: judge.smoothed(999),
        }));
        """
        % (json.dumps(JUDGE_SCORE_SEQUENCE), threshold),
        tmp_path,
    )

    assert len(results["steps"]) == len(expected)
    for index, (js_step, py_step) in enumerate(zip(results["steps"], expected)):
        assert js_step["fired"] == py_step["fired"], f"verdict differs at step {index}"
        assert js_step["ema"] == py_step["ema"], f"smoothed score differs at step {index}"

    # A reset really does forget the player, so a new round starts clean.
    python_judge.reset()
    assert results["ema_after_reset"] == python_judge.smoothed(7) == 0.0
    assert results["ema_before_reset"] != 0.0
    assert results["unseen_player"] == 0.0


def test_js_motion_judge_fires_on_the_same_frame_as_python(tmp_path):
    """Confirmation is a count of frames, and both sides count the same one."""
    results = run_node(
        """
        const { MotionJudge } = require(PATHS.judge);
        const judge = new MotionJudge(0.1, 3, 1.0);  // smoothing 1.0 -> raw scores
        const fired = [0, 1, 2, 3, 4].map(() => judge.update(1, 0.5));
        process.stdout.write(JSON.stringify({ fired }));
        """,
        tmp_path,
    )

    python_judge = MotionJudge(threshold=0.1, confirm_frames=3, smoothing=1.0)
    expected = [python_judge.update(1, 0.5) for _ in range(5)]

    assert results["fired"] == expected == [False, False, True, True, True]


# --------------------------------------------------------------------------
# FrameSampler parity
# --------------------------------------------------------------------------


def test_js_frame_sampler_matches_python(tmp_path):
    """Same fixed clock, same released pairs, same dt — including the tolerance.

    The 30 fps run is the one that matters: `i / 30` lands a fraction of an
    ulp under a clean 0.1 about half the time, and without the tolerance the
    sampler paces itself at half the rate it was asked for. Both languages
    have to make the same call on those boundary timestamps.
    """
    timestamps = [i / 30.0 for i in range(31)] + [1.0 + i / 10.0 for i in range(1, 6)]

    sampler = FrameSampler()
    expected = []
    for index, ts in enumerate(timestamps):
        sampled = sampler.offer(np.uint8(index), ts)
        expected.append(None if sampled is None else {"prev": int(sampled[0]), "dt": sampled[2]})

    results = run_node(
        """
        const { FrameSampler } = require(PATHS.judge);
        const timestamps = %s;
        const sampler = new FrameSampler();
        const released = timestamps.map((ts, i) => {
          const out = sampler.offer(i, ts);
          return out === null ? null : { prev: out.prev, dt: out.dt };
        });
        sampler.reset();
        process.stdout.write(JSON.stringify({
          released,
          primes_after_reset: sampler.offer(99, 99.0) === null,
        }));
        """
        % json.dumps(timestamps),
        tmp_path,
    )

    assert results["released"] == expected
    assert results["primes_after_reset"] is True

    released_count = sum(1 for entry in expected if entry is not None)
    assert released_count >= 8, (
        f"the tolerance should keep the 30 fps clock releasing near 10 Hz, got {released_count}"
    )


# --------------------------------------------------------------------------
# Game semantics parity
# --------------------------------------------------------------------------

# Every scenario pins phase_min_s == phase_max_s so `uniform(a, a) == a` and
# the two languages' PRNGs drop out of the comparison entirely.
GAME_SCENARIOS = [
    {
        "name": "grace_boundary",
        "config": {
            "seed": 42,
            "countdown_s": 1.0,
            "duration_s": 100.0,
            "phase_min_s": 1.0,
            "phase_max_s": 1.0,
            "grace_s": 0.5,
        },
        "players": [1, 2],
        # RED starts at t=2.0; motion at 2.4 is inside grace, at 2.6 it is not.
        "ticks": [
            {"now": 1.0},
            {"now": 2.0},
            {"now": 2.4, "violations": [1]},
            {"now": 2.6, "violations": [1]},
        ],
    },
    {
        "name": "registration_ignores_strangers",
        "config": {
            "seed": 42,
            "countdown_s": 1.0,
            "duration_s": 100.0,
            "phase_min_s": 1.0,
            "phase_max_s": 1.0,
            "grace_s": 0.0,
        },
        "players": [1, 2],
        "ticks": [
            {"now": 1.0},
            {"now": 2.0},
            {"now": 2.0, "violations": [999]},  # never registered: ignored
            {"now": 2.0, "violations": [1]},  # registered: out
            {"now": 2.0, "violations": [1]},  # already out: ignored
            {"now": 2.0, "missing": [1, 999]},  # still ignored
        ],
    },
    {
        "name": "wipeout_beats_victory_on_the_same_tick",
        "config": {
            "seed": 42,
            "countdown_s": 1.0,
            "duration_s": 2.0,
            "phase_min_s": 1.0,
            "phase_max_s": 1.0,
            "grace_s": 0.0,
        },
        "players": [1, 2],
        "ticks": [
            {"now": 1.0},
            {"now": 2.0},
            {"now": 3.0, "violations": [1, 2]},  # clock expiry and last elimination collide
        ],
    },
    {
        "name": "catch_up_flips_replay_every_boundary",
        "config": {
            "seed": 42,
            "countdown_s": 1.0,
            "duration_s": 1000.0,
            "phase_min_s": 1.0,
            "phase_max_s": 1.0,
            "grace_s": 0.6,
        },
        "players": [1],
        # One late update skipping three whole phase boundaries at once.
        "ticks": [{"now": 1.0}, {"now": 4.5}],
    },
    {
        "name": "missing_eliminates_under_green",
        "config": {
            "seed": 42,
            "countdown_s": 1.0,
            "duration_s": 100.0,
            "phase_min_s": 5.0,
            "phase_max_s": 5.0,
            "grace_s": 0.6,
        },
        "players": [1, 2],
        "ticks": [
            {"now": 1.0},
            {"now": 1.2, "violations": [1]},  # green forgives motion
            {"now": 1.4, "missing": [1]},  # but not a lost track
        ],
    },
]


def drive_python_scenario(scenario: dict) -> dict:
    """Run one scenario through the Python game and record everything observable."""
    config = GameConfig(**scenario["config"])
    game = Game(config)

    log = []
    for event in game.start(0.0, scenario["players"]):
        log.append([event.type.name, event.phase.name, event.track_id, event.reason])

    states = []
    for tick in scenario["ticks"]:
        events = game.update(
            tick["now"],
            violations=tick.get("violations"),
            missing=tick.get("missing"),
        )
        log.extend(
            [event.type.name, event.phase.name, event.track_id, event.reason]
            for event in events
        )
        states.append(
            {
                "now": tick["now"],
                "phase": game.phase.name,
                "armed": game.armed(tick["now"]),
                "alive": game.alive_count,
                "time_left": game.time_left(tick["now"]),
            }
        )

    return {"events": log, "states": states, "final_phase": game.phase.name}


@pytest.mark.parametrize("scenario", GAME_SCENARIOS, ids=lambda s: s["name"])
def test_js_game_matches_python_scenario(scenario, tmp_path):
    """The browser referee and the Python referee call the same match."""
    expected = drive_python_scenario(scenario)

    results = run_node(
        """
        const { Game } = require(PATHS.game);
        const scenario = %s;
        const c = scenario.config;
        const game = new Game({
          seed: c.seed,
          countdownS: c.countdown_s,
          durationS: c.duration_s,
          phaseMinS: c.phase_min_s,
          phaseMaxS: c.phase_max_s,
          graceS: c.grace_s,
        });

        const log = [];
        const record = (events) => {
          for (const e of events) log.push([e.type, e.phase, e.trackId, e.reason]);
        };

        record(game.start(0.0, scenario.players));

        const states = scenario.ticks.map((tick) => {
          record(game.update(tick.now, tick.violations || null, tick.missing || null));
          return {
            now: tick.now,
            phase: game.phase,
            armed: game.armed(tick.now),
            alive: game.aliveCount,
            time_left: game.timeLeft(tick.now),
          };
        });

        process.stdout.write(JSON.stringify({
          events: log,
          states,
          final_phase: game.phase,
        }));
        """
        % json.dumps(scenario),
        tmp_path,
    )

    assert results["events"] == expected["events"]
    assert results["states"] == expected["states"]
    assert results["final_phase"] == expected["final_phase"]


def test_grace_scenario_really_exercises_both_sides_of_the_window():
    """Guard the scenario table itself: the grace case must actually flip.

    A scenario that forgave both ticks, or eliminated on both, would still
    pass the parity comparison while testing nothing about grace.
    """
    scenario = next(s for s in GAME_SCENARIOS if s["name"] == "grace_boundary")
    result = drive_python_scenario(scenario)

    armed_flags = [state["armed"] for state in result["states"]]
    assert armed_flags == [False, False, False, True]
    eliminations = [event for event in result["events"] if event[0] == "PLAYER_ELIMINATED"]
    assert eliminations == [["PLAYER_ELIMINATED", "RED", 1, "moved"]]


def test_catch_up_scenario_really_skips_multiple_boundaries():
    """Guard the catch-up case: one update must cross more than one boundary."""
    scenario = next(s for s in GAME_SCENARIOS if s["name"] == "catch_up_flips_replay_every_boundary")
    result = drive_python_scenario(scenario)

    phase_changes = [event for event in result["events"] if event[0] == "PHASE_CHANGED"]
    # COUNTDOWN, GREEN, then three flips crossed by the single late update.
    assert [event[1] for event in phase_changes] == [
        "COUNTDOWN",
        "GREEN",
        "RED",
        "GREEN",
        "RED",
    ]


def test_js_game_schedule_is_deterministic_within_the_browser(tmp_path):
    """Same seed, same schedule; a different seed genuinely diverges.

    The browser draws its light schedule from mulberry32 rather than from
    Python's Mersenne Twister, so the two languages are not expected to
    produce the same *lengths* — only the same rules. What has to hold on
    the browser side is that a seed pins a match: replay it and you get the
    identical run.
    """
    results = run_node(
        """
        const { Game, mulberry32 } = require(PATHS.game);

        function flipTimes(seed) {
          const game = new Game({
            seed,
            countdownS: 1.0,
            durationS: 10000.0,
            phaseMinS: 1.0,
            phaseMaxS: 3.0,
            graceS: 0.0,
          });
          game.start(0.0, [1]);
          game.update(1.0);

          const times = [];
          let now = 1.0;
          while (times.length < 6) {
            now += 0.05;
            const events = game.update(now);
            if (events.some((e) => e.type === "PHASE_CHANGED")) times.push(game.phase);
          }
          return times;
        }

        const draws = (seed) => {
          const rng = mulberry32(seed);
          return [rng(), rng(), rng()];
        };

        process.stdout.write(JSON.stringify({
          same_a: flipTimes(42),
          same_b: flipTimes(42),
          other: flipTimes(7),
          draws_a: draws(42),
          draws_b: draws(42),
          draws_other: draws(7),
        }));
        """,
        tmp_path,
    )

    assert results["same_a"] == results["same_b"]
    assert results["same_a"] == ["RED", "GREEN", "RED", "GREEN", "RED", "GREEN"]
    assert results["draws_a"] == results["draws_b"]
    assert results["draws_a"] != results["draws_other"]
    for draw in results["draws_a"]:
        assert 0.0 <= draw < 1.0


def test_js_game_rejects_invalid_setup(tmp_path):
    """Zero-length phases would spin the catch-up loop forever, so they are refused.

    The Python config validator already treats a non-positive phase length as
    a configuration error for exactly this reason. The browser has to refuse
    it too, or a bad seed value would hang the tab instead of erroring.
    """
    results = run_node(
        """
        const { Game } = require(PATHS.game);
        const base = {
          seed: 1, countdownS: 1, durationS: 10,
          phaseMinS: 1, phaseMaxS: 1, graceS: 0,
        };
        const threw = (over) => {
          try { new Game({ ...base, ...over }); return false; } catch (e) { return true; }
        };

        const started = new Game(base);
        started.start(0, [1]);
        let restartThrew = false;
        try { started.start(1, [2]); } catch (e) { restartThrew = true; }

        let emptyThrew = false;
        try { new Game(base).start(0, []); } catch (e) { emptyThrew = true; }

        process.stdout.write(JSON.stringify({
          zero_min: threw({ phaseMinS: 0 }),
          negative_max: threw({ phaseMaxS: -1 }),
          min_over_max: threw({ phaseMinS: 4, phaseMaxS: 2 }),
          negative_grace: threw({ graceS: -1 }),
          restart: restartThrew,
          empty_roster: emptyThrew,
        }));
        """,
        tmp_path,
    )

    for name, threw in results.items():
        assert threw is True, f"expected the browser game to reject {name}"

    # Python refuses the same setups.
    with pytest.raises(Exception):
        GameConfig(phase_min_s=0.0).validate()
    with pytest.raises(Exception):
        GameConfig(phase_min_s=4.0, phase_max_s=2.0).validate()
