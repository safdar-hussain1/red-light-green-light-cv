"""The shipped defaults must match what the benchmark measured.

`GameConfig`'s `threshold`/`diff_threshold` are not picked by hand: they are
the benchmark's margin-maximizing midpoint between measured stillness and
the slowest measured walking (see `benchmark._choose_threshold`), rounded to
4dp. These tests read `reports/benchmark_results.json` and assert the
shipped defaults still match it — if someone reruns the benchmark on new
footage and forgets to update `GameConfig`, this is what catches it.

The JSON is a generated artifact, not something every clone of this repo
carries (a fresh checkout has no `reports/` until someone runs the
benchmark), so both tests skip rather than fail when it is absent.

The rest of this file smoke-tests `scripts/make_figures.py`'s pure
JSON-reading helpers against the real results file. The figure-drawing
functions themselves are not exercised here — building every PNG is a
multi-minute job involving YOLO and a video decode, not something a test
suite should pay for on every run. What's worth a test is that the small
functions responsible for pulling numbers out of the JSON do that correctly,
since a typo in a dict key there fails silently (a wrong bar height) rather
than loudly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from redlight.config import GameConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "reports" / "benchmark_results.json"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import make_figures  # noqa: E402


def _load_results() -> dict:
    if not RESULTS_PATH.exists():
        pytest.skip(f"{RESULTS_PATH} not present in this checkout")
    return json.loads(RESULTS_PATH.read_text())


class TestThresholdSync:
    def test_flow_threshold_matches_benchmark_midpoint(self):
        results = _load_results()
        expected = round(results["chosen_thresholds"]["flow"], 4)

        assert GameConfig().threshold == expected

    def test_diff_threshold_matches_benchmark_midpoint(self):
        results = _load_results()
        expected = round(results["chosen_thresholds"]["diff"], 4)

        assert GameConfig().diff_threshold == expected


class TestMakeFiguresDataHelpers:
    """Smoke tests: the pure readers parse the real JSON without exploding."""

    @pytest.fixture(scope="class")
    @classmethod
    def results(cls):
        return _load_results()

    def test_load_results_reads_the_real_file(self):
        if not RESULTS_PATH.exists():
            pytest.skip(f"{RESULTS_PATH} not present in this checkout")

        results = make_figures.load_results(RESULTS_PATH)

        assert results["meta"]["video"] == "data/vtest.avi"

    def test_runtime_rows_covers_every_runtime_entry(self, results):
        rows = make_figures.runtime_rows(results)

        assert {label for label, _ms in rows} == {
            "YOLO11n detection",
            "HOG detection (weights-free)",
            "flow_norm per player",
            "diff_norm per player",
        }
        for _label, ms in rows:
            assert ms >= 0.0

    def test_resolution_sweep_rows_covers_every_metric_and_scale(self, results):
        rows = make_figures.resolution_sweep_rows(results)

        metrics = {row["metric"] for row in rows}
        scales = {row["scale"] for row in rows}
        assert metrics == {"brightness", "raw_flow", "flow_norm", "diff_norm"}
        assert scales == {"0.5x", "1x", "1.667x"}
        for row in rows:
            assert 0.0 <= row["moving_flagged"] <= 100.0
            assert 0.0 <= row["frozen_flagged"] <= 100.0

    def test_chosen_thresholds_rounds_to_four_places(self, results):
        thresholds = make_figures.chosen_thresholds(results)

        assert thresholds == {
            "flow": round(results["chosen_thresholds"]["flow"], 4),
            "diff": round(results["chosen_thresholds"]["diff"], 4),
        }
