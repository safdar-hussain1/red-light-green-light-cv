"""Tests for the measurement harness.

Every test here drives the real harness over `data/vtest.avi` in `--fast`
mode, which caps the footage at `benchmark.FAST_FRAMES` frames so the whole
file stays under a minute even though it runs the genuine detect -> track ->
score -> sweep pipeline. The fast run is built once per module and shared;
the determinism test is the one place a second run is paid for.

These are plumbing-and-honesty tests, not calibration tests: they assert the
harness reports a complete, well-formed, reproducible measurement, never
that a particular metric scores a particular number. The numbers are
whatever the footage says they are.
"""

from __future__ import annotations

import json

import pytest

from redlight import __version__, benchmark

CLASSIFIER_METRICS = ("brightness", "frame_diff_area", "raw_flow", "flow_norm", "diff_norm")
CLASSIFIER_FIELDS = (
    "auc",
    "threshold",
    "frozen_flagged",
    "moving_flagged",
    "frozen_median",
    "moving_median",
)
SWEEP_METRICS = ("brightness", "raw_flow", "flow_norm", "diff_norm")
SCALE_KEYS = ("0.5x", "1x", "1.667x")


@pytest.fixture(scope="module")
def fast_run(tmp_path_factory):
    """One `--fast` benchmark, as (payload, written JSON text)."""
    out_path = tmp_path_factory.mktemp("benchmark") / "results.json"
    payload = benchmark.run_benchmark(str(out_path), fast=True)
    return payload, out_path.read_text()


@pytest.fixture(scope="module")
def results(fast_run):
    return fast_run[0]


class TestOutputFile:
    def test_writes_json_matching_the_returned_payload(self, fast_run):
        payload, text = fast_run

        assert json.loads(text) == payload

    def test_creates_missing_parent_directories(self, tmp_path):
        out_path = tmp_path / "nested" / "deeper" / "results.json"

        benchmark.run_benchmark(str(out_path), fast=True)

        assert out_path.exists()


class TestSchema:
    def test_has_every_top_level_block(self, results):
        assert set(results) == {
            "meta",
            "classifiers",
            "chosen_thresholds",
            "threshold_separation",
            "resolution_sweep",
            "fps_sweep",
            "background_scenario",
            "runtime_ms",
            "traces",
            "demo_match",
        }

    def test_meta_records_the_ground_truth_it_measured(self, results):
        meta = results["meta"]

        assert meta["video"] == benchmark.VIDEO
        assert meta["fps"] > 0
        assert meta["frames_used"] <= benchmark.FAST_FRAMES
        assert 0 < meta["pairs_sampled"] <= meta["frames_used"]
        assert meta["fast"] is True
        assert meta["seed"] == benchmark.SEED
        assert meta["sample_interval_s"] == benchmark.SAMPLE_INTERVAL_S
        assert meta["noise_sigma"] == benchmark.NOISE_SIGMA
        assert meta["package_version"] == __version__
        assert meta["n_moving"] > 0
        # Moving and frozen samples are paired: the same retained boxes on the
        # same frames, so the two classes cannot differ in box distribution.
        assert meta["n_frozen"] == meta["n_moving"]
        assert set(meta["thresholds"]) == set(CLASSIFIER_METRICS)

    def test_meta_reports_every_exclusion_category(self, results):
        exclusions = results["meta"]["exclusions"]

        for key in ("no_prior_box", "static_box", "unusable_crop"):
            assert exclusions[key] >= 0
        assert exclusions["min_center_displacement_px"] > 0

    def test_classifier_table_covers_every_metric(self, results):
        assert set(results["classifiers"]) == set(CLASSIFIER_METRICS)
        for metric in CLASSIFIER_METRICS:
            assert set(results["classifiers"][metric]) == set(CLASSIFIER_FIELDS)

    def test_resolution_sweep_covers_every_metric_at_every_scale(self, results):
        sweep = results["resolution_sweep"]

        assert set(sweep) == {*SWEEP_METRICS, "samples"}
        assert sweep["samples"] > 0
        for metric in SWEEP_METRICS:
            assert set(sweep[metric]) == set(SCALE_KEYS)
            for scale in SCALE_KEYS:
                entry = sweep[metric][scale]
                assert set(entry) == {"frozen_flagged", "moving_flagged"}

    def test_fps_sweep_covers_both_normalized_metrics(self, results):
        sweep = results["fps_sweep"]

        assert set(sweep) == {"flow_norm", "diff_norm"}
        for metric in ("flow_norm", "diff_norm"):
            entry = sweep[metric]
            assert set(entry) == {
                "10fps",
                "30fps",
                "pairs_compared",
                "dt_mismatch_pct",
                "decisions_compared",
                "decisions_agree_pct",
            }
            for arm in ("10fps", "30fps"):
                assert set(entry[arm]) == {"pairs_released", "samples", "flagged_pct"}

    def test_background_scenario_reports_its_measurements(self, results):
        scenario = results["background_scenario"]

        assert set(scenario) == {
            "frames",
            "samples",
            "frozen_box",
            "frame_diff_area_flagged_pct",
            "first_crossing_s",
            "per_box_flagged_pct",
        }
        assert set(scenario["frozen_box"]) == {"x", "y", "w", "h"}
        assert set(scenario["per_box_flagged_pct"]) == {"flow_norm", "diff_norm"}

    def test_runtime_block_times_both_detectors_and_both_metrics(self, results):
        assert set(results["runtime_ms"]) == {
            "yolo11n",
            "hog",
            "flow_norm_per_player",
            "diff_norm_per_player",
        }

    def test_demo_match_reports_a_finished_run(self, results):
        match = results["demo_match"]

        assert set(match) == {"outcome", "players", "survivors", "eliminations", "config"}
        assert match["outcome"] in ("victory", "wipeout", "aborted")
        assert 0 <= match["survivors"] <= match["players"]
        for track_id, elapsed, reason in match["eliminations"]:
            assert isinstance(track_id, int)
            assert elapsed >= 0
            assert reason in ("moved", "left_arena")


class TestClassifierNumbers:
    def test_every_auc_is_a_probability(self, results):
        for metric in CLASSIFIER_METRICS:
            auc = results["classifiers"][metric]["auc"]
            assert 0.0 <= auc <= 1.0, f"{metric} auc out of range: {auc}"

    def test_flagged_rates_are_percentages(self, results):
        for metric in CLASSIFIER_METRICS:
            entry = results["classifiers"][metric]
            assert 0.0 <= entry["frozen_flagged"] <= 100.0
            assert 0.0 <= entry["moving_flagged"] <= 100.0

    def test_thresholds_match_the_shipped_ones(self, results):
        for metric in CLASSIFIER_METRICS:
            assert results["classifiers"][metric]["threshold"] == benchmark.THRESHOLDS[metric]

    def test_normalized_metrics_separate_moving_from_frozen(self, results):
        """The claim the product rests on, on the fast subset: both normalized
        metrics rank a walking pedestrian above a held frame nearly perfectly.
        """
        for metric in ("flow_norm", "diff_norm"):
            assert results["classifiers"][metric]["auc"] > 0.95


class TestChosenThresholds:
    def test_reports_a_threshold_for_both_normalized_metrics(self, results):
        assert set(results["chosen_thresholds"]) == {"flow", "diff"}
        assert set(results["threshold_separation"]) == {"flow", "diff"}

    def test_separation_is_asserted_before_a_threshold_is_offered(self, results):
        """No threshold may be invented out of overlapping distributions."""
        for key in ("flow", "diff"):
            separation = results["threshold_separation"][key]
            chosen = results["chosen_thresholds"][key]
            if separation["separated"]:
                assert chosen is not None
            else:
                assert chosen is None

    def test_chosen_thresholds_sit_between_the_fast_set_distributions(self, results):
        for key in ("flow", "diff"):
            separation = results["threshold_separation"][key]
            chosen = results["chosen_thresholds"][key]
            assert separation["separated"], f"{key} distributions overlap on the fast subset"
            assert separation["frozen_p99"] <= chosen < separation["moving_p1"]

    def test_chosen_thresholds_separate_the_fast_set_samples(self, results):
        """Applied to the samples they were derived from, they must clear the
        frozen class and keep nearly all of the moving one.
        """
        for key, metric in (("flow", "flow_norm"), ("diff", "diff_norm")):
            applied = results["threshold_separation"][key]["applied"]
            assert applied["frozen_flagged"] == 0.0, f"{metric} flags held frames"
            assert applied["moving_flagged"] >= 95.0, f"{metric} misses walkers"


class TestInvarianceSweeps:
    def test_resolution_sweep_rates_are_percentages(self, results):
        for metric in SWEEP_METRICS:
            for scale in SCALE_KEYS:
                entry = results["resolution_sweep"][metric][scale]
                assert 0.0 <= entry["frozen_flagged"] <= 100.0
                assert 0.0 <= entry["moving_flagged"] <= 100.0

    def test_normalized_metrics_hold_their_verdict_across_resolutions(self, results):
        """The resolution-invariance claim: rescaling the footage must not move
        the verdict for the normalized metrics.
        """
        for metric in ("flow_norm", "diff_norm"):
            rates = [
                results["resolution_sweep"][metric][scale]["moving_flagged"]
                for scale in SCALE_KEYS
            ]
            assert max(rates) - min(rates) <= 10.0, f"{metric} drifted with resolution: {rates}"
            for scale in SCALE_KEYS:
                assert results["resolution_sweep"][metric][scale]["frozen_flagged"] == 0.0

    def test_fps_sweep_agreement_is_a_percentage_over_real_decisions(self, results):
        for metric in ("flow_norm", "diff_norm"):
            entry = results["fps_sweep"][metric]
            assert entry["decisions_compared"] > 0
            assert entry["pairs_compared"] > 0
            assert 0.0 <= entry["decisions_agree_pct"] <= 100.0
            assert 0.0 <= entry["dt_mismatch_pct"] <= 100.0

    def test_same_content_at_10_and_30fps_ticks_decides_the_same(self, results):
        """The frame-rate-invariance claim, measured through `FrameSampler`.

        Not pinned at exactly 100%: the sampler releases on a wall clock, so
        the two tick rates hand the judge intervals that differ by up to a
        third for the very same pair of frames, and a score sitting on the
        threshold can land either side of it. That residual is real and is
        reported as `dt_mismatch_pct` beside the agreement, so the bar here
        is what invariance actually buys, not a round number.
        """
        for metric in ("flow_norm", "diff_norm"):
            entry = results["fps_sweep"][metric]
            assert entry["decisions_agree_pct"] >= 99.0, entry


class TestBackgroundScenario:
    def test_a_frozen_player_is_never_flagged_by_the_per_box_metrics(self, results):
        """The spectator claim: traffic moving around a still player must not
        put that player out.
        """
        per_box = results["background_scenario"]["per_box_flagged_pct"]

        assert per_box["flow_norm"] == 0.0
        assert per_box["diff_norm"] == 0.0

    def test_whole_frame_differencing_flags_the_same_still_player(self, results):
        """The naive design's failure, measured rather than asserted."""
        scenario = results["background_scenario"]

        assert scenario["samples"] > 0
        assert scenario["frame_diff_area_flagged_pct"] > 0.0
        assert scenario["first_crossing_s"] is not None
        assert scenario["first_crossing_s"] >= 0.0


class TestRuntime:
    def test_every_timing_is_a_positive_number_of_milliseconds(self, results):
        for key, value in results["runtime_ms"].items():
            assert value > 0.0, f"{key} timed at {value} ms"
            assert value < 10_000.0, f"{key} implausibly slow: {value} ms"

    def test_the_normalized_metrics_are_cheaper_than_detection(self, results):
        runtime = results["runtime_ms"]

        assert runtime["flow_norm_per_player"] < runtime["yolo11n"]
        assert runtime["diff_norm_per_player"] < runtime["yolo11n"]


class TestTraces:
    def test_traces_are_non_empty_and_well_formed(self, results):
        traces = results["traces"]

        assert traces["metric"] == "flow_norm"
        assert traces["threshold"] > 0
        assert traces["red_phase_player_traces"]
        for trace in traces["red_phase_player_traces"]:
            assert set(trace) == {"track_id", "t", "score"}
            assert trace["t"], "trace carries no timestamps"
            assert len(trace["t"]) == len(trace["score"])
            assert trace["t"] == sorted(trace["t"])
            assert all(score >= 0 for score in trace["score"])


class TestReproducibility:
    def test_no_timestamp_or_date_leaks_into_the_json(self, fast_run):
        """A rerun has to be byte-comparable, so nothing wall-clock may be
        recorded — not even a harmless "generated at" stamp.
        """
        _, text = fast_run
        lowered = text.lower()

        for banned in ('"date', '"timestamp', '"generated', '"run_at', '"created'):
            assert banned not in lowered

    def test_two_runs_agree_byte_for_byte_apart_from_the_timings(self, fast_run, tmp_path):
        """Everything the harness measures about the footage is reproducible.

        `runtime_ms` is the one exception and is excluded here: it measures
        how fast this machine is, which genuinely varies between runs. Faking
        it stable would mean publishing a number that was not measured.
        """
        _, first_text = fast_run
        second_path = tmp_path / "again.json"
        benchmark.run_benchmark(str(second_path), fast=True)

        first = json.loads(first_text)
        second = json.loads(second_path.read_text())
        first.pop("runtime_ms")
        second.pop("runtime_ms")

        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
