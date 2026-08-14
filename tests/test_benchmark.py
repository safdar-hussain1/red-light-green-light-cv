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

import numpy as np
import pytest

from redlight import __version__, benchmark

CLASSIFIER_METRICS = ("brightness", "frame_diff_area", "raw_flow", "flow_norm", "diff_norm")
CLASSIFIER_FIELDS = (
    "auc",
    "scope",
    "threshold",
    "frozen_flagged",
    "moving_flagged",
    "frozen_median",
    "moving_median",
)
SCENARIO_FIELDS = {
    "frames",
    "samples",
    "overlap_frames",
    "overlap_samples",
    "frozen_box",
    "frame_diff_area_flagged_pct",
    "first_crossing_s",
    "per_box_flagged_pct",
    "per_box_flagged_pct_overlap",
    "note",
}
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

    def test_background_scenario_reports_both_variants(self, results):
        scenarios = results["background_scenario"]

        assert set(scenarios) == {"isolated", "occluded"}
        for variant in scenarios.values():
            assert set(variant) == SCENARIO_FIELDS
            assert set(variant["frozen_box"]) == {"x", "y", "w", "h"}
            for key in ("per_box_flagged_pct", "per_box_flagged_pct_overlap"):
                assert set(variant[key]) == {"flow_norm", "diff_norm"}
            assert variant["note"]

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

    def test_chosen_threshold_is_the_exact_midpoint_of_the_two_edges(self, results):
        """Pins the arithmetic, not a range that the construction guarantees.

        `frozen_p99 <= chosen < moving_p1` is true of any midpoint by
        definition and so tests nothing; this pins the value itself, which a
        changed rule would move.
        """
        for key in ("flow", "diff"):
            separation = results["threshold_separation"][key]
            chosen = results["chosen_thresholds"][key]
            assert separation["separated"], f"{key} distributions overlap on the fast subset"
            expected = (separation["frozen_p99"] + separation["moving_p1"]) / 2
            assert chosen == pytest.approx(expected, abs=1e-6)
            assert separation["rule"] == "midpoint"

    def test_chosen_thresholds_clear_every_held_frame(self, results):
        """The half of the claim that is not entailed by the construction.

        A high moving-flag rate follows from picking a cutoff under the
        moving 1st percentile, so asserting it proves nothing. That the
        frozen class stays *entirely* below the cutoff does not follow — the
        rule only looks at the frozen 99th percentile, so the top 1% is free
        to sit above it and does not.
        """
        for key, metric in (("flow", "flow_norm"), ("diff", "diff_norm")):
            applied = results["threshold_separation"][key]["applied"]
            assert applied["frozen_flagged"] == 0.0, f"{metric} flags held frames"

    def test_midpoint_rule_holds_when_the_frozen_edge_is_nonzero(self):
        """The rule on synthetic distributions, where the answer is known.

        The real footage produces `frozen_p99 == 0`, so the full run alone
        could not tell the surviving midpoint rule apart from one that
        ignored the frozen edge entirely.
        """
        samples = benchmark._Samples()
        samples.frozen["flow_norm"] = [0.2] * 100
        samples.moving["flow_norm"] = [0.6] * 100

        chosen, separation = benchmark._choose_threshold(samples, "flow_norm")

        assert separation["frozen_p99"] == pytest.approx(0.2)
        assert separation["moving_p1"] == pytest.approx(0.6)
        assert chosen == pytest.approx(0.4)  # not sqrt(0.2*0.6) == 0.3464
        assert separation["rule"] == "midpoint"

    def test_overlapping_distributions_get_no_threshold_at_all(self):
        """No cutoff may be invented where none separates the classes."""
        samples = benchmark._Samples()
        samples.frozen["flow_norm"] = list(np.linspace(0.0, 1.0, 100))
        samples.moving["flow_norm"] = list(np.linspace(0.0, 1.0, 100))

        chosen, separation = benchmark._choose_threshold(samples, "flow_norm")

        assert chosen is None
        assert separation["separated"] is False
        assert separation["rule"] == "none"
        assert separation["applied"]["frozen_flagged"] is None


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
    def test_a_frozen_player_is_never_flagged_when_nothing_crosses_them(self, results):
        """The spectator claim: traffic moving elsewhere in the shot must not
        put a still player out.
        """
        per_box = results["background_scenario"]["isolated"]["per_box_flagged_pct"]

        assert per_box["flow_norm"] == 0.0
        assert per_box["diff_norm"] == 0.0

    def test_the_isolated_variant_had_walkers_it_painted_over(self, results):
        """Its 0% has to be a measurement, not a definition.

        Walkers do cross that box in the footage; the isolated composite
        draws the still player over them. Counting the crossings anyway is
        what makes its clean result comparable with the occluded one instead
        of true by construction.
        """
        isolated = results["background_scenario"]["isolated"]

        assert isolated["overlap_frames"] > 0
        assert isolated["per_box_flagged_pct_overlap"]["flow_norm"] == 0.0

    def test_whole_frame_differencing_flags_the_same_still_player(self, results):
        """The naive design's failure, measured rather than asserted."""
        scenario = results["background_scenario"]["isolated"]

        assert scenario["samples"] > 0
        assert scenario["frame_diff_area_flagged_pct"] > 0.0
        assert scenario["first_crossing_s"] is not None
        assert scenario["first_crossing_s"] >= 0.0

    def test_the_occluded_variant_actually_puts_walkers_across_the_box(self, results):
        """The harder scenario has to be harder, or its result means nothing.

        No assertion on whether occlusion fires the judge: that number is
        whatever the footage says, and it is a documented limitation either
        way. What must hold is that walkers really did cross the box —
        otherwise the variant is silently measuring the isolated case again.
        """
        occluded = results["background_scenario"]["occluded"]

        assert occluded["overlap_frames"] > 0
        assert occluded["overlap_samples"] > 0

    def test_the_two_variants_stand_the_player_in_different_places(self, results):
        isolated = results["background_scenario"]["isolated"]["frozen_box"]
        occluded = results["background_scenario"]["occluded"]["frozen_box"]

        assert (isolated["x"], isolated["y"]) != (occluded["x"], occluded["y"])
        assert (isolated["w"], isolated["h"]) == (occluded["w"], occluded["h"])


class TestRuntime:
    def test_every_timing_is_a_positive_number_of_milliseconds(self, results):
        for key, value in results["runtime_ms"].items():
            assert value > 0.0, f"{key} timed at {value} ms"
            assert value < 10_000.0, f"{key} implausibly slow: {value} ms"

    def test_the_normalized_metrics_are_cheaper_than_detection(self, results):
        runtime = results["runtime_ms"]

        assert runtime["flow_norm_per_player"] < runtime["yolo11n"]
        assert runtime["diff_norm_per_player"] < runtime["yolo11n"]


class TestHeldFrame:
    """`_held_frame` builds the entire frozen class. If it ever stopped
    adding noise, every frozen score would drop to a clean zero and every
    result in the file would improve — silently, and for no real reason.
    """

    @staticmethod
    def _frame() -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.integers(40, 210, size=(64, 64), dtype=np.uint8)

    def test_output_is_a_uint8_frame_of_the_same_shape(self):
        frame = self._frame()

        held = benchmark._held_frame(frame, np.random.default_rng(1))

        assert held.dtype == np.uint8
        assert held.shape == frame.shape

    def test_the_held_frame_is_not_the_frame_it_was_given(self):
        frame = self._frame()

        held = benchmark._held_frame(frame, np.random.default_rng(1))

        assert not np.array_equal(held, frame), "no grain was added"
        assert np.count_nonzero(held != frame) > frame.size // 4

    def test_deviation_matches_the_declared_sigma(self):
        """Grain of sigma 2, not sigma 20 and not sigma 0.2.

        Measured on the signed difference, away from the 0/255 clip, so the
        bound is on what was actually added.
        """
        frame = self._frame()

        held = benchmark._held_frame(frame, np.random.default_rng(2))
        deviation = held.astype(np.int16) - frame.astype(np.int16)

        assert deviation.std() == pytest.approx(benchmark.NOISE_SIGMA, rel=0.15)
        assert np.abs(deviation).max() <= 6 * benchmark.NOISE_SIGMA

    def test_output_stays_inside_the_byte_range_at_the_extremes(self):
        """Clipping, not wrapping: black must not roll over to white."""
        for value in (0, 255):
            frame = np.full((32, 32), value, dtype=np.uint8)

            held = benchmark._held_frame(frame, np.random.default_rng(3))

            assert held.min() >= 0 and held.max() <= 255
            assert np.abs(held.astype(np.int16) - value).max() <= 6 * benchmark.NOISE_SIGMA

    def test_the_same_generator_state_reproduces_the_same_grain(self):
        frame = self._frame()

        first = benchmark._held_frame(frame, np.random.default_rng(4))
        second = benchmark._held_frame(frame, np.random.default_rng(4))
        other = benchmark._held_frame(frame, np.random.default_rng(5))

        assert np.array_equal(first, second)
        assert not np.array_equal(first, other)


class TestTraces:
    def test_traces_are_non_empty_and_well_formed(self, results):
        traces = results["traces"]

        assert traces["metric"] == "flow_norm"
        assert traces["threshold"] > 0
        assert traces["red_phase_player_traces"]
        for trace in traces["red_phase_player_traces"]:
            assert set(trace) == {"track_id", "kind", "t", "score"}
            assert trace["kind"] in ("moving", "frozen")
            assert trace["t"], "trace carries no timestamps"
            assert len(trace["t"]) == len(trace["score"])
            assert trace["t"] == sorted(trace["t"])
            assert all(score >= 0 for score in trace["score"])

    def test_the_lab_shows_both_a_walker_and_a_still_player(self, results):
        """A threshold lab with only walkers on it can be dragged to zero
        without ever showing the cost, so a held-frame trace has to be there
        to be dragged past.
        """
        traces = results["traces"]["red_phase_player_traces"]
        kinds = [trace["kind"] for trace in traces]

        assert kinds.count("moving") >= 1
        assert kinds.count("frozen") >= 1

    def test_the_still_player_is_cleared_by_the_shipped_threshold(self, results):
        threshold = results["traces"]["threshold"]
        frozen = [t for t in results["traces"]["red_phase_player_traces"] if t["kind"] == "frozen"]

        assert frozen
        for trace in frozen:
            assert max(trace["score"]) <= threshold, trace["track_id"]

    def test_frozen_and_moving_traces_cover_the_same_samples(self, results):
        """The contrast is only fair if it is the same player over the same
        segment — otherwise the lab is comparing two different moments.
        """
        traces = results["traces"]["red_phase_player_traces"]
        by_kind = {"moving": {}, "frozen": {}}
        for trace in traces:
            by_kind[trace["kind"]][trace["track_id"]] = trace["t"]

        for track_id, timestamps in by_kind["frozen"].items():
            assert track_id in by_kind["moving"]
            assert timestamps == by_kind["moving"][track_id]


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
