"""Tests for the greedy IoU tracker. Pure geometry, no video I/O."""

from redlight.detection import Detection
from redlight.tracking import Tracker


class TestStableIdAcrossOverlappingBoxes:
    def test_same_id_when_box_overlaps_previous(self):
        tracker = Tracker()
        tracks = tracker.update([Detection(0, 0, 50, 50)])
        assert len(tracks) == 1
        first_id = tracks[0].track_id

        tracks = tracker.update([Detection(5, 5, 50, 50)])
        assert len(tracks) == 1
        assert tracks[0].track_id == first_id
        assert tracks[0].misses == 0


class TestIdSurvivesMissGap:
    def test_id_not_reused_after_short_miss_gap(self):
        tracker = Tracker(max_misses=15)
        tracks = tracker.update([Detection(0, 0, 50, 50)])
        original_id = tracks[0].track_id

        # Box vanishes for two frames.
        tracker.update([])
        tracker.update([])

        # Box reappears in roughly the same place.
        tracks = tracker.update([Detection(2, 2, 50, 50)])
        assert len(tracks) == 1
        assert tracks[0].track_id == original_id
        assert tracks[0].misses == 0


class TestNewIdForNonOverlappingBox:
    def test_non_overlapping_detection_gets_new_id(self):
        tracker = Tracker()
        first_tracks = tracker.update([Detection(0, 0, 10, 10)])
        first_id = first_tracks[0].track_id

        second_tracks = tracker.update([Detection(500, 500, 10, 10)])
        ids = {t.track_id for t in second_tracks}
        assert first_id in ids  # original track ages, still alive
        new_ids = ids - {first_id}
        assert len(new_ids) == 1
        assert next(iter(new_ids)) != first_id


class TestPopDead:
    def test_pop_dead_returns_id_after_max_misses_plus_one_empty_updates(self):
        tracker = Tracker(max_misses=2)
        tracks = tracker.update([Detection(0, 0, 10, 10)])
        track_id = tracks[0].track_id

        # max_misses + 1 empty updates should push it over the edge.
        for _ in range(3):
            tracker.update([])

        assert tracker.pop_dead() == {track_id}

    def test_pop_dead_is_empty_when_nothing_died(self):
        tracker = Tracker(max_misses=15)
        tracker.update([Detection(0, 0, 10, 10)])
        assert tracker.pop_dead() == set()

    def test_pop_dead_only_returns_ids_since_last_call(self):
        tracker = Tracker(max_misses=1)
        tracks = tracker.update([Detection(0, 0, 10, 10)])
        track_id = tracks[0].track_id
        for _ in range(2):
            tracker.update([])
        assert tracker.pop_dead() == {track_id}
        assert tracker.pop_dead() == set()


class TestIoUTieBrokenByHighestIoU:
    def test_detection_matches_track_with_highest_iou(self):
        tracker = Tracker(iou_threshold=0.3)
        # Two well-separated tracks established in frame 1.
        tracker.update([Detection(0, 0, 10, 10), Detection(9, 0, 10, 10)])

        # A single detection in frame 2 overlaps both prior boxes above
        # threshold, but more with the first (iou ~0.43) than the second
        # (iou ~0.33).
        tracks = tracker.update([Detection(4, 0, 10, 10)])

        matched = [t for t in tracks if t.misses == 0]
        assert len(matched) == 1
        assert matched[0].box == Detection(4, 0, 10, 10)

        aged = [t for t in tracks if t.misses > 0]
        assert len(aged) == 1
        assert aged[0].box == Detection(9, 0, 10, 10)

        # No new track was spawned; the detection went to an existing track.
        assert len(tracks) == 2


class TestMonotonicIds:
    def test_ids_start_at_one_and_increase(self):
        tracker = Tracker()
        t1 = tracker.update([Detection(0, 0, 10, 10)])[0]
        t2 = tracker.update([Detection(500, 500, 10, 10)])
        new = [t for t in t2 if t.track_id != t1.track_id][0]
        assert t1.track_id == 1
        assert new.track_id == 2
