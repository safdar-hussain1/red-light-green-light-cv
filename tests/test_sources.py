"""Tests for FrameSource."""

import numpy as np
import pytest

from redlight.sources import FrameSource, SourceError


class TestFrameSourceErrors:
    """Opening a source that doesn't exist should fail loudly."""

    def test_nonexistent_path_raises_source_error(self):
        bad_path = "data/does_not_exist_at_all.avi"
        with pytest.raises(SourceError) as exc_info:
            FrameSource(bad_path)
        assert bad_path in str(exc_info.value)


class TestFrameSourceReadsSyntheticVideo:
    """Reading frames from a tiny synthetic video file."""

    def test_reads_all_frames_then_none(self, synthetic_video):
        with FrameSource(synthetic_video) as source:
            frames = []
            while True:
                result = source.read()
                if result is None:
                    break
                frames.append(result)
        assert len(frames) == 8
        for frame, ts in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (48, 64, 3)
            assert isinstance(ts, float)

    def test_read_past_end_returns_none(self, synthetic_video):
        with FrameSource(synthetic_video) as source:
            for _ in range(8):
                assert source.read() is not None
            assert source.read() is None
            # Calling again should stay None, not raise.
            assert source.read() is None

    def test_fps_reported(self, synthetic_video):
        with FrameSource(synthetic_video) as source:
            assert source.fps == pytest.approx(10.0)

    def test_timestamps_derived_from_frame_index_and_fps(self, synthetic_video):
        with FrameSource(synthetic_video) as source:
            timestamps = []
            while True:
                result = source.read()
                if result is None:
                    break
                timestamps.append(result[1])
        expected = [i / 10.0 for i in range(8)]
        for actual, exp in zip(timestamps, expected):
            assert actual == pytest.approx(exp)

    def test_timestamps_are_monotonic(self, synthetic_video):
        with FrameSource(synthetic_video) as source:
            timestamps = []
            while True:
                result = source.read()
                if result is None:
                    break
                timestamps.append(result[1])
        assert timestamps == sorted(timestamps)
        assert len(set(timestamps)) == len(timestamps)


class TestFrameSourceContextManager:
    """FrameSource must work as a context manager and release cleanly."""

    def test_release_can_be_called_explicitly(self, synthetic_video):
        source = FrameSource(synthetic_video)
        source.read()
        source.release()  # should not raise

    def test_context_manager_releases_on_exit(self, synthetic_video):
        with FrameSource(synthetic_video) as source:
            assert source.read() is not None
        # Source is released; no assertion beyond "no exception".
