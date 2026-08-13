"""Tests for GameConfig validation."""

import pytest

from redlight.config import ConfigError, GameConfig


class TestGameConfigDefaults:
    """Test that default configuration is valid."""

    def test_defaults_validate(self):
        """Defaults should create a valid config without raising."""
        config = GameConfig()
        config.validate()


class TestGameConfigNegativeDurations:
    """Test validation of duration fields."""

    def test_negative_countdown_s(self):
        """countdown_s cannot be negative."""
        config = GameConfig(countdown_s=-1.0)
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert "countdown_s" in str(exc_info.value)

    def test_negative_duration_s(self):
        """duration_s cannot be negative."""
        config = GameConfig(duration_s=-1.0)
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert "duration_s" in str(exc_info.value)

    def test_negative_phase_min_s(self):
        """phase_min_s cannot be negative."""
        config = GameConfig(phase_min_s=-1.0)
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert "phase_min_s" in str(exc_info.value)

    def test_negative_phase_max_s(self):
        """phase_max_s cannot be negative."""
        config = GameConfig(phase_max_s=-1.0)
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert "phase_max_s" in str(exc_info.value)

    def test_zero_phase_min_s(self):
        """phase_min_s cannot be zero: a zero-length phase is meaningless."""
        config = GameConfig(phase_min_s=0.0, phase_max_s=5.0)
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert "phase_min_s" in str(exc_info.value)

    def test_zero_phase_max_s(self):
        """phase_max_s cannot be zero: a zero-length phase is meaningless."""
        config = GameConfig(phase_min_s=0.0, phase_max_s=0.0)
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert "phase_max_s" in str(exc_info.value)

    def test_negative_grace_s(self):
        """grace_s cannot be negative."""
        config = GameConfig(grace_s=-1.0)
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert "grace_s" in str(exc_info.value)


class TestGameConfigPhaseOrder:
    """Test validation of phase duration relationships."""

    def test_phase_min_greater_than_phase_max(self):
        """phase_min_s cannot be greater than phase_max_s."""
        config = GameConfig(phase_min_s=10.0, phase_max_s=5.0)
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        error_msg = str(exc_info.value)
        assert "phase_min_s" in error_msg or "phase_max_s" in error_msg


class TestGameConfigThreshold:
    """Test validation of threshold fields."""

    def test_threshold_zero(self):
        """threshold cannot be zero."""
        config = GameConfig(threshold=0.0)
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert "threshold" in str(exc_info.value)

    def test_threshold_negative(self):
        """threshold cannot be negative."""
        config = GameConfig(threshold=-0.1)
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert "threshold" in str(exc_info.value)

    def test_diff_threshold_zero(self):
        """diff_threshold cannot be zero."""
        config = GameConfig(diff_threshold=0.0)
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert "diff_threshold" in str(exc_info.value)

    def test_diff_threshold_negative(self):
        """diff_threshold cannot be negative."""
        config = GameConfig(diff_threshold=-0.1)
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert "diff_threshold" in str(exc_info.value)


class TestGameConfigSmoothing:
    """Test validation of smoothing field."""

    def test_smoothing_zero(self):
        """smoothing must be in (0, 1], cannot be zero."""
        config = GameConfig(smoothing=0.0)
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert "smoothing" in str(exc_info.value)

    def test_smoothing_greater_than_one(self):
        """smoothing must be in (0, 1], cannot exceed 1.0."""
        config = GameConfig(smoothing=1.5)
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert "smoothing" in str(exc_info.value)

    def test_smoothing_negative(self):
        """smoothing must be in (0, 1], cannot be negative."""
        config = GameConfig(smoothing=-0.5)
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert "smoothing" in str(exc_info.value)

    def test_smoothing_valid_lower_bound(self):
        """smoothing can be just above zero."""
        config = GameConfig(smoothing=0.01)
        config.validate()

    def test_smoothing_valid_upper_bound(self):
        """smoothing can be exactly 1.0."""
        config = GameConfig(smoothing=1.0)
        config.validate()


class TestGameConfigConfirmFrames:
    """Test validation of confirm_frames field."""

    def test_confirm_frames_zero(self):
        """confirm_frames must be at least 1."""
        config = GameConfig(confirm_frames=0)
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert "confirm_frames" in str(exc_info.value)

    def test_confirm_frames_negative(self):
        """confirm_frames cannot be negative."""
        config = GameConfig(confirm_frames=-5)
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert "confirm_frames" in str(exc_info.value)


class TestGameConfigMetric:
    """Test validation of metric field."""

    def test_valid_metric_flow(self):
        """metric='flow' is valid."""
        config = GameConfig(metric="flow")
        config.validate()

    def test_valid_metric_diff(self):
        """metric='diff' is valid."""
        config = GameConfig(metric="diff")
        config.validate()

    def test_invalid_metric(self):
        """metric must be one of 'flow' or 'diff'."""
        config = GameConfig(metric="invalid")
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert "metric" in str(exc_info.value)


class TestGameConfigDetector:
    """Test validation of detector field."""

    def test_valid_detector_yolo(self):
        """detector='yolo' is valid."""
        config = GameConfig(detector="yolo")
        config.validate()

    def test_invalid_detector(self):
        """detector must be a valid value."""
        config = GameConfig(detector="invalid_detector")
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert "detector" in str(exc_info.value)


class TestGameConfigMultipleViolations:
    """Test that validate() reports ALL violations at once."""

    def test_multiple_violations_in_error_message(self):
        """validate() should list all violations in one error."""
        config = GameConfig(
            countdown_s=-1.0,  # bad
            phase_min_s=10.0,  # bad: > phase_max_s
            phase_max_s=5.0,
            threshold=0.0,  # bad
            smoothing=1.5,  # bad
            confirm_frames=0,  # bad
            metric="bad_metric",  # bad
        )
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        error_msg = str(exc_info.value)
        # Check that ALL expected violations are mentioned
        expected_violations = [
            "countdown_s",
            "phase_min_s",
            "threshold",
            "smoothing",
            "confirm_frames",
            "metric",
        ]
        for violation in expected_violations:
            assert violation in error_msg, (
                f"Expected violation '{violation}' not found in error message. "
                f"Got: {error_msg}"
            )

    def test_two_field_violations_reported_together(self):
        """validate() reports both violations when two fields are bad."""
        config = GameConfig(
            countdown_s=-1.0,  # bad
            threshold=0.0,  # bad
        )
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        error_msg = str(exc_info.value)
        # Both violations must be present in the error message
        assert "countdown_s" in error_msg, f"countdown_s not in: {error_msg}"
        assert "threshold" in error_msg, f"threshold not in: {error_msg}"
