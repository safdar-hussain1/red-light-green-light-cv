"""Game configuration with validation."""

from dataclasses import dataclass


class ConfigError(ValueError):
    """Raised when game configuration is invalid."""

    pass


@dataclass(frozen=True)
class GameConfig:
    """Configuration for the game referee.

    Attributes:
        seed: Random seed for reproducibility. None for non-deterministic.
        countdown_s: Duration of countdown phase in seconds.
        duration_s: Total game duration in seconds.
        phase_min_s: Minimum duration for red/green phase in seconds.
        phase_max_s: Maximum duration for red/green phase in seconds.
        grace_s: Grace period in seconds.
        metric: Detection metric ("flow" for optical flow, "diff" for frame difference).
        threshold: Flow threshold (body fraction per second).
        diff_threshold: Diff threshold for frame difference detection. Provisional value pending measured benchmark.
        confirm_frames: Number of frames to confirm motion before elimination.
        smoothing: Temporal smoothing factor, in (0, 1].
        detector: Object detector to use ("yolo").
        conf: Confidence threshold for detector.
        max_misses: Maximum consecutive frames without detection before resetting.
    """

    seed: int | None = None
    countdown_s: float = 3.0
    duration_s: float = 60.0
    phase_min_s: float = 2.0
    phase_max_s: float = 5.0
    grace_s: float = 0.6
    metric: str = "flow"
    threshold: float = 0.12
    diff_threshold: float = 0.60
    confirm_frames: int = 3
    smoothing: float = 0.5
    detector: str = "yolo"
    conf: float = 0.35
    max_misses: int = 15

    def validate(self) -> None:
        """Validate configuration, raising ConfigError with all violations.

        Raises:
            ConfigError: If any field is invalid. The error message lists all violations.
        """
        violations = []

        # Check durations are non-negative
        if self.countdown_s < 0:
            violations.append(f"countdown_s must be non-negative, got {self.countdown_s}")
        if self.duration_s < 0:
            violations.append(f"duration_s must be non-negative, got {self.duration_s}")
        # Zero-length phases are meaningless for the game (and would spin the
        # state machine's phase-flip loop forever), so phase_min_s/phase_max_s
        # must be strictly positive, not merely non-negative.
        if self.phase_min_s <= 0:
            violations.append(f"phase_min_s must be positive, got {self.phase_min_s}")
        if self.phase_max_s <= 0:
            violations.append(f"phase_max_s must be positive, got {self.phase_max_s}")
        if self.grace_s < 0:
            violations.append(f"grace_s must be non-negative, got {self.grace_s}")

        # Check phase ordering
        if self.phase_min_s > self.phase_max_s:
            violations.append(
                f"phase_min_s ({self.phase_min_s}) must not exceed phase_max_s ({self.phase_max_s})"
            )

        # Check thresholds are positive
        if self.threshold <= 0:
            violations.append(f"threshold must be positive, got {self.threshold}")
        if self.diff_threshold <= 0:
            violations.append(f"diff_threshold must be positive, got {self.diff_threshold}")

        # Check smoothing is in (0, 1]
        if self.smoothing <= 0 or self.smoothing > 1:
            violations.append(
                f"smoothing must be in (0, 1], got {self.smoothing}"
            )

        # Check confirm_frames is at least 1
        if self.confirm_frames < 1:
            violations.append(
                f"confirm_frames must be at least 1, got {self.confirm_frames}"
            )

        # Check metric is valid
        valid_metrics = {"flow", "diff"}
        if self.metric not in valid_metrics:
            violations.append(
                f"metric must be one of {valid_metrics}, got {self.metric!r}"
            )

        # Check detector is valid
        valid_detectors = {"yolo"}
        if self.detector not in valid_detectors:
            violations.append(
                f"detector must be one of {valid_detectors}, got {self.detector!r}"
            )

        if violations:
            error_msg = "Configuration errors:\n" + "\n".join(f"  - {v}" for v in violations)
            raise ConfigError(error_msg)

    @property
    def active_threshold(self) -> float:
        """The threshold that actually applies, given `metric`.

        `threshold` and `diff_threshold` are on different scales (flow's
        body-fractions-per-second vs. diff's changed-pixel-fraction), so
        exactly one of them is ever the live cutoff a judge should compare
        scores against. Any caller that needs "the" threshold for the
        configured metric should read this instead of re-deriving the same
        `metric == "flow"` ternary.
        """
        return self.threshold if self.metric == "flow" else self.diff_threshold
