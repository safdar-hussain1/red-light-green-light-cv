"""Result figures for the benchmark: what the numbers in
`reports/benchmark_results.json` look like.

Run with `PYTHONPATH=src python scripts/make_figures.py` from the repo
root. Every PNG lands in `reports/figures/`; the demo clip lands there too,
as `demo_match.gif` plus three HUD stills.

**ROC curves, not AUC bars.** The benchmark JSON stores rates and AUCs, not
the raw per-sample scores an ROC curve is drawn from, so this script
recomputes them — importing `redlight.benchmark`'s detect/track pass and
sample collector directly, on the same footage and the same seed the
benchmark itself uses (`benchmark.SEED`), which makes the rerun
deterministic. That recompute is the expensive part of this script: a
detect-and-track pass over the full 795-frame ground-truth video plus one
scoring pass, timed at well under a minute on a modern CPU — comfortably
inside the "a few minutes, once" budget a figure script gets. Given that,
plotting the honest curve beat plotting a bar chart of a number the reader
could already read straight off the JSON. `score_distributions.png` reuses
the same recomputed samples, so the cost is paid once for both figures.

**The frozen class is not a distribution.** Every metric here is scored
against a held frame with fresh sensor grain on top, and `diff_norm`'s
whole point is that grain that small never crosses its change threshold —
so the frozen `diff_norm` (and, on this footage, `flow_norm`) sample set is
every value equal to `0.0`. A log-x histogram cannot place a bar at zero,
and forcing one on with a fabricated bin would draw a distribution that
does not exist. `score_distributions.png` instead marks the frozen class
with a single labeled spike at zero and its sample count, which is what was
actually measured.

**Runtimes are read straight from the JSON**, not rerun — `runtime_ms`
documents in the benchmark module itself that it varies run to run by
design (it is measuring the machine as much as the code), so a script that
reran it would just publish a different number than the one everything else
here is consistent with.

**The demo clip is the real match.** `demo_match.gif` and the three HUD
stills come from one seeded run of `redlight.app.run` on the ground-truth
footage, using the exact `DEMO_CONFIG`/`DEMO_SKIP`/`DEMO_AUTO_START` the
benchmark module uses for its own demo. `redlight.hud.draw` is wrapped for
the duration of that one call so each rendered frame can be tagged with the
game phase it was drawn in — the only way to pick "a green-light frame" or
"the final banner frame" out of the sequence without re-deriving the light
schedule by hand — and the wrapper is restored immediately after. The video
itself is untouched: `app.run(..., record=...)` writes it exactly as it
would for any other caller.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "reports" / "benchmark_results.json"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

CLASSIFIER_METRICS = ("brightness", "frame_diff_area", "raw_flow", "flow_norm", "diff_norm")
SWEEP_METRICS = ("brightness", "raw_flow", "flow_norm", "diff_norm")
SCALE_ORDER = ("0.5x", "1x", "1.667x")

_METRIC_LABELS = {
    "brightness": "brightness_count",
    "frame_diff_area": "frame_diff_area",
    "raw_flow": "raw_flow_max",
    "flow_norm": "flow_norm (shipped)",
    "diff_norm": "diff_norm (shipped)",
}

_RUNTIME_LABELS = (
    ("yolo11n", "YOLO11n detection"),
    ("hog", "HOG detection (weights-free)"),
    ("flow_norm_per_player", "flow_norm per player"),
    ("diff_norm_per_player", "diff_norm per player"),
)

_DPI = 150
_FIGSIZE_WIDE = (10, 6)


# --------------------------------------------------------------------------
# pure data-reading helpers — safe to unit test, no plotting or recompute
# --------------------------------------------------------------------------


def load_results(path: Path = RESULTS_PATH) -> dict:
    """Read and parse `reports/benchmark_results.json`."""
    return json.loads(Path(path).read_text())


def chosen_thresholds(results: dict) -> dict[str, float]:
    """The shipped flow/diff thresholds, rounded to 4dp, as `GameConfig` ships them."""
    return {
        "flow": round(results["chosen_thresholds"]["flow"], 4),
        "diff": round(results["chosen_thresholds"]["diff"], 4),
    }


def runtime_rows(results: dict) -> list[tuple[str, float]]:
    """`(label, milliseconds)` for every `runtime_ms` entry, in display order."""
    runtime = results["runtime_ms"]
    return [(label, float(runtime[key])) for key, label in _RUNTIME_LABELS]


def resolution_sweep_rows(results: dict) -> list[dict]:
    """One row per (metric, scale): `moving_flagged`/`frozen_flagged`, flat.

    Flattening the JSON's nested `{metric: {scale: {...}}}` here means the
    plotting code just iterates rows instead of re-deriving the same nested
    walk, and it's the shape a test can assert over without caring how the
    JSON happens to be nested.
    """
    sweep = results["resolution_sweep"]
    rows = []
    for metric in SWEEP_METRICS:
        for scale in SCALE_ORDER:
            cell = sweep[metric][scale]
            rows.append(
                {
                    "metric": metric,
                    "scale": scale,
                    "moving_flagged": float(cell["moving_flagged"]),
                    "frozen_flagged": float(cell["frozen_flagged"]),
                }
            )
    return rows


# --------------------------------------------------------------------------
# the recompute: raw per-sample scores for ROC + distribution figures
# --------------------------------------------------------------------------


def _collect_raw_samples():
    """Rerun the benchmark's detect/track/score pass, for its raw scores.

    Same footage, same seed, same code path as `benchmark.run_benchmark` —
    only the sweeps, scenario, and runtime measurement are skipped, since
    ROC curves and score distributions need nothing from them. Deterministic
    given `benchmark.VIDEO` and `benchmark.SEED` are unchanged.
    """
    from redlight import benchmark

    boxes_by_frame, _frames_used = benchmark._track_pass(None)
    return benchmark._collect_samples(None, boxes_by_frame)


def build_roc_curves(samples, out_path: Path = FIGURES_DIR / "roc_curves.png") -> Path:
    """ROC curve for each of the 5 classifier metrics, on one axes."""
    from sklearn.metrics import roc_auc_score, roc_curve

    # Several of these designs separate the two classes almost perfectly on
    # this footage, so their curves all hug the top-left corner and would
    # sit exactly on top of each other with a uniform solid line. Varying
    # linestyle alongside color keeps them individually traceable even
    # where AUC rounds to 1.000 for more than one metric at once.
    _linestyles = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))

    fig, ax = plt.subplots(figsize=(7, 7))
    for metric, linestyle in zip(CLASSIFIER_METRICS, _linestyles):
        moving = np.asarray(samples.moving[metric], dtype=float)
        frozen = np.asarray(samples.frozen[metric], dtype=float)
        labels = np.concatenate([np.ones(len(moving)), np.zeros(len(frozen))])
        scores = np.concatenate([moving, frozen])
        fpr, tpr, _thresholds = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            linestyle=linestyle,
            label=f"{_METRIC_LABELS[metric]} (AUC {auc:.4f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1, label="chance")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("false positive rate (frozen flagged as moving)")
    ax.set_ylabel("true positive rate (moving flagged as moving)")
    ax.set_title("ROC: each design's separation of moving vs. frozen samples")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return out_path


def build_score_distributions(
    samples, out_path: Path = FIGURES_DIR / "score_distributions.png"
) -> Path:
    """Moving-score histograms for flow_norm/diff_norm, frozen shown as a labeled spike."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, metric in zip(axes, ("flow_norm", "diff_norm")):
        moving = np.asarray(samples.moving[metric], dtype=float)
        frozen = np.asarray(samples.frozen[metric], dtype=float)
        positive = moving[moving > 0]

        if len(positive) > 0:
            bins = np.logspace(
                np.log10(positive.min()), np.log10(positive.max()), 40
            )
            ax.hist(positive, bins=bins, color="#3b7dd8", alpha=0.85, label="moving")
            ax.set_xscale("log")

        zero_moving = int(np.count_nonzero(moving == 0))
        if zero_moving:
            ax.set_title(
                f"{_METRIC_LABELS[metric]}\n({zero_moving} moving samples at 0.0, not shown on log axis)",
                fontsize=10,
            )
        else:
            ax.set_title(_METRIC_LABELS[metric], fontsize=10)

        n_frozen_zero = int(np.count_nonzero(frozen == 0))
        n_frozen_nonzero = len(frozen) - n_frozen_zero
        # Frozen is drawn as a delta at the left edge with a count label,
        # rather than folded onto the log-x axis it cannot represent.
        left_edge = positive.min() if len(positive) else 1e-6
        ax.axvline(left_edge, color="#d84b3b", linewidth=2, label="frozen (spike)")
        ax.text(
            0.02,
            0.95,
            f"frozen: {n_frozen_zero}/{len(frozen)} at exactly 0.0"
            + (f", {n_frozen_nonzero} above 0.0" if n_frozen_nonzero else ""),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="#d84b3b",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#d84b3b", alpha=0.9),
        )
        ax.set_xlabel(f"{metric} score (log scale)")
        ax.set_ylabel("sample count")
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Moving vs. frozen score distributions (frozen is all-zero here)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# figures read straight from the JSON
# --------------------------------------------------------------------------


def build_resolution_sweep(
    results: dict, out_path: Path = FIGURES_DIR / "resolution_sweep.png"
) -> Path:
    """Grouped bars: moving_flagged & frozen_flagged per scale, one panel per metric."""
    rows = resolution_sweep_rows(results)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=False)

    width = 0.35
    x = np.arange(len(SCALE_ORDER))
    for ax, metric in zip(axes.flat, SWEEP_METRICS):
        by_scale = {row["scale"]: row for row in rows if row["metric"] == metric}
        moving = [by_scale[scale]["moving_flagged"] for scale in SCALE_ORDER]
        frozen = [by_scale[scale]["frozen_flagged"] for scale in SCALE_ORDER]

        bars_m = ax.bar(x - width / 2, moving, width, label="moving flagged %", color="#3b7dd8")
        bars_f = ax.bar(x + width / 2, frozen, width, label="frozen flagged %", color="#d84b3b")
        ax.bar_label(bars_m, fmt="%.1f", fontsize=7, padding=2)
        ax.bar_label(bars_f, fmt="%.1f", fontsize=7, padding=2)

        ax.set_title(_METRIC_LABELS[metric], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(SCALE_ORDER)
        ax.set_ylabel("% flagged")
        ax.set_ylim(0, 112)

    # A per-panel legend would sit inside the axes and collide with the
    # near-100% bars several of these metrics produce (that collision is
    # exactly what an earlier version of this figure shipped with), so
    # there's one legend for the whole figure, parked above the panels
    # where no bar ever reaches.
    fig.legend(
        *axes.flat[0].get_legend_handles_labels(),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.0),
        fontsize=9,
    )
    fig.suptitle(
        f"Resolution sweep: {results['resolution_sweep']['samples']} samples "
        "scored at each of 3 scales",
        y=1.045,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return out_path


def build_runtime(results: dict, out_path: Path = FIGURES_DIR / "runtime.png") -> Path:
    """Horizontal bars of every `runtime_ms` entry, on a log axis (values span ~3000x)."""
    rows = runtime_rows(results)
    labels = [label for label, _ms in rows]
    values = [ms for _label, ms in rows]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(rows))
    bars = ax.barh(y, values, color="#3b7dd8")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("milliseconds (log scale)")
    ax.set_title("Per-call runtime, median over warm calls")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() * 1.05,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3g} ms",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(min(values) * 0.5 if min(values) > 0 else 0.01, max(values) * 3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# the demo clip: real match, real HUD, tagged by phase for still selection
# --------------------------------------------------------------------------


def _run_demo_and_capture() -> tuple[Path, list[dict]]:
    """Run the seeded demo match, returning its recorded mp4 path and per-frame HUD captures.

    `redlight.hud.draw` is temporarily wrapped so each frame written by
    `app.run(..., record=...)` can be paired with the game phase it was
    drawn in. The wrapper is restored in a `finally` block regardless of
    outcome, and the mp4 itself is exactly what `app.run` would have
    produced unwrapped — the wrapper only observes, it does not alter the
    frame `draw` returns.
    """
    from redlight import app, benchmark, hud

    captures: list[dict] = []
    original_draw = hud.draw

    def _capturing_draw(frame, game, tracks, judge, config, now):
        rendered = original_draw(frame, game, tracks, judge, config, now)
        captures.append(
            {
                "phase": game.phase.name,
                "finished": game.finished,
                "phase_elapsed": game.phase_elapsed(now),
                "frame": rendered,
            }
        )
        return rendered

    tmp_mp4 = Path(tempfile.mkstemp(suffix=".mp4")[1])
    hud.draw = _capturing_draw
    try:
        app.run(
            benchmark.DEMO_CONFIG,
            benchmark.VIDEO,
            headless=True,
            record=str(tmp_mp4),
            auto_start_frames=benchmark.DEMO_AUTO_START,
            skip=benchmark.DEMO_SKIP,
            max_seconds=benchmark.DEMO_MAX_SECONDS,
        )
    finally:
        hud.draw = original_draw

    return tmp_mp4, captures


_GIF_FRAME_STRIDE = 3
_GIF_SCALE = 0.45
_GIF_FRAME_MS = 300  # 3x the source sample interval, matching the stride


def build_demo_gif_and_stills(out_dir: Path = FIGURES_DIR) -> dict[str, Path]:
    """The seeded demo match as an animated GIF, plus three representative HUD stills."""
    from PIL import Image

    from redlight.game import Phase

    tmp_mp4, captures = _run_demo_and_capture()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        pil_frames = []
        for i, capture in enumerate(captures):
            if i % _GIF_FRAME_STRIDE != 0:
                continue
            frame = capture["frame"]
            h, w = frame.shape[:2]
            small = cv2.resize(
                frame, (int(w * _GIF_SCALE), int(h * _GIF_SCALE)), interpolation=cv2.INTER_AREA
            )
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb))

        gif_path = out_dir / "demo_match.gif"
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=_GIF_FRAME_MS,
            loop=0,
            optimize=True,
        )

        stills = _pick_stills(captures)
        paths = {}
        for name, capture in stills.items():
            still_path = out_dir / f"hud_{name}.png"
            cv2.imwrite(str(still_path), capture["frame"])
            paths[name] = still_path

        return {"gif": gif_path, **paths}
    finally:
        tmp_mp4.unlink(missing_ok=True)


def _pick_stills(captures: list[dict]) -> dict[str, dict]:
    """One representative capture each for green light, red light, and match end."""
    from redlight.game import Phase

    green = [c for c in captures if c["phase"] == Phase.GREEN.name]
    red_armed = [
        c
        for c in captures
        if c["phase"] == Phase.RED.name and c["phase_elapsed"] > 0.0
    ]
    red_any = [c for c in captures if c["phase"] == Phase.RED.name]
    finished = [c for c in captures if c["finished"]]

    picks = {}
    if green:
        picks["green_light"] = green[len(green) // 2]
    if red_armed:
        picks["red_light"] = red_armed[int(len(red_armed) * 0.7)]
    elif red_any:
        picks["red_light"] = red_any[len(red_any) // 2]
    if finished:
        picks["match_over"] = finished[-1]
    return picks


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main() -> None:
    results = load_results()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Recomputing raw samples (detect/track/score pass over the full video)...")
    samples = _collect_raw_samples()

    print("Building roc_curves.png...")
    build_roc_curves(samples)
    print("Building score_distributions.png...")
    build_score_distributions(samples)
    print("Building resolution_sweep.png...")
    build_resolution_sweep(results)
    print("Building runtime.png...")
    build_runtime(results)
    print("Building demo_match.gif and HUD stills...")
    made = build_demo_gif_and_stills()
    for name, path in made.items():
        print(f"  {name}: {path} ({path.stat().st_size / 1024:.0f} KiB)")

    print(f"Done. Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    sys.exit(main())
