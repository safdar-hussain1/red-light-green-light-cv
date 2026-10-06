/**
 * The published results, drawn.
 *
 * Every number here is read out of `RL_DATA.benchmark` at run time; there is
 * no figure typed into this file. If the benchmark is re-run and the page
 * built again, the charts move on their own.
 *
 * Three decisions are deliberate and worth defending.
 *
 * **Red and green never appear in a chart.** On this page they mean the
 * light and only the light, and pink means something a visitor can press,
 * so the series palette is a fixed four-slot set — teal, amber, blue,
 * violet — kept apart from all three in both themes. Every chart carries
 * direct value labels and a table view, so nothing is gated behind picking
 * a colour out of a legend.
 *
 * **Runtime is two charts, not one.** A detector costs tens of milliseconds
 * a frame and a scoring metric costs fractions of one per player. Putting
 * both on a single axis would render the scoring bars as invisible slivers,
 * and putting them on two axes would be worse. They are different measures,
 * so they get different charts.
 *
 * **Frame-rate agreement is not a chart.** It is one number, 100%, and a
 * single number is a stat, not a plot.
 *
 * If the chart library fails to load the page is still complete: every chart
 * ships the same numbers as a table underneath it, and the section flips to
 * showing them.
 */

/** Series slots, in fixed order. Never cycled, never reassigned by rank. */
const SERIES_VARS = ["--s1", "--s2", "--s3", "--s4"];

/** Bars stay thin: the data is the only thing allowed to be loud. */
const MAX_BAR = 22;

/**
 * Round an axis maximum up to a number a person would have chosen.
 *
 * Chart.js will happily label an axis 0 / 50 / 100 / 187.5 if handed a raw
 * maximum, and a tick nobody would write by hand reads as a bug.
 */
function niceMax(value) {
  const magnitude = Math.pow(10, Math.floor(Math.log10(value)));
  for (const step of [1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10]) {
    if (value <= step * magnitude) return step * magnitude;
  }
  return 10 * magnitude;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function chartTheme() {
  return {
    series: SERIES_VARS.map(cssVar),
    ink: cssVar("--ink"),
    inkDim: cssVar("--ink-mute"),
    grid: cssVar("--rule"),
    surface: cssVar("--surface"),
    font: cssVar("--font-body"),
  };
}

/**
 * Direct value labels at the end of every bar.
 *
 * Chart.js draws no labels of its own, and the palette's two light slots owe
 * the reader relief from a colour-only read. Labels sit outside the bar end
 * so they never overflow the mark or get clipped by it.
 */
const valueLabels = {
  id: "valueLabels",
  afterDatasetsDraw(chart, args, options) {
    const { ctx } = chart;
    const suffix = options.suffix || "";
    const digits = options.digits === undefined ? 1 : options.digits;
    ctx.save();
    ctx.font = "600 10px " + chartTheme().font;
    ctx.fillStyle = chartTheme().inkDim;

    chart.data.datasets.forEach((dataset, index) => {
      const meta = chart.getDatasetMeta(index);
      if (meta.hidden) return;
      meta.data.forEach((element, i) => {
        const value = dataset.data[i];
        if (value === null || value === undefined) return;
        const text = Number(value).toFixed(digits) + suffix;
        if (chart.options.indexAxis === "y") {
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";
          ctx.fillText(text, element.x + 6, element.y);
        } else {
          ctx.textAlign = "center";
          ctx.textBaseline = "bottom";
          ctx.fillText(text, element.x, element.y - 5);
        }
      });
    });
    ctx.restore();
  },
};

function baseOptions(theme, { indexAxis, suffix, digits, max, axisTitle }) {
  return {
    indexAxis: indexAxis || "x",
    responsive: true,
    maintainAspectRatio: false,
    layout: { padding: { right: indexAxis === "y" ? 44 : 8, top: 14 } },
    animation: { duration: 400 },
    plugins: {
      legend: {
        display: false,
        labels: { color: theme.ink },
      },
      tooltip: {
        backgroundColor: theme.ink,
        titleColor: theme.surface,
        bodyColor: theme.surface,
        titleFont: { family: theme.font, size: 11 },
        bodyFont: { family: theme.font, size: 11 },
        padding: 8,
        displayColors: true,
        callbacks: {
          label(item) {
            return (
              item.dataset.label + ": " + Number(item.parsed[indexAxis === "y" ? "x" : "y"])
                .toFixed(digits === undefined ? 1 : digits) + (suffix || "")
            );
          },
        },
      },
      valueLabels: { suffix, digits },
    },
    scales: {
      x: {
        beginAtZero: true,
        max: indexAxis === "y" ? max : undefined,
        grid: {
          color: indexAxis === "y" ? theme.grid : "transparent",
          drawTicks: false,
        },
        border: { color: theme.grid },
        ticks: {
          color: theme.inkDim,
          font: { family: theme.font, size: 10 },
        },
        title: indexAxis === "y" && axisTitle
          ? { display: true, text: axisTitle, color: theme.inkDim, font: { family: theme.font, size: 10 } }
          : undefined,
      },
      y: {
        beginAtZero: true,
        max: indexAxis === "y" ? undefined : max,
        grid: { color: indexAxis === "y" ? "transparent" : theme.grid, drawTicks: false },
        border: { color: theme.grid },
        ticks: {
          color: theme.inkDim,
          font: { family: theme.font, size: 10 },
        },
        title: indexAxis !== "y" && axisTitle
          ? { display: true, text: axisTitle, color: theme.inkDim, font: { family: theme.font, size: 10 } }
          : undefined,
      },
    },
  };
}

/**
 * One series of bars: thin, rounded at the data end, square at the baseline.
 *
 * `barPercentage` leaves a gap of surface between neighbouring bars rather
 * than drawing a stroke around each one — separation without extra ink.
 */
function bar(colour, label, data) {
  return {
    label,
    data,
    backgroundColor: colour,
    borderColor: colour,
    borderRadius: 4,
    borderSkipped: "start",
    maxBarThickness: MAX_BAR,
    categoryPercentage: 0.72,
    barPercentage: 0.84,
  };
}

/** Human names for the five referee designs the benchmark scored. */
const CLASSIFIER_LABELS = {
  brightness: "Brightness",
  frame_diff_area: "Whole-frame diff",
  raw_flow: "Raw pixel flow",
  flow_norm: "Flow, per box",
  diff_norm: "Diff, per box",
};

const SWEEP_LABELS = {
  brightness: "Brightness",
  raw_flow: "Raw pixel flow",
  flow_norm: "Flow, per box",
  diff_norm: "Diff, per box",
};

class Results {
  constructor(data) {
    this.benchmark = data.benchmark;
    this.charts = [];
    this.renderTables();
    this.renderStats();
    this.render();

    window.addEventListener("rl-theme-change", () => this.render());
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    if (media.addEventListener) media.addEventListener("change", () => this.render());
  }

  render() {
    if (typeof window.Chart === "undefined") {
      document.getElementById("results").classList.add("charts-unavailable");
      return;
    }
    for (const chart of this.charts) chart.destroy();
    this.charts = [];

    const theme = chartTheme();
    this.classifierChart(theme);
    this.resolutionChart(theme);
    this.runtimeCharts(theme);
  }

  mount(id, config) {
    const canvas = document.getElementById(id);
    if (!canvas) return;
    this.charts.push(new Chart(canvas, config));
  }

  classifierChart(theme) {
    const names = Object.keys(CLASSIFIER_LABELS);
    this.mount("chartClassifiers", {
      type: "bar",
      data: {
        labels: names.map((name) => CLASSIFIER_LABELS[name]),
        datasets: [
          bar(theme.series[0], "Flagged while moving",
            names.map((name) => this.benchmark.classifiers[name].moving_flagged)),
          bar(theme.series[1], "Flagged while frozen",
            names.map((name) => this.benchmark.classifiers[name].frozen_flagged)),
        ],
      },
      options: baseOptions(theme, {
        indexAxis: "y",
        suffix: "%",
        digits: 1,
        max: 100,
        axisTitle: "percent of samples flagged",
      }),
      plugins: [valueLabels],
    });
    this.legend("legendClassifiers", [
      [theme.series[0], "Flagged while moving — catches"],
      [theme.series[1], "Flagged while frozen — false alarms"],
    ]);
  }

  resolutionChart(theme) {
    const names = Object.keys(SWEEP_LABELS);
    const scales = ["0.5x", "1x", "1.667x"];
    this.mount("chartResolution", {
      type: "bar",
      data: {
        labels: ["Half size", "As measured", "1.7x closer"],
        datasets: names.map((name, index) =>
          bar(theme.series[index], SWEEP_LABELS[name],
            scales.map((scale) => this.benchmark.resolution_sweep[name][scale].moving_flagged))
        ),
      },
      // No direct labels here. Twelve columns in three groups puts twelve
      // numbers within a few pixels of each other, and labels that collide
      // are worse than none — the legend, the tooltip and the table view
      // carry identity and value between them.
      options: baseOptions(theme, {
        suffix: "%",
        digits: 0,
        max: 100,
        axisTitle: "moving players flagged",
      }),
    });
    this.legend(
      "legendResolution",
      names.map((name, index) => [theme.series[index], SWEEP_LABELS[name]])
    );
  }

  runtimeCharts(theme) {
    const runtime = this.benchmark.runtime_ms;
    this.mount("chartDetector", {
      type: "bar",
      data: {
        labels: ["YOLO11n", "HOG"],
        datasets: [
          bar(theme.series[2], "Milliseconds per frame", [runtime.yolo11n, runtime.hog]),
        ],
      },
      options: baseOptions(theme, {
        indexAxis: "y",
        suffix: " ms",
        digits: 1,
        max: niceMax(runtime.hog * 1.25),
        axisTitle: "milliseconds per frame",
      }),
      plugins: [valueLabels],
    });

    this.mount("chartScoring", {
      type: "bar",
      data: {
        labels: ["Flow, per box", "Diff, per box"],
        datasets: [
          bar(theme.series[3], "Milliseconds per player", [
            runtime.flow_norm_per_player,
            runtime.diff_norm_per_player,
          ]),
        ],
      },
      options: baseOptions(theme, {
        indexAxis: "y",
        suffix: " ms",
        digits: 3,
        max: niceMax(runtime.flow_norm_per_player * 1.35),
        axisTitle: "milliseconds per player",
      }),
      plugins: [valueLabels],
    });
  }

  /**
   * A legend, in HTML rather than on the canvas.
   *
   * Two series or more always get one — colour is never the only channel —
   * and building it outside the canvas keeps it in the page's own type and
   * lets it wrap on a narrow screen instead of eating the plot.
   */
  legend(id, entries) {
    const mount = document.getElementById(id);
    if (!mount) return;
    mount.innerHTML = entries
      .map(
        ([colour, label]) =>
          '<span class="legend-key"><i style="background:' + colour + '"></i>' + label + "</span>"
      )
      .join("");
  }

  renderStats() {
    const fps = this.benchmark.fps_sweep.diff_norm;
    const mount = document.getElementById("fpsStats");
    if (!mount) return;
    mount.innerHTML = [
      ["Verdicts that agree", fps.decisions_agree_pct.toFixed(1) + "%"],
      ["Decisions compared", fps.decisions_compared.toLocaleString("en-US")],
      ["Pairs released, 10 fps", fps["10fps"].pairs_released.toLocaleString("en-US")],
      ["Pairs released, 30 fps", fps["30fps"].pairs_released.toLocaleString("en-US")],
      ["Interval mismatch", fps.dt_mismatch_pct.toFixed(1) + "%"],
    ]
      .map(
        ([key, value]) =>
          '<div class="stat"><dt class="k">' + key + '</dt><dd class="v">' + value +
          "</dd></div>"
      )
      .join("");
  }

  renderTables() {
    const classifiers = document.getElementById("tableClassifiers");
    if (classifiers) {
      const rows = Object.keys(CLASSIFIER_LABELS).map((name) => {
        const row = this.benchmark.classifiers[name];
        return (
          "<tr><td>" + CLASSIFIER_LABELS[name] + "</td><td>" + row.auc.toFixed(4) +
          "</td><td>" + row.moving_flagged.toFixed(2) + "%</td><td>" +
          row.frozen_flagged.toFixed(2) + "%</td><td>" +
          (row.scope === "per_box" ? "per box" : "whole frame") + "</td></tr>"
        );
      });
      classifiers.innerHTML =
        "<thead><tr><th>Design</th><th>AUC</th><th>Moving flagged</th>" +
        "<th>Frozen flagged</th><th>Scope</th></tr></thead><tbody>" +
        rows.join("") + "</tbody>";
    }

    const sweep = document.getElementById("tableResolution");
    if (sweep) {
      const rows = Object.keys(SWEEP_LABELS).map((name) => {
        const entry = this.benchmark.resolution_sweep[name];
        return (
          "<tr><td>" + SWEEP_LABELS[name] + "</td>" +
          ["0.5x", "1x", "1.667x"]
            .map(
              (scale) =>
                "<td>" + entry[scale].moving_flagged.toFixed(2) + "% / " +
                entry[scale].frozen_flagged.toFixed(2) + "%</td>"
            )
            .join("") +
          "</tr>"
        );
      });
      sweep.innerHTML =
        "<thead><tr><th>Design</th><th>Half size</th><th>As measured</th>" +
        "<th>1.7x closer</th></tr></thead><tbody>" + rows.join("") + "</tbody>" +
        '<caption class="visually-hidden">Moving flagged / frozen flagged, per apparent size</caption>';
    }

    // The fallbacks shown when the chart library did not arrive carry the
    // same figures, filled from the same data — not a second copy typed in.
    const runtime = this.benchmark.runtime_ms;
    const missingDetector = document.getElementById("missingDetector");
    if (missingDetector) {
      missingDetector.textContent =
        "The chart library did not load. YOLO11n runs at " +
        runtime.yolo11n.toFixed(1) + " ms per frame, HOG at " +
        runtime.hog.toFixed(1) + " ms.";
    }
    const missingScoring = document.getElementById("missingScoring");
    if (missingScoring) {
      missingScoring.textContent =
        "The chart library did not load. Flow costs " +
        runtime.flow_norm_per_player.toFixed(3) +
        " ms per player, the diff metric " +
        runtime.diff_norm_per_player.toFixed(3) + " ms.";
    }

    const meta = document.getElementById("benchMeta");
    if (meta) {
      const source = this.benchmark.meta;
      meta.textContent =
        source.frames_used + " frames at " + source.fps + " fps, " +
        source.pairs_sampled.toLocaleString("en-US") + " sampled pairs, " +
        source.n_moving.toLocaleString("en-US") + " moving and " +
        source.n_frozen.toLocaleString("en-US") + " frozen player-samples, " +
        source.detector + " detector at confidence " + source.detector_conf +
        ", sensor noise sigma " + source.noise_sigma + ", seed " + source.seed + ".";
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  if (!window.RL_DATA || !document.getElementById("results")) return;
  // The charts live on the measurements view, which starts hidden, and a
  // chart laid out inside a hidden element measures itself as zero by zero.
  // So they are built the first time that view is on screen, and a visitor
  // who never opens it never pays for them. Chart.js is deferred, so it is
  // present by then; the guard in `render` covers a CDN that never answered.
  let built = false;
  const build = () => {
    if (built) return;
    built = true;
    new Results(window.RL_DATA);
  };
  const view = document.getElementById("measurements");
  if (!view || view.getClientRects().length > 0) {
    build();
    return;
  }
  window.addEventListener("rl-view", (event) => {
    if (event.detail === "measurements") build();
  });
});
