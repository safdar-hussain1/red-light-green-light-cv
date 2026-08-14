/**
 * The referee lab, and the exhibits for the designs that look obvious.
 *
 * Both are driven entirely from `RL_DATA.benchmark` — the same JSON
 * `redlight benchmark` writes. Nothing here has a number typed into it.
 *
 * The lab is the page's one genuinely open question made touchable. Eight
 * real per-player traces from a red light are plotted, five from players who
 * were moving and three from players who were frozen, and the cutoff is a
 * line the visitor drags. Every verdict is recomputed through the same
 * `MotionJudge` the arena runs, with the same hysteresis — three consecutive
 * confirmations over a 0.5-weighted moving average — so what is on screen is
 * not an illustration of the rule, it *is* the rule.
 *
 * The band drawn at the bottom is computed rather than asserted: for each
 * trace, the cutoff at which its verdict flips is found by bisection through
 * the real judge, and the lowest of those is where the safe band ends. That
 * is the honest form of "the threshold is not delicate" — a range, measured,
 * rather than a claim.
 */

/** Plot geometry, in the SVG's own units. */
const LAB_W = 660;
const LAB_H = 300;
const LAB_PAD = { left: 46, right: 18, top: 16, bottom: 42 };

/** Top of the score axis. Above every trace in the set, with headroom. */
const LAB_Y_MAX = 3.0;

/**
 * How far below zero the plot floor sits, in score units.
 *
 * The three frozen traces read exactly 0.00 for every sample, so drawn
 * against a floor of zero they lie under the frame edge and look like an
 * axis. Dropping the floor lifts them into the plot where they can be seen
 * to be data. The axis is still labelled from 0; nothing below it is drawn.
 */
const LAB_Y_FLOOR = -0.3;

/** Highest cutoff the drag and the slider reach. */
const LAB_T_MAX = 3.0;

/**
 * Replay one trace through the real judge and report where it is called.
 *
 * @returns {number|null} The timestamp of the call, or null if this player
 *   is never confirmed as moving at this cutoff.
 */
function verdictFor(trace, threshold, config) {
  const judge = new MotionJudge(threshold, config.confirmFrames, config.smoothing);
  for (let i = 0; i < trace.score.length; i += 1) {
    if (judge.update(trace.track_id, trace.score[i])) return trace.t[i];
  }
  return null;
}

/**
 * The cutoff at which this trace's verdict changes, found by bisection.
 *
 * Scores only ever push a verdict one way — raising the cutoff can turn a
 * call into a pass but never the reverse — so the flip point is unique and
 * bisection finds it. Forty halvings of a 0–5 range settles it far finer
 * than the plot can draw.
 */
function flipThreshold(trace, config) {
  let low = 0;
  let high = 5;
  if (verdictFor(trace, low, config) === null) return 0;
  for (let i = 0; i < 40; i += 1) {
    const mid = (low + high) / 2;
    if (verdictFor(trace, mid, config) === null) high = mid;
    else low = mid;
  }
  return low;
}

class RefereeLab {
  constructor(data) {
    this.config = data.config;
    this.traces = data.benchmark.traces;
    this.chosenFlow = data.benchmark.chosen_thresholds.flow;
    this.threshold = this.traces.threshold;

    this.mount = document.getElementById("labPlot");
    this.slider = document.getElementById("labSlider");
    this.tally = document.getElementById("labTally");
    this.headline = document.getElementById("labHeadline");
    this.bandNote = document.getElementById("labBand");
    if (!this.mount) return;

    this.safeUntil = Math.min.apply(
      null,
      this.traces.red_phase_player_traces
        .filter((trace) => trace.kind === "moving")
        .map((trace) => flipThreshold(trace, this.config))
    );

    this.tMax = Math.max.apply(
      null,
      this.traces.red_phase_player_traces.map((trace) => trace.t[trace.t.length - 1])
    );

    this.drawStatic();
    this.bindDrag();
    this.update();
  }

  x(t) {
    const span = LAB_W - LAB_PAD.left - LAB_PAD.right;
    return LAB_PAD.left + (t / this.tMax) * span;
  }

  y(score) {
    const span = LAB_H - LAB_PAD.top - LAB_PAD.bottom;
    const fraction = (Math.min(score, LAB_Y_MAX) - LAB_Y_FLOOR) / (LAB_Y_MAX - LAB_Y_FLOOR);
    return LAB_PAD.top + span - fraction * span;
  }

  scoreAt(clientY) {
    const rect = this.mount.querySelector("svg").getBoundingClientRect();
    const unitY = ((clientY - rect.top) / rect.height) * LAB_H;
    const span = LAB_H - LAB_PAD.top - LAB_PAD.bottom;
    const fraction = (LAB_PAD.top + span - unitY) / span;
    const score = LAB_Y_FLOOR + fraction * (LAB_Y_MAX - LAB_Y_FLOOR);
    return Math.max(Math.min(score, LAB_T_MAX), 0);
  }

  drawStatic() {
    const parts = [];
    parts.push(
      '<svg viewBox="0 0 ' + LAB_W + " " + LAB_H + '" role="img" ' +
        'aria-label="Eight recorded player traces during one red light, with a draggable cutoff">'
    );

    // The band every cutoff inside gives the same eight verdicts.
    parts.push(
      '<rect class="safe-band" x="' + LAB_PAD.left + '" y="' + this.y(this.safeUntil) +
        '" width="' + (LAB_W - LAB_PAD.left - LAB_PAD.right) +
        '" height="' + (this.y(0) - this.y(this.safeUntil)) + '"/>'
    );

    for (let i = 0; i <= 3; i += 1) {
      const value = (LAB_Y_MAX / 3) * i;
      parts.push(
        '<line class="axis-line" x1="' + LAB_PAD.left + '" y1="' + this.y(value) +
          '" x2="' + (LAB_W - LAB_PAD.right) + '" y2="' + this.y(value) + '"/>'
      );
      parts.push(
        '<text class="axis-text" x="' + (LAB_PAD.left - 8) + '" y="' + (this.y(value) + 3) +
          '" text-anchor="end">' + value.toFixed(1) + "</text>"
      );
    }
    for (let t = 0; t <= this.tMax + 0.01; t += 0.5) {
      parts.push(
        '<text class="axis-text" x="' + this.x(t) + '" y="' + (LAB_H - 10) +
          '" text-anchor="middle">' + t.toFixed(1) + "s</text>"
      );
    }
    parts.push(
      '<text class="axis-text" x="' + LAB_PAD.left + '" y="' + (LAB_PAD.top - 4) +
        '">body-fractions per second</text>'
    );

    // The cutoff the benchmark actually selected, as a fixed reference.
    parts.push(
      '<line class="measured-line" x1="' + LAB_PAD.left + '" y1="' + this.y(this.chosenFlow) +
        '" x2="' + (LAB_W - LAB_PAD.right) + '" y2="' + this.y(this.chosenFlow) + '"/>'
    );

    for (const trace of this.traces.red_phase_player_traces) {
      const points = trace.t
        .map((t, i) => this.x(t).toFixed(1) + "," + this.y(trace.score[i]).toFixed(1))
        .join(" ");
      parts.push(
        '<polyline class="trace-' + trace.kind + '" data-trace="' + trace.kind +
          "-" + trace.track_id + '" points="' + points + '"/>'
      );
    }

    // The frozen traces are three lines on top of each other at exactly the
    // same value. Say so, rather than leaving a reader to wonder why they
    // can only count one.
    parts.push(
      '<text class="axis-text frozen-note" x="' + (LAB_PAD.left + 6) + '" y="' +
        (this.y(0) + 15) + '">3 frozen players, all exactly 0.00, superimposed</text>'
    );

    parts.push('<g id="labCalls"></g>');
    parts.push(
      '<g id="labGrip"><line class="thresh-line" x1="' + LAB_PAD.left + '" y1="0" x2="' +
        (LAB_W - LAB_PAD.right) + '" y2="0"/>' +
        '<rect class="thresh-grip" x="' + (LAB_W - LAB_PAD.right - 74) +
        '" y="-11" width="74" height="22" rx="11"/>' +
        '<text class="thresh-grip-text" x="' + (LAB_W - LAB_PAD.right - 37) +
        '" y="4" text-anchor="middle"></text></g>'
    );
    parts.push("</svg>");

    this.mount.innerHTML = parts.join("");
    this.svg = this.mount.querySelector("svg");
    this.grip = this.svg.querySelector("#labGrip");
    this.calls = this.svg.querySelector("#labCalls");

    if (this.slider) {
      this.slider.min = "0";
      this.slider.max = String(LAB_T_MAX);
      this.slider.step = "0.005";
      this.slider.value = String(this.threshold);
      this.slider.addEventListener("input", () => {
        this.threshold = Number(this.slider.value);
        this.update();
      });
    }

    const reset = document.getElementById("labReset");
    if (reset) {
      reset.addEventListener("click", () => {
        this.threshold = this.chosenFlow;
        if (this.slider) this.slider.value = String(this.threshold);
        this.update();
      });
    }
  }

  bindDrag() {
    let dragging = false;
    const set = (event) => {
      this.threshold = this.scoreAt(event.clientY);
      if (this.slider) this.slider.value = String(this.threshold);
      this.update();
    };

    this.svg.addEventListener("pointerdown", (event) => {
      dragging = true;
      this.svg.setPointerCapture(event.pointerId);
      set(event);
      event.preventDefault();
    });
    this.svg.addEventListener("pointermove", (event) => {
      if (dragging) set(event);
    });
    this.svg.addEventListener("pointerup", (event) => {
      dragging = false;
      this.svg.releasePointerCapture(event.pointerId);
    });
    this.svg.addEventListener("pointercancel", () => {
      dragging = false;
    });
  }

  /**
   * Hovering a verdict lifts that player's line out of the tangle.
   *
   * Five moving traces share one colour — colour here encodes what a player
   * was actually doing, not which player they are — so picking one out by
   * eye is hard. Rather than spend four more hues on identity, the tally
   * beside the plot does the picking.
   */
  bindFocus() {
    for (const row of this.tally.querySelectorAll(".tally-row")) {
      const key = row.dataset.trace;
      const focus = () => {
        this.mount.dataset.focus = key;
        for (const line of this.svg.querySelectorAll("polyline")) {
          line.classList.toggle("is-focused", line.dataset.trace === key);
        }
      };
      const blur = () => {
        delete this.mount.dataset.focus;
        for (const line of this.svg.querySelectorAll("polyline")) {
          line.classList.remove("is-focused");
        }
      };
      row.addEventListener("mouseenter", focus);
      row.addEventListener("focusin", focus);
      row.addEventListener("mouseleave", blur);
      row.addEventListener("focusout", blur);
    }
  }

  update() {
    const y = this.y(this.threshold);
    this.grip.setAttribute("transform", "translate(0," + y.toFixed(2) + ")");
    this.grip.querySelector("text").textContent = this.threshold.toFixed(3);

    const rows = [];
    const dots = [];
    let movingCalled = 0;
    let movingTotal = 0;
    let frozenCalled = 0;
    let frozenTotal = 0;
    let earliest = null;

    for (const trace of this.traces.red_phase_player_traces) {
      const calledAt = verdictFor(trace, this.threshold, this.config);
      const moving = trace.kind === "moving";
      if (moving) {
        movingTotal += 1;
        if (calledAt !== null) movingCalled += 1;
      } else {
        frozenTotal += 1;
        if (calledAt !== null) frozenCalled += 1;
      }
      if (calledAt !== null && (earliest === null || calledAt < earliest)) earliest = calledAt;

      if (calledAt !== null) {
        const index = trace.t.indexOf(calledAt);
        dots.push(
          '<circle class="caught-dot" cx="' + this.x(calledAt).toFixed(1) + '" cy="' +
            this.y(trace.score[index]).toFixed(1) + '" r="5"/>'
        );
      }

      rows.push(
        '<div class="tally-row" data-trace="' + trace.kind + "-" + trace.track_id + '">' +
          '<span class="swatch swatch-' + trace.kind + '"></span>' +
          "<span>P" + trace.track_id + " · " + trace.kind + "</span>" +
          '<span class="' + (calledAt === null ? "verdict-safe" : "verdict-out") + '">' +
          (calledAt === null ? "safe" : "out " + calledAt.toFixed(1) + "s") +
          "</span></div>"
      );
    }

    this.calls.innerHTML = dots.join("");
    if (this.tally) {
      this.tally.innerHTML = rows.join("");
      this.bindFocus();
    }

    if (this.headline) {
      this.headline.innerHTML =
        movingCalled + "/" + movingTotal + " <small>moving players called</small>" +
        '<span class="split-read">' + frozenCalled + "/" + frozenTotal +
        " <small>frozen players called</small></span>";
    }

    if (this.bandNote) {
      this.bandNote.textContent =
        earliest === null
          ? "No player is called at this cutoff. Every cutoff from 0 up to " +
            this.safeUntil.toFixed(3) + " gives the same eight verdicts."
          : "Earliest call at " + earliest.toFixed(1) +
            "s. Every cutoff from 0 up to " + this.safeUntil.toFixed(3) +
            " gives the same eight verdicts — the shaded band.";
    }
  }
}

/* ------------------------------------------------ the designs that look obvious */

/**
 * The three exhibits, all reading from the published sweep.
 *
 * One control drives two of them: how large the player appears in the frame.
 * That is the axis the two size-sensitive designs fail along, and putting
 * both under the same slider is what makes the failure legible — the same
 * player, the same footage, three apparent sizes, and two designs that
 * change their mind about it.
 */
class NaiveExhibits {
  constructor(data) {
    this.sweep = data.benchmark.resolution_sweep;
    this.classifiers = data.benchmark.classifiers;
    this.background = data.benchmark.background_scenario;
    this.scale = "1x";

    this.control = document.getElementById("scaleControl");
    if (!this.control) return;

    for (const button of this.control.querySelectorAll("button")) {
      button.addEventListener("click", () => {
        this.scale = button.dataset.scale;
        for (const other of this.control.querySelectorAll("button")) {
          other.setAttribute("aria-pressed", other.dataset.scale === this.scale ? "true" : "false");
        }
        this.update();
      });
    }

    const auc = document.getElementById("bxAuc");
    if (auc) {
      auc.textContent = this.classifiers.brightness.auc.toFixed(4);
    }

    this.renderBackground();
    this.update();
  }

  set(id, text, flavour) {
    const node = document.getElementById(id);
    if (!node) return;
    node.querySelector(".v").textContent = text;
    if (flavour) node.dataset.flavour = flavour;
  }

  update() {
    const brightness = this.sweep.brightness[this.scale];
    const raw = this.sweep.raw_flow[this.scale];
    const normalised = this.sweep.flow_norm[this.scale];
    const diff = this.sweep.diff_norm[this.scale];

    this.set("bxFrozen", brightness.frozen_flagged.toFixed(2) + "%", "bad");
    this.set("bxMoving", brightness.moving_flagged.toFixed(2) + "%", "bad");
    this.set("rawMoving", raw.moving_flagged.toFixed(2) + "%", "bad");
    this.set("normMoving", normalised.moving_flagged.toFixed(2) + "%", "good");
    this.set("diffMoving", diff.moving_flagged.toFixed(2) + "%", "good");

    const spread = document.getElementById("rawSpread");
    if (spread) {
      const rawValues = ["0.5x", "1x", "1.667x"].map((k) => this.sweep.raw_flow[k].moving_flagged);
      const normValues = ["0.5x", "1x", "1.667x"].map((k) => this.sweep.flow_norm[k].moving_flagged);
      spread.textContent =
        "Across the three sizes, raw pixel flow swings " +
        rawValues[0].toFixed(0) + "% → " + rawValues[2].toFixed(0) +
        "% on the same moving players. Dividing by the box first holds it at " +
        normValues[0].toFixed(0) + "% → " + normValues[2].toFixed(0) + "%.";
    }
  }

  renderBackground() {
    const isolated = this.background.isolated;
    const occluded = this.background.occluded;

    const rows = [
      {
        label: "whole frame",
        value: isolated.frame_diff_area_flagged_pct,
        flavour: "bad",
      },
      {
        label: "per box, flow",
        value: isolated.per_box_flagged_pct.flow_norm,
        flavour: "good",
      },
      {
        label: "per box, diff",
        value: isolated.per_box_flagged_pct.diff_norm,
        flavour: "good",
      },
    ];

    const mount = document.getElementById("bgBars");
    if (mount) {
      mount.innerHTML = rows
        .map(
          (row) =>
            '<div class="barline-row" data-flavour="' + row.flavour + '">' +
            "<span>" + row.label + "</span>" +
            '<div class="barline-track"><div class="barline-fill" style="width:' +
            row.value.toFixed(2) + '%"></div></div>' +
            '<span class="barline-val">' + row.value.toFixed(1) + "%</span></div>"
        )
        .join("");
    }

    const first = document.getElementById("bgFirst");
    if (first) first.querySelector(".v").textContent = isolated.first_crossing_s.toFixed(1) + "s";

    const limitation = document.getElementById("bgLimitation");
    if (limitation) {
      limitation.textContent =
        "Cropping protects against traffic elsewhere in the shot, and nothing more. Move the " +
        "same still player into the busiest part of the frame and let walkers physically cross " +
        "their box, and the raw per-box score trips on " +
        occluded.per_box_flagged_pct_overlap.flow_norm.toFixed(2) +
        "% of the samples where somebody is actually in front of them (" +
        occluded.per_box_flagged_pct_overlap.diff_norm.toFixed(2) +
        "% for the diff metric). Somebody walking through your box is somebody moving inside " +
        "your box, and this referee cannot tell the difference.";
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  if (!window.RL_DATA) return;
  new RefereeLab(window.RL_DATA);
  new NaiveExhibits(window.RL_DATA);
});
