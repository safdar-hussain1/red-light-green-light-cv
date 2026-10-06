/**
 * The playable arena: a frame in, verdicts out.
 *
 * This file is the wiring, not the rules. The rules live in `judge.js` (the
 * scoring kernel, pinned to Python by golden fixtures) and `game.js` (the
 * phase machine, pinned by the same parity suite). What happens here is the
 * part a browser has to do for itself: get a frame, find the players, cut
 * their boxes into 96x96 windows, pace those windows onto the referee's
 * fixed 0.1 s clock, and turn the events that come back into something a
 * person can read while standing still in front of a laptop.
 *
 * One stage shows three things, and only one at a time:
 *
 * - **The demo match** (`demo.js`), which opens the page: drawn players on a
 *   canvas, refereed by everything below exactly as a camera match is. It
 *   pauses whenever the stage is off screen.
 * - **Your camera**, after Play: the webcam, the pose model, the real match.
 * - **The recorded match**: the Python engine's own annotated footage, with
 *   the light, the doll, the timer and the players kept in step with it.
 *
 * A few details are worth knowing before changing anything.
 *
 * **One sampler, not one per player.** The referee's clock is a property of
 * the match, not of a person: every player is compared across the same slice
 * of real time, so nobody is judged over a shorter interval because their
 * box appeared late. The sampler therefore holds a map of every player's
 * window and releases the whole map at once.
 *
 * **Judge and sampler reset on every phase change**, exactly as `app.run`
 * does in the desktop engine. A player who was sprinting as the light turned
 * must start the red light as still as everyone else, and the first interval
 * of a new phase must not be measured against a frame from the old one.
 *
 * **The clock is `performance.now()`**, which is monotonic. A wall clock can
 * step backwards — an NTP correction mid-match is enough — and a negative dt
 * would make `diffScoreWindow` throw rather than quietly score.
 *
 * **The window a player is scored through is not the box drawn around them.**
 * `BoxStabilizer` smooths it, and pins it still for the countdown and every
 * red light, so detector wobble cannot be mistaken for a player moving.
 * `site/pose.js` explains the mechanism at length.
 */

/**
 * Multipliers on the lab-measured cutoff, offered as difficulty.
 *
 * The measured number — the midpoint between the walking class and the held
 * class on the benchmark footage — is 1.0x, and it is offered as *Ruthless*
 * rather than as the default. It was measured on stable crops of people
 * walking continuously, and neither half of that describes a living room: a
 * real player breathes, sways, and is lit by whatever is in the room. So the
 * default carries real headroom over the lab number, and the page says which
 * is which instead of implying the lab number is the natural setting.
 */
const DIFFICULTY = Object.freeze({
  forgiving: 4.0,
  standard: 2.0,
  ruthless: 1.0,
});

/** What each difficulty means, in the words a player needs. */
const DIFFICULTY_NOTES = Object.freeze({
  forgiving: "Forgiving: small fidgets are allowed.",
  standard: "Standard: leaves room for breathing and camera noise.",
  ruthless: "Ruthless: the strictest line, measured on the benchmark footage.",
});

/** Headroom over the noise floor measured on the player's own camera. */
const CAL_MARGIN = 3.0;

/** Baseline samples a player needs before their camera has been measured. */
const CAL_MIN_SAMPLES = 3;

/** How long at least one player must be in frame before the countdown opens. */
const REGISTRATION_HOLD_S = 1.0;

/** Samples kept per player, for the trace shown when they are called out. */
const TRACE_MEMORY = 40;

/** Meter full-scale, as a multiple of the live threshold. */
const METER_SCALE = 2.0;

/**
 * The demo match's schedule: a short match, so a visitor sees a whole one,
 * calls and result included, in about twenty seconds. The seed fixes the
 * lengths of the lights, so the demo plays the same way on every visit.
 */
const DEMO_MATCH = Object.freeze({
  seed: 11,
  countdownS: 1.5,
  durationS: 14.0,
  phaseMinS: 1.7,
  phaseMaxS: 2.6,
});

/** How long the demo's result stays up before the next demo match starts. */
const DEMO_RESTART_MS = 5200;

/** How long the card explaining a call stays up. */
const WHY_CARD_MS = 3800;

/**
 * When each light started in the recorded demo match, in match seconds.
 *
 * `docs/demo_match.mp4` is the seeded demo run (seed 3, lights of 1.5 to
 * 3.0 s). The browser's random number generator is not Python's, so the
 * schedule cannot be drawn again here from the seed; it is written down
 * instead, read from that run: each value is the time of the first frame of
 * a new light minus the game's own time-into-phase on that frame. The first
 * green starts at 0 and the lights alternate from there.
 */
const REPLAY_LIGHT_STARTS_S = Object.freeze([1.857, 4.173, 6.228, 8.634, 11.073]);

/**
 * The replay schedule for one match, in seconds.
 *
 * Green from 0, then alternating at the recorded starts. The last slot's
 * `to` is exactly `duration`, so any `matchT` inside `[0, duration)` — the
 * only range the caller ever looks up — is covered.
 */
function buildReplaySchedule(duration) {
  const edges = [0, ...REPLAY_LIGHT_STARTS_S.filter((t) => t < duration), duration];
  const slots = [];
  for (let i = 0; i + 1 < edges.length; i += 1) {
    slots.push({ phase: i % 2 === 0 ? "GREEN" : "RED", from: edges[i], to: edges[i + 1] });
  }
  return slots;
}

/** How long the arena holds its cold, frozen look at a green-to-red snap. */
const FREEZE_MS = 260;

/**
 * True when the visitor has asked the platform for less movement.
 *
 * The stylesheet flattens every transition and animation on its own. This is
 * for the effects CSS cannot reason about — a demo match that plays by
 * itself, a full-frame flash, a burst of thrown paper, a box breaking into
 * shards — which are not worth flattening, only worth skipping.
 */
function lessMotion() {
  return (
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

/* ------------------------------------------------------------------ theme */

/**
 * The theme toggle.
 *
 * The stored choice is applied by a tiny script in the document head, before
 * the first paint, so a visitor who chose light never sees a dark flash on
 * reload. All this does is flip it and write it down.
 */
function initTheme() {
  const button = document.getElementById("themeToggle");
  if (!button) return;

  const isDark = () =>
    document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;

  function label() {
    const text = isDark() ? "Switch to the light theme" : "Switch to the dark theme";
    button.setAttribute("aria-label", text);
    button.title = text;
  }

  button.addEventListener("click", () => {
    const next = isDark() ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      window.localStorage.setItem("rl-theme", next);
    } catch (err) {
      // Private mode, or storage disabled. The toggle still works for this
      // page view; only the memory of it is lost.
    }
    label();
    window.dispatchEvent(new CustomEvent("rl-theme-change", { detail: next }));
  });

  label();
}

/* -------------------------------------------------------------- the light */

/** What the panel says for each phase: the light, and what it asks of you. */
const LIGHT_WORDS = Object.freeze({
  LOBBY: ["Lobby", "Waiting for players"],
  COUNTDOWN: ["Get ready", "Stand still"],
  GREEN: ["Green light", "Move"],
  RED: ["Red light", "Freeze"],
  VICTORY: ["Victory", "The clock ran out"],
  WIPEOUT: ["Wipeout", "Nobody left standing"],
});

/**
 * Show one light everywhere it appears: the sign in the headline, the small
 * lamp in the top bar, and the big lamp behind the doll. Three places, one
 * light — a sign saying "red light" over a green stage would be a bug a
 * reader could see.
 */
function showLight(phase) {
  const sign = document.getElementById("sign");
  if (sign) {
    sign.dataset.lit =
      phase === "GREEN" || phase === "VICTORY"
        ? "green"
        : phase === "RED" || phase === "WIPEOUT"
          ? "red"
          : "none";
  }
  for (const lamp of document.querySelectorAll(".brand-lamp, #lamp")) {
    lamp.dataset.phase = phase;
  }
}

/* -------------------------------------------------------------- utilities */

function el(id) {
  return document.getElementById(id);
}

function fmt(value, digits = 2) {
  return Number(value).toFixed(digits);
}

/** A player's number as it is printed on their patch: 1 is "001". */
function bib(trackId) {
  return String(trackId).padStart(3, "0");
}

/**
 * Linearly interpolated percentile of a set of samples.
 *
 * Used on the countdown's baseline scores, where the point of reaching for a
 * p95 rather than a max is that one cough, one passing car headlight or one
 * dropped frame should not set the cutoff for the whole match.
 */
function percentile(values, q) {
  if (values.length === 0) return 0;
  const sorted = Array.from(values).sort((a, b) => a - b);
  const rank = (sorted.length - 1) * q;
  const low = Math.floor(rank);
  const high = Math.ceil(rank);
  if (low === high) return sorted[low];
  return sorted[low] + (sorted[high] - sorted[low]) * (rank - low);
}

/**
 * A player's recent scores as a sparkline, with the threshold drawn across.
 *
 * This is the "why" behind a call: not a verdict restated, but the actual
 * numbers the judge was given, and the line they had to stay under.
 */
function sparkline(samples, threshold) {
  const width = 240;
  const height = 40;
  if (samples.length < 2) return "";

  const top = Math.max(threshold * 1.6, ...samples) || 1;
  const step = width / (samples.length - 1);
  const points = samples
    .map((score, index) => {
      const x = index * step;
      const y = height - Math.min(score / top, 1) * (height - 4) - 2;
      return fmt(x, 1) + "," + fmt(y, 1);
    })
    .join(" ");
  const thresholdY = height - Math.min(threshold / top, 1) * (height - 4) - 2;

  return [
    '<svg viewBox="0 0 ' + width + " " + height + '" preserveAspectRatio="none" aria-hidden="true">',
    '<line x1="0" y1="' + fmt(thresholdY, 1) + '" x2="' + width + '" y2="' + fmt(thresholdY, 1) +
      '" stroke="currentColor" stroke-width="1" stroke-dasharray="4 3"/>',
    '<polyline points="' + points + '" fill="none" stroke="var(--red-lit)" stroke-width="2" ',
    'stroke-linejoin="round" stroke-linecap="round"/>',
    "</svg>",
  ].join("");
}

/* ------------------------------------------------------------ timer board */

/** Which of the seven segments each character lights. */
const SEGMENTS = Object.freeze({
  0: "abcdef",
  1: "bc",
  2: "abged",
  3: "abgcd",
  4: "fgbc",
  5: "afgcd",
  6: "afgedc",
  7: "abc",
  8: "abcdefg",
  9: "abcdfg",
  "-": "g",
  " ": "",
});

/**
 * The arena's timer board: red seven-segment digits, drawn as SVG.
 *
 * Each segment is a six-sided bar, so the unlit ones stay faintly visible
 * the way a real board's do, and the digits never reflow as they change.
 */
class SevenSegment {
  constructor(mount, digits = 2) {
    this.mount = mount;
    this.digits = digits;
    this.cells = [];
    if (!mount) return;

    const W = 40;
    const H = 70;
    const T = 7;
    const GAP = 14;
    const across = (yc) => {
      const x1 = 5;
      const x2 = W - 5;
      return [
        [x1, yc],
        [x1 + T / 2, yc - T / 2],
        [x2 - T / 2, yc - T / 2],
        [x2, yc],
        [x2 - T / 2, yc + T / 2],
        [x1 + T / 2, yc + T / 2],
      ];
    };
    const down = (xc, y1, y2) => [
      [xc, y1],
      [xc + T / 2, y1 + T / 2],
      [xc + T / 2, y2 - T / 2],
      [xc, y2],
      [xc - T / 2, y2 - T / 2],
      [xc - T / 2, y1 + T / 2],
    ];
    const shapes = {
      a: across(T / 2),
      g: across(H / 2),
      d: across(H - T / 2),
      f: down(T / 2, 5, H / 2 - 1.5),
      b: down(W - T / 2, 5, H / 2 - 1.5),
      e: down(T / 2, H / 2 + 1.5, H - 5),
      c: down(W - T / 2, H / 2 + 1.5, H - 5),
    };

    const total = digits * W + (digits - 1) * GAP;
    const parts = ['<svg viewBox="-2 -2 ' + (total + 4) + " " + (H + 4) + '">'];
    for (let i = 0; i < digits; i += 1) {
      parts.push('<g transform="translate(' + i * (W + GAP) + ' 0) skewX(-6)">');
      for (const [name, points] of Object.entries(shapes)) {
        parts.push(
          '<polygon class="seg" data-seg="' + name + '" points="' +
            points.map(([x, y]) => x + "," + y).join(" ") + '"/>'
        );
      }
      parts.push("</g>");
    }
    parts.push("</svg>");
    mount.innerHTML = parts.join("");
    this.cells = Array.from(mount.querySelectorAll("g"));
    this.shown = null;
  }

  /** Show up to `digits` characters, right-aligned. */
  set(text) {
    if (!this.mount) return;
    const value = String(text).slice(-this.digits).padStart(this.digits, " ");
    if (value === this.shown) return;
    this.shown = value;
    this.cells.forEach((cell, index) => {
      const lit = SEGMENTS[value[index]] || "";
      for (const seg of cell.children) {
        seg.classList.toggle("on", lit.includes(seg.dataset.seg));
      }
    });
  }
}

/* ---------------------------------------------------------------- probing */

/**
 * A read-out of what the arena did, for a headless browser to assert on.
 *
 * Only active under `?probe=1`. `scripts/verify_site.py` drives the play
 * path, the replay and the demo match and reads the result out of the page
 * title, which is the one piece of state a `--dump-dom` run can see without
 * a screenshot or a console scrape.
 */
const Probe = {
  on: new URLSearchParams(window.location.search).get("probe") === "1",
  seen: [],
  note(what) {
    if (!this.on) return;
    if (this.seen[this.seen.length - 1] === what) return;
    this.seen.push(what);
    document.title = "PROBE " + this.seen.join(">");
  },
};

/* ------------------------------------------------------------------ arena */

/** What the stage says it is showing, and the line of text under it. */
const MODE_TEXT = Object.freeze({
  demo: [
    "Demo match",
    "Drawn players walk on green and freeze on red. The boxes, movement " +
      "meters and calls come from the real referee; only the pose model is skipped.",
  ],
  camera: [
    "Your camera",
    "Stand back until your whole body is in frame, then play by the light. " +
      "Up to four players at once.",
  ],
  replay: [
    "Recorded match",
    "The Python engine refereeing public courtyard footage. Every box, " +
      "label and call in the video is the engine's own, not staged.",
  ],
  idle: ["", ""],
});

/**
 * Owns the stage: the frame source, the pose source, the judge, the game,
 * and everything drawn over and beside them.
 *
 * Built once on page load. Nothing is requested and no camera is touched
 * until `play()` is called from a click; the demo match needs neither.
 */
class Arena {
  constructor(data) {
    this.data = data;
    this.config = data.config;

    this.stage = el("stage");
    this.video = el("cam");
    this.overlay = el("overlay");
    this.flash = el("flash");
    this.lightsOut = el("lightsOut");
    this.fx = el("fx");
    this.idle = el("stageIdle");
    this.whyCard = el("whyCard");
    this.endCard = el("endCard");
    this.roster = el("roster");
    this.status = el("arenaStatus");

    this.doll = window.createDoll ? createDoll(el("arenaDoll")) : null;
    this.chant = new ChantPlayer(data.chant);
    this.timer = new SevenSegment(el("timerDigits"), 2);
    this.scene = typeof createDemoScene === "function" ? createDemoScene(el("demoCanvas")) : null;

    /** "demo", "camera", "replay", or "idle" while nothing is running. */
    this.mode = "idle";
    /** What frames are read from: the camera video or the demo canvas. */
    this.source = null;
    /** The camera is shown mirrored, so its boxes are drawn mirrored too. */
    this.mirror = false;
    this.matchConfig = DEMO_MATCH;
    this.demoPaused = false;
    /** Whether the demo may start, pause and restart by itself. */
    this.autoplay = false;
    /**
     * Moves on with every `halt()`. A camera start awaits the permission
     * prompt and then the pose model; if anything else has happened by the
     * time either answers, the counter has moved and that start is abandoned
     * instead of taking over the stage.
     */
    this.runId = 0;

    this.difficulty = "standard";
    this.threshold = this.config.diffThreshold * DIFFICULTY.standard;
    /** Noise floor measured on this camera during the countdown, or null. */
    this.calibration = null;
    /** Baseline scores collected per player while the countdown runs. */
    this.baselines = new Map();
    this.calibrationDone = false;
    this.judge = new MotionJudge(
      this.threshold,
      this.config.confirmFrames,
      this.config.smoothing
    );
    this.sampler = new FrameSampler(this.config.sampleIntervalS);
    this.tracker = new BoxTracker();
    this.stabilizer = new BoxStabilizer();
    this.game = null;
    this.poseSource = null;
    this.stream = null;
    this.rafId = null;
    this.registrationSince = null;
    this.traces = new Map();
    this.rows = new Map();
    this.replayTimer = null;
    this.replayVideo = null;
    /** Cancels a confetti burst still in flight, or null. */
    this.stopConfetti = null;
    this.freezeTimer = null;
    this.whyTimer = null;
    this.restartTimer = null;

    const canvas = document.createElement("canvas");
    canvas.width = 96;
    canvas.height = 96;
    // `willReadFrequently` is the difference between a readback that stays
    // on the CPU and one that stalls the pipeline every frame; the arena
    // does exactly one `getImageData` per player per frame.
    this.cropCtx = canvas.getContext("2d", { willReadFrequently: true });
  }

  /* ------------------------------------------------------------ lifecycle */

  /**
   * The polite live region. The demo writes nothing to it: a match nobody
   * started, announced every twenty seconds, would be noise to anyone
   * listening to the page.
   */
  setStatus(text) {
    if (this.mode === "demo") return;
    if (this.status) this.status.textContent = text;
  }

  setMode(mode) {
    this.mode = mode;
    this.stage.dataset.mode = mode;
    const frame = el("console");
    if (frame) frame.dataset.mode = mode;
    const [badge, caption] = MODE_TEXT[mode] || MODE_TEXT.idle;
    const badgeText = el("stageBadgeText");
    if (badgeText) badgeText.textContent = badge;
    const badgeNode = el("stageBadge");
    if (badgeNode) badgeNode.hidden = !badge;
    const captionNode = el("stageCaption");
    if (captionNode && caption) captionNode.textContent = caption;
    const stop = el("btnStop");
    if (stop) stop.hidden = mode !== "camera";
    // Play is only ever held down while a camera match is starting; leaving
    // for any other mode, even halfway through that start, releases it.
    const play = el("btnPlay");
    if (play && mode !== "camera") play.disabled = false;
    // The stage keeps a 4:3 shape unless a camera says otherwise.
    if (mode !== "camera") this.stage.style.aspectRatio = "";
    this.renderThresholdNotes();
  }

  /**
   * Restart a one-shot effect on an element, tagged with what kind it is.
   *
   * The reflow between the two writes is load-bearing: without it a second
   * call while the animation is still running does nothing, and the case
   * that matters — two players called out a few frames apart — is exactly
   * that case.
   */
  fire(node, kind) {
    if (!node) return;
    node.dataset.on = "false";
    node.dataset.kind = kind;
    void node.offsetWidth;
    node.dataset.on = "true";
  }

  /**
   * The green-to-red snap, felt rather than read.
   *
   * A camera flash and a quarter of a second where the stage goes cold and
   * stops looking like a live picture. This is the one moment in a match a
   * player has to react to, so it is the one moment the page interrupts
   * itself. Skipped outright under reduced motion — a full-frame flash is
   * precisely the effect that setting exists to turn off.
   */
  snapToRed() {
    if (lessMotion()) return;
    this.fire(this.flash, "snap");
    this.stage.dataset.freeze = "true";
    if (this.freezeTimer !== null) window.clearTimeout(this.freezeTimer);
    this.freezeTimer = window.setTimeout(() => {
      delete this.stage.dataset.freeze;
      this.freezeTimer = null;
    }, FREEZE_MS);
  }

  /** Put the stage back to a state a new match can start from. */
  resetStage() {
    delete this.stage.dataset.outcome;
    delete this.stage.dataset.freeze;
    delete this.stage.dataset.armed;
    if (this.lightsOut) this.lightsOut.dataset.on = "false";
    if (this.flash) this.flash.dataset.on = "false";
    if (this.stopConfetti) {
      this.stopConfetti();
      this.stopConfetti = null;
    }
  }

  setPhase(phase) {
    const previous = this.stage.dataset.phase;
    this.stage.dataset.phase = phase;
    if (previous === "GREEN" && phase === "RED") this.snapToRed();
    const [word, sub] = LIGHT_WORDS[phase] || LIGHT_WORDS.LOBBY;
    const pill = el("phasePill");
    if (pill) pill.dataset.phase = phase;
    const wordNode = el("phaseWord");
    if (wordNode) wordNode.textContent = word;
    this.setPhaseSub(sub);
    showLight(phase);
    if (this.doll) this.doll.setPhase(phase);
    Probe.note(phase);
  }

  setPhaseSub(text) {
    const sub = el("phaseSub");
    if (sub) sub.textContent = text;
  }

  /**
   * Stop whatever the stage is doing and clear it.
   *
   * Every mode starts here, because two of them running at once — a camera
   * loop still writing boxes over a replay, or a demo player walking through
   * a webcam match — would be indistinguishable from a bug.
   */
  halt() {
    this.runId += 1;
    if (this.rafId !== null) cancelAnimationFrame(this.rafId);
    this.rafId = null;
    this.safeAudio(() => this.chant.stop());
    this.releaseCamera();
    if (this.poseSource && this.poseSource !== this.scene && this.poseSource.close) {
      try {
        this.poseSource.close();
      } catch (err) {
        // A runtime that will not close is still a runtime we are done with.
      }
    }
    this.poseSource = null;
    this.source = null;
    this.stopReplay();
    for (const name of ["whyTimer", "restartTimer"]) {
      if (this[name] !== null) window.clearTimeout(this[name]);
      this[name] = null;
    }
    this.resetStage();
    this.overlay.innerHTML = "";
    this.roster.innerHTML = "";
    delete this.roster.dataset.compact;
    this.rows.clear();
    this.endCard.dataset.show = "false";
    this.whyCard.dataset.show = "false";
    if (this.idle) this.idle.hidden = true;
    this.game = null;
    this.demoPaused = false;
    // Whatever the last match measured belongs to that match; a camera that
    // is only being asked for has not been measured yet.
    this.baselines.clear();
    this.calibration = null;
    this.calibrationDone = false;
    this.applyThreshold();
    this.timer.set("--");
    const label = el("timerLabel");
    if (label) label.textContent = "waiting";
    this.setPhase("LOBBY");
  }

  /** Start a fresh match on whatever source and pose source are set. */
  beginMatch() {
    this.game = null;
    this.registrationSince = null;
    // A fresh tracker per match, so numbering starts again at 001.
    this.tracker = new BoxTracker();
    this.stabilizer.reset();
    this.judge.reset();
    this.sampler.reset();
    this.traces.clear();
    // Every match measures the room it is played in from scratch. The lamp
    // may have been switched off since the last one.
    this.baselines.clear();
    this.calibration = null;
    this.calibrationDone = false;
    this.applyThreshold();
    this.setPhase("LOBBY");
    this.setPhaseSub(this.mode === "demo" ? "Players lining up" : "Step into frame");
    this.loop();
  }

  /** The demo match: drawn players, no camera, no network. */
  startDemo() {
    if (!this.scene) return;
    this.halt();
    this.setMode("demo");
    this.scene.reset();
    this.source = this.scene.canvas;
    this.poseSource = this.scene;
    this.mirror = false;
    this.matchConfig = DEMO_MATCH;
    this.beginMatch();
    Probe.note("demo");
  }

  /**
   * Stop the demo while nobody can see it, and leave the stage ready to pick
   * it up again. A match playing to an empty room costs a laptop battery for
   * nothing.
   */
  pauseDemo() {
    if (this.mode === "demo" && this.demoPaused) return;
    this.halt();
    this.setMode("demo");
    if (this.scene) this.scene.reset();
    this.demoPaused = true;
  }

  /**
   * The demo, drawn and waiting behind a card, for a visitor who has asked
   * for less motion: it plays only when they start it.
   */
  idleDemo() {
    if (!this.scene) {
      this.halt();
      return;
    }
    this.halt();
    this.setMode("demo");
    this.scene.reset();
    this.showIdle(
      "A demo match is ready",
      "Drawn players, refereed by the real code. It plays when you start it.",
      true
    );
  }

  async play() {
    const button = el("btnPlay");
    if (button) button.disabled = true;
    this.halt();
    const run = this.runId;
    this.setMode("camera");
    this.showIdle(
      "Starting your camera",
      "Allow camera access when your browser asks. The pose model downloads once; after that everything runs on this device.",
      false
    );
    this.setStatus("Asking for the camera.");

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
    } catch (err) {
      // A prompt dismissed by the visitor moving on is not a missing camera.
      if (run !== this.runId) return;
      this.fail(
        "No camera",
        "The referee needs a camera to see you. Watch the demo match or the recorded one instead; neither needs a camera."
      );
      Probe.note("nocamera");
      return;
    }
    if (run !== this.runId) {
      // The visitor moved on while the permission prompt was up.
      for (const track of stream.getTracks()) track.stop();
      return;
    }

    this.stream = stream;
    this.video.srcObject = stream;
    this.stage.dataset.live = "true";
    this.video.play().catch(() => {});
    this.setStatus("Loading the pose model.");

    let poseSource;
    try {
      poseSource = await createPoseSource();
    } catch (err) {
      // If the visitor has moved on, `halt()` already stopped this camera.
      if (run !== this.runId) return;
      this.fail(
        "The pose model did not load",
        "Without it there is nobody to referee. Check your connection and try again, or watch the demo match, which needs no download."
      );
      Probe.note("noposeruntime");
      this.releaseCamera();
      return;
    }
    if (run !== this.runId) {
      if (poseSource && poseSource.close) poseSource.close();
      return;
    }

    this.poseSource = poseSource;
    this.source = this.video;
    this.mirror = true;
    this.matchConfig = {
      countdownS: this.config.countdownS,
      durationS: this.config.durationS,
      phaseMinS: this.config.phaseMinS,
      phaseMaxS: this.config.phaseMaxS,
    };
    if (this.idle) this.idle.hidden = true;
    if (button) button.disabled = false;
    this.safeAudio(() => this.chant.resume());
    this.beginMatch();
    this.setStatus("Waiting for a player to stand in frame.");
    Probe.note("camera");
  }

  /** The card over the stage: a message, and optionally a way out of it. */
  showIdle(title, detail, offerDemo) {
    if (!this.idle) return;
    this.idle.hidden = false;
    this.idle.querySelector(".idle-title").textContent = title;
    this.idle.querySelector(".idle-text").textContent = detail;
    this.idle.querySelector(".cta-row").hidden = !offerDemo;
  }

  fail(title, detail) {
    this.setMode("idle");
    this.showIdle(title, detail, true);
    const button = el("btnPlay");
    if (button) button.disabled = false;
    this.setStatus(title + ". " + detail);
  }

  releaseCamera() {
    this.stage.dataset.live = "false";
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    this.video.srcObject = null;
  }

  /**
   * Stop the camera and the pose loop, without touching the chant or the
   * end card. Called when a match finishes on its own, where the chant has
   * its own sting to play and the end card has to stay up. A finished match
   * still has a live camera and a running loop until this runs — nothing in
   * `loop()` checks whether the game is over.
   */
  stopEngine() {
    if (this.rafId !== null) cancelAnimationFrame(this.rafId);
    this.rafId = null;
    this.releaseCamera();
    if (this.poseSource && this.poseSource !== this.scene && this.poseSource.close) {
      try {
        this.poseSource.close();
      } catch (err) {
        // As in `halt`: closing is a courtesy.
      }
    }
    this.poseSource = null;
  }

  safeAudio(action) {
    try {
      action();
    } catch (err) {
      // Never let a muted or blocked audio context stop a match.
    }
  }

  /** Sound belongs to a match somebody is playing, not to the demo. */
  audible() {
    return this.mode === "camera";
  }

  /* ----------------------------------------------------------- main loop */

  sourceReady() {
    if (!this.source) return false;
    if (this.source === this.video) {
      if (this.video.readyState < 2) return false;
      this.fitStageToCamera();
      return true;
    }
    return true;
  }

  /**
   * Give the stage the camera's own shape.
   *
   * Boxes are placed in the video frame's coordinates. A stage of any other
   * shape would crop the picture, and every box would drift off its player.
   */
  fitStageToCamera() {
    const w = this.video.videoWidth;
    const h = this.video.videoHeight;
    if (!w || !h) return;
    const ratio = w + " / " + h;
    if (this.stage.style.aspectRatio !== ratio) this.stage.style.aspectRatio = ratio;
  }

  loop() {
    this.rafId = requestAnimationFrame(() => this.loop());
    if (!this.sourceReady()) return;

    // Monotonic, in seconds — the same unit `Game` and `FrameSampler` take.
    const now = performance.now() / 1000;
    if (this.mode === "demo") this.scene.step(now, this.game);

    let boxes = [];
    try {
      boxes = this.poseSource.detect(this.source, now * 1000);
    } catch (err) {
      return;
    }

    const { tracks, lost } = this.tracker.update(boxes, now);

    // Scoring rectangles hold still through the countdown and every red
    // light — the two stretches where a player is being asked not to move,
    // and so the two stretches where the window they are measured in must
    // not move either. Under green they follow, because everybody is walking
    // and a pinned window would lose them.
    const phase = this.game ? this.game.phase : "LOBBY";
    const hold = phase === "RED" || phase === "COUNTDOWN";
    const rects = this.stabilizer.update(tracks, hold);
    const violations = this.score(tracks, rects, now);

    if (this.game === null) {
      this.awaitRegistration(tracks, now);
    } else {
      const events = this.game.update(now, violations, lost);
      this.handleEvents(events, now);
    }

    if (this.rafId !== null) this.render(tracks, now);
  }

  /**
   * Crop every tracked player, pace them onto the referee's clock, score.
   *
   * @param {Array<object>} tracks Live tracks this frame.
   * @param {Map<number, object>} rects The stabilized rectangle to score each by.
   * @returns {Array<number>} Ids the judge has now confirmed are moving.
   */
  score(tracks, rects, now) {
    const crops = {};
    for (const track of tracks) {
      const rect = rects.get(track.id);
      if (!rect) continue;
      const window96 = cropWindow(this.source, rect, this.cropCtx);
      if (window96) crops[track.id] = window96;
    }

    const pair = this.sampler.offer(crops, now);
    if (!pair) return [];

    // Same gate the desktop engine applies in `app.run`: a pair is only fed
    // to the judge once the sampler has released it *and* the current red
    // light is armed. Grace exists so a player mid-stride is not punished
    // for momentum they had before the light turned — feeding the judge
    // during grace would let the EMA and streak it builds there survive
    // into the armed window and trigger a call the instant grace ends,
    // which defeats the point of having grace at all.
    const armed = this.game ? this.game.armed(now) : false;
    // The countdown is the one stretch of a match where every player is
    // standing set and nothing is at stake, which makes it the only honest
    // chance to measure what this camera reads on somebody holding still.
    // These scores are collected, never judged.
    const calibrating = this.game !== null && this.game.phase === "COUNTDOWN";

    const violations = [];
    for (const key of Object.keys(pair.cur)) {
      const previous = pair.prev[key];
      if (!previous) continue;
      const id = Number(key);
      // Computed unconditionally: the raw diff is cheap, and nothing below
      // this line about *displaying* a score should depend on whether the
      // light happens to be armed right now.
      const score = diffScoreWindow(previous, pair.cur[key], pair.dt);
      if (calibrating) this.noteBaseline(id, score);
      if (armed && this.judge.update(id, score)) violations.push(id);

      let trace = this.traces.get(id);
      if (!trace) {
        trace = [];
        this.traces.set(id, trace);
      }
      trace.push(this.judge.smoothed(id));
      if (trace.length > TRACE_MEMORY) trace.shift();
    }
    return violations;
  }

  /** Hold the countdown until somebody has actually been standing there. */
  awaitRegistration(tracks, now) {
    if (tracks.length === 0) {
      this.registrationSince = null;
      this.setPhaseSub(this.mode === "demo" ? "Players lining up" : "Step into frame");
      return;
    }
    if (this.registrationSince === null) this.registrationSince = now;
    const held = now - this.registrationSince;
    if (held < REGISTRATION_HOLD_S) {
      this.setPhaseSub(
        "Registering " + tracks.length + (tracks.length === 1 ? " player" : " players")
      );
      return;
    }

    this.game = new Game({
      ...this.matchConfig,
      graceS: this.config.graceS,
    });
    const ids = tracks.map((track) => track.id);
    this.handleEvents(this.game.start(now, ids), now);
    this.setStatus(
      ids.length + (ids.length === 1 ? " player registered." : " players registered.")
    );
  }

  handleEvents(events, now) {
    for (const event of events) {
      if (event.type === "PHASE_CHANGED") {
        // Same reset the desktop engine performs, for the same reason: no
        // phase inherits the previous one's baseline.
        this.judge.reset();
        this.sampler.reset();
        // A match always opens on green, and the countdown is what precedes
        // it — so the first green light is both the last moment the baseline
        // is complete and the first moment the cutoff it produces matters.
        // Later greens find the calibration already done and cost nothing.
        if (event.phase === "GREEN") this.finishCalibration();
        this.setPhase(event.phase);

        if (event.phase === "GREEN" && this.audible()) {
          this.safeAudio(() => this.chant.start());
        } else {
          this.safeAudio(() => this.chant.stop());
        }
        if (event.phase === "RED") this.setPhaseSub("Freeze: grace");
        if (this.game && this.game.finished) this.endMatch(event.phase);
      } else if (event.type === "PLAYER_ELIMINATED") {
        this.eliminate(event, now);
      }
    }
  }

  /**
   * The moment a player stops being a player.
   *
   * Three things land together, because one call is one event and it should
   * not arrive in instalments: the box they were scored through is stamped
   * OUT and breaks, their meter falls to the floor, and a low thump marks it
   * hitting. Only the stamp, the meter and the thump survive reduced motion
   * — a stamp is information, six shards flying apart is not.
   */
  breakPlayer(trackId) {
    if (this.audible()) this.safeAudio(() => this.chant.thud());

    const row = this.rows.get(trackId);
    if (row) {
      row.dataset.dropped = "true";
      window.setTimeout(() => {
        if (row.isConnected) delete row.dataset.dropped;
      }, 700);
    }

    if (lessMotion()) return;
    const box = this.overlay.querySelector('[data-track="' + trackId + '"]');
    if (!box || box.querySelector(".shard")) return;
    box.dataset.broke = "true";
    const shards = [];
    for (let i = 0; i < 6; i += 1) {
      const shard = document.createElement("span");
      shard.className = "shard";
      shard.style.setProperty("--throw", i * 60 + 12 + "deg");
      shard.style.setProperty("--wait", (i % 3) * 45 + "ms");
      box.appendChild(shard);
      shards.push(shard);
    }
    // The shards are litter once they have landed, and `renderBoxes` only
    // ever removes whole boxes — a player who stays in frame after being
    // called out would otherwise keep six spent spans forever.
    window.setTimeout(() => {
      for (const shard of shards) shard.remove();
    }, 900);
  }

  eliminate(event, now) {
    if (this.audible()) this.safeAudio(() => this.chant.buzz());
    this.fire(this.flash, "call");
    if (this.doll) this.doll.snap();
    this.breakPlayer(event.trackId);

    const trace = this.traces.get(event.trackId) || [];
    const reason =
      event.reason === "moved"
        ? "Kept moving after the red light's grace ran out."
        : "Left the frame, so the referee lost sight of them.";

    this.whyCard.innerHTML =
      "<h3>Player " + bib(event.trackId) + " is out</h3>" +
      '<p class="why-line">' + reason + "</p>" +
      sparkline(trace, this.threshold);
    this.whyCard.dataset.show = "true";
    if (this.whyTimer !== null) window.clearTimeout(this.whyTimer);
    this.whyTimer = window.setTimeout(() => {
      this.whyCard.dataset.show = "false";
      this.whyTimer = null;
    }, WHY_CARD_MS);

    this.setStatus(
      "Player " + bib(event.trackId) + " is out: " +
        (event.reason === "moved" ? "moved on a red light." : "left the frame.")
    );
    // The id, not just the fact. `scripts/verify_site.py` runs a walker and a
    // statue side by side and has to be able to tell which of them was called.
    Probe.note("out:" + event.trackId);
  }

  endMatch(phase) {
    this.safeAudio(() => this.chant.stop());
    if (this.audible()) this.safeAudio(() => this.chant.sting(phase === "VICTORY"));
    // The match is decided; nothing after this point needs a frame or a pose
    // detection. Stopping here — not waiting for "Play again" — is what
    // actually releases the camera, since `loop()` has no idea the game is
    // finished and would otherwise keep detecting and rendering against a
    // stage nobody can act on any more.
    this.stopEngine();
    const button = el("btnPlay");
    if (button) button.disabled = false;
    const survivors = this.game ? this.game.aliveCount : 0;
    const registered = this.game ? this.game.players.size : 0;
    const won = phase === "VICTORY";
    const detail =
      this.mode === "demo"
        ? this.autoplay
          ? "That was the demo. The next one starts in a moment, or play it yourself."
          : "That was the demo. Watch it again, or play it yourself."
        : won
          ? survivors === 1
            ? "One player stood still long enough. The clock ran out first."
            : survivors + " players stood still long enough. The clock ran out first."
          : "Every player was called out before the clock ran down.";
    this.showEnd(phase, survivors, registered, detail);
    // The loop has stopped, so the timer would otherwise hold its last
    // in-match reading under the end card; it shows who is left instead.
    this.renderTimer(0);
    // The next demo starts by itself only for a visitor who has not asked
    // for less motion; one who started it by hand gets the card to stay.
    if (this.mode === "demo" && this.autoplay) {
      const restart = () => {
        this.restartTimer = null;
        if (this.mode !== "demo" || this.demoPaused) return;
        // Somebody tabbing through the result card keeps it until they leave.
        if (this.endCard.contains(document.activeElement)) {
          this.restartTimer = window.setTimeout(restart, 1500);
          return;
        }
        this.startDemo();
      };
      this.restartTimer = window.setTimeout(restart, DEMO_RESTART_MS);
    }
    Probe.note("end:" + phase);
  }

  /**
   * The end card, and the moment around it.
   *
   * The card leads with the count rather than the word: how many people were
   * left standing out of how many started is the result, and "Victory" is
   * only the name for it. Underneath, the arena reacts once — paper thrown
   * from the corners and a small bow for a win, the lights swept out for a
   * wipeout — and then holds still, because a card somebody is reading
   * should not be moving.
   *
   * @param {string} phase "VICTORY" or "WIPEOUT".
   * @param {number} survivors Players still standing at the final whistle.
   * @param {number} registered Players the referee started the match with.
   * @param {string} detail One sentence of plain English under the count.
   */
  showEnd(phase, survivors, registered, detail) {
    const won = phase === "VICTORY";
    this.stage.dataset.outcome = phase;
    this.endCard.dataset.show = "true";
    this.endCard.dataset.outcome = phase;

    const eyebrow = el("endEyebrow");
    if (eyebrow) {
      eyebrow.textContent =
        this.mode === "demo"
          ? "Demo match over"
          : this.mode === "replay"
            ? "Recorded match over"
            : won
              ? "The clock ran out"
              : "The referee got everyone";
    }
    el("endTitle").textContent = won ? "Victory" : "Wipeout";
    const score = el("endScore");
    if (score) {
      score.innerHTML =
        '<span class="num">' + survivors + "</span>" +
        '<span class="of">of ' + registered + " still standing</span>";
    }
    el("endDetail").textContent = detail;

    // The way on from here depends on what just finished.
    const again = el("btnAgainText");
    if (again) again.textContent = this.mode === "camera" ? "Play again" : "Play with your camera";
    const second = el("btnEndSecond");
    if (second) {
      // After a demo that restarts by itself, the other thing to watch is the
      // recorded match; anywhere else, it is the demo.
      const toReplay = this.mode === "demo" && this.autoplay;
      second.dataset.action = toReplay ? "replay" : "demo";
      second.textContent = toReplay
        ? "Watch a recorded match"
        : this.mode === "demo"
          ? "Watch the demo again"
          : "Watch the demo match";
    }

    if (won) {
      this.celebrate();
      if (this.doll) this.doll.bow();
    } else if (this.lightsOut) {
      this.lightsOut.dataset.on = "true";
    }
  }

  /**
   * Throw paper across the stage, in the game's own colours: tracksuit teal,
   * guard pink, and the doll's orange and yellow, which read on the dark
   * stage in either theme.
   */
  celebrate() {
    if (lessMotion() || !this.fx || typeof burstConfetti !== "function") return;
    if (this.stopConfetti) this.stopConfetti();
    this.stopConfetti = burstConfetti(this.fx, {
      colors: ["#22a08d", "#ff5c9b", "#f08a24", "#ffc531"],
    });
  }

  /* -------------------------------------------------------------- render */

  render(tracks, now) {
    const armed = this.game ? this.game.armed(now) : false;
    if (this.doll) this.doll.setArmed(armed);
    if (armed) this.stage.dataset.armed = "true";
    else delete this.stage.dataset.armed;
    if (this.game && this.stage.dataset.phase === "RED") {
      this.setPhaseSub(armed ? "Freeze: watching" : "Freeze: grace");
    }

    this.renderBoxes(tracks);
    this.renderRoster(tracks);
    this.renderTimer(now);
  }

  renderBoxes(tracks) {
    const live = new Set();
    for (const track of tracks) {
      live.add(track.id);
      let node = this.overlay.querySelector('[data-track="' + track.id + '"]');
      if (!node) {
        node = document.createElement("div");
        node.className = "player-box";
        node.dataset.track = String(track.id);
        node.innerHTML =
          '<span class="player-tag"></span>' +
          '<div class="meter"><div class="fill"></div><div class="tick"></div></div>';
        this.overlay.appendChild(node);
      }

      // The drawn box is the smoothed one, never the pinned one. It follows
      // the player under every light, so nobody has to wonder why the outline
      // stopped tracking them — while what they are actually *scored* through
      // holds still underneath.
      const drawn = this.stabilizer.smoothed(track.id) || track.box;

      // A mirrored camera needs a mirrored overlay — by arithmetic, not by a
      // transform, or every label would come out backwards.
      const left = this.mirror ? 1 - drawn.x2 : drawn.x1;
      node.style.left = fmt(left * 100, 2) + "%";
      node.style.top = fmt(drawn.y1 * 100, 2) + "%";
      node.style.width = fmt((drawn.x2 - drawn.x1) * 100, 2) + "%";
      node.style.height = fmt((drawn.y2 - drawn.y1) * 100, 2) + "%";

      const player = this.game ? this.game.players.get(track.id) : null;
      const out = Boolean(player && !player.alive);
      const smoothed = out ? 0 : this.judge.smoothed(track.id);
      const over = smoothed > this.threshold;
      node.dataset.state = out ? "out" : over ? "warn" : "safe";
      node.querySelector(".player-tag").textContent = bib(track.id);
      const meter = node.querySelector(".meter");
      meter.dataset.over = over ? "true" : "false";
      meter.querySelector(".fill").style.width =
        fmt(Math.min(smoothed / (this.threshold * METER_SCALE), 1) * 100, 1) + "%";
    }

    for (const node of Array.from(this.overlay.children)) {
      if (!live.has(Number(node.dataset.track))) node.remove();
    }
  }

  /** One patch per player, in the panel beside the stage. */
  rosterRow(id, withMeter) {
    let row = this.rows.get(id);
    if (row) return row;
    row = document.createElement("div");
    row.className = "bib";
    row.innerHTML =
      '<span class="bib-num">' + bib(id) + "</span>" +
      '<span class="bib-body">' +
      (withMeter ? '<span class="meter"><span class="fill"></span><span class="tick"></span></span>' : "") +
      '<span class="bib-state"></span></span>';
    this.roster.appendChild(row);
    this.rows.set(id, row);
    return row;
  }

  renderRoster(tracks) {
    const ids = this.game ? Array.from(this.game.players.keys()) : tracks.map((t) => t.id);
    const wanted = new Set(ids);

    for (const [id, row] of this.rows) {
      if (!wanted.has(id)) {
        row.remove();
        this.rows.delete(id);
      }
    }

    for (const id of ids) {
      const row = this.rosterRow(id, true);
      const player = this.game ? this.game.players.get(id) : null;
      const out = player ? !player.alive : false;
      const smoothed = out ? 0 : this.judge.smoothed(id);
      row.dataset.out = out ? "true" : "false";
      row.querySelector(".bib-state").textContent = out ? "Out" : "";
      const meter = row.querySelector(".meter");
      if (!meter) continue;
      meter.dataset.over = !out && smoothed > this.threshold ? "true" : "false";
      meter.querySelector(".fill").style.width = out
        ? "0%"
        : fmt(Math.min(smoothed / (this.threshold * METER_SCALE), 1) * 100, 1) + "%";
    }
  }

  renderTimer(now) {
    const label = el("timerLabel");
    if (!this.game) {
      this.timer.set("--");
      if (label) label.textContent = "waiting";
      return;
    }
    let value;
    let text;
    if (this.game.phase === "COUNTDOWN") {
      value = Math.ceil(this.game.countdownLeft(now));
      text = "get ready";
    } else if (this.game.inMatch) {
      value = Math.ceil(this.game.timeLeft(now));
      text = value === 1 ? "second left" : "seconds left";
    } else {
      value = this.game.aliveCount;
      text = "still standing";
    }
    this.timer.set(String(Math.max(value, 0)));
    if (label) label.textContent = text;
  }

  /* ------------------------------------------ difficulty and calibration */

  setDifficulty(name) {
    this.difficulty = name;
    this.applyThreshold();
    for (const button of document.querySelectorAll("#difficulty button")) {
      button.setAttribute("aria-pressed", button.dataset.level === name ? "true" : "false");
    }
  }

  /** The chosen preset, before this camera has had anything to say about it. */
  get presetThreshold() {
    return this.config.diffThreshold * DIFFICULTY[this.difficulty];
  }

  /**
   * Set the live cutoff from the preset and whatever calibration measured.
   *
   * The preset is a floor, never a ceiling: calibration can only ever raise
   * the cutoff. A camera quieter than the lab footage does not earn a
   * stricter game than the player asked for. And because a baseline is only
   * ever admitted from a player who was already under the preset,
   * `CAL_MARGIN` times it can never exceed three times the preset — the
   * calibrated cutoff is bounded without needing a second knob to bound it.
   */
  applyThreshold() {
    const preset = this.presetThreshold;
    let threshold = preset;
    if (this.calibration !== null) {
      threshold = Math.max(preset, CAL_MARGIN * this.calibration);
    }
    this.threshold = threshold;
    this.judge.threshold = threshold;
    this.renderThresholdNotes();
  }

  /** Keep one countdown sample for a player. */
  noteBaseline(id, score) {
    let scores = this.baselines.get(id);
    if (!scores) {
      scores = [];
      this.baselines.set(id, scores);
    }
    scores.push(score);
  }

  /**
   * Turn the countdown's baseline scores into a cutoff for this camera.
   *
   * Per player, the 95th percentile of what their own crop scored; across
   * players, the largest of those, because the cutoff is one number for the
   * whole match and the noisiest crop in the room is the one that would
   * produce the first wrong call.
   *
   * Two players are left out of that maximum rather than trusted.
   *
   * One who contributed fewer than `CAL_MIN_SAMPLES` samples has measured
   * nothing — they walked into a countdown that was nearly over.
   *
   * One whose baseline is already past the preset cutoff was *moving*, and
   * movement is not noise. Admitting them would let anyone buy a cutoff
   * nothing could cross by waving through the countdown, and would hand the
   * whole room a threshold set by the least still person in it. A noise
   * floor is a property of the camera and the light, and the players who
   * were genuinely still are the ones who reveal it.
   *
   * If that leaves nobody, the preset stands unchanged and the page says so
   * rather than implying a measurement that never happened.
   */
  finishCalibration() {
    if (this.calibrationDone) return;
    this.calibrationDone = true;

    const preset = this.presetThreshold;
    let floor = null;
    for (const scores of this.baselines.values()) {
      if (scores.length < CAL_MIN_SAMPLES) continue;
      const p95 = percentile(scores, 0.95);
      if (p95 >= preset) continue;
      floor = floor === null ? p95 : Math.max(floor, p95);
    }
    this.calibration = floor;
    this.applyThreshold();
  }

  /**
   * The two lines under the stage: what the chosen difficulty means, and —
   * on a camera match — what measuring the camera found. The numbers behind
   * them are on the measurements page; here they are said in words.
   */
  renderThresholdNotes() {
    const note = el("difficultyNote");
    if (note) note.textContent = DIFFICULTY_NOTES[this.difficulty] || "";

    const live = el("calibrationNote");
    if (!live) return;
    if (this.mode !== "camera") {
      live.textContent = "";
      return;
    }
    if (!this.calibrationDone) {
      live.textContent =
        "The countdown measures your camera while everyone stands still, and " +
        "raises the line if the picture is noisy.";
      return;
    }
    if (this.calibration === null) {
      live.textContent =
        "Nobody held still long enough during the countdown to measure your " +
        "camera, so the line stays where the difficulty puts it.";
      return;
    }
    const raised = this.threshold > this.presetThreshold + 1e-12;
    live.textContent = raised
      ? "Calibrated to your camera: the line was raised to clear its noise."
      : "Calibrated to your camera: its noise sits well under the line.";
  }

  /* -------------------------------------------------------------- replay */

  /**
   * The recorded match — for anyone without a camera, or unwilling.
   *
   * `docs/demo_match.mp4` is real HUD-annotated footage of `redlight play`
   * refereeing the benchmark video, boxes and verdicts and all — not staged,
   * the same run the README's commands reproduce. Beside it, the panel shows
   * the run's recorded light schedule (`REPLAY_LIGHT_STARTS_S`) and its
   * calls, read from `RL_DATA.benchmark.demo_match`.
   *
   * The video is the clock. The light, doll, timer and players are all
   * computed from `video.currentTime`, so a slow first frame, buffering, a
   * pause or a scrub moves them with the footage instead of letting them run
   * ahead of it. The recording opens in the lobby: the match was
   * auto-started on the frame where `auto_start_frames` frames in a row had
   * seen a player (frame 19 at 10 fps, since every frame of this footage has
   * a walker in it), and the match clock began `countdown_s` after that. If
   * the video cannot play at all, the page clock stands in for it.
   *
   * Everything shown is recomputed from the time on every frame rather than
   * latched, so scrubbing backwards brings eliminated players back.
   */
  startReplay() {
    this.halt();
    const button = el("btnPlay");
    if (button) button.disabled = false;
    this.setMode("replay");

    const demo = this.data.benchmark.demo_match;
    const grace = demo.config.grace_s;
    const countdown = demo.config.countdown_s;
    const duration = demo.config.duration_s;
    const schedule = buildReplaySchedule(duration);
    const fps = this.data.benchmark.meta.fps;
    // Seconds into the recording at which the countdown and the match begin.
    const countdownAt = (demo.config.auto_start_frames - 1) / fps;
    const matchAt = countdownAt + countdown;

    const video = document.createElement("video");
    video.className = "replay-video";
    video.src = "demo_match.mp4";
    video.poster = "demo_poster.jpg";
    video.muted = true;
    video.controls = true;
    video.playsInline = true;
    video.setAttribute("aria-label", "Recorded match refereed by the Python engine");
    this.stage.insertBefore(video, this.overlay);
    this.replayVideo = video;
    video.play().catch(() => {});

    const pageOrigin = performance.now() / 1000;
    const recordingTime = () =>
      video.error ? performance.now() / 1000 - pageOrigin : video.currentTime;

    const players = [];
    this.roster.dataset.compact = "true";
    for (let id = 1; id <= demo.players; id += 1) {
      const row = this.rosterRow(id, false);
      players.push({ id, row, out: false });
    }

    const enter = (phase) => {
      if (this.stage.dataset.phase !== phase) this.setPhase(phase);
    };
    const label = el("timerLabel");

    const step = () => {
      const matchT = recordingTime() - matchAt;

      if (matchT >= duration) {
        // The final whistle: the timer shows who is left, the way it does at
        // the end of a live match.
        const outcome = demo.outcome === "victory" ? "VICTORY" : "WIPEOUT";
        enter(outcome);
        if (this.doll) this.doll.setArmed(false);
        delete this.stage.dataset.armed;
        this.timer.set(String(demo.survivors));
        if (label) label.textContent = "still standing";
        this.showEnd(
          outcome,
          demo.survivors,
          demo.players,
          "Recorded by the Python engine on the benchmark footage. Not staged, " +
            "and reproducible from the README's install steps."
        );
        this.replayTimer = null;
        Probe.note("replay-end");
        return;
      }

      let armed = false;
      if (matchT < -countdown) {
        enter("LOBBY");
        this.setPhaseSub("Registering players");
        this.timer.set("--");
        if (label) label.textContent = "waiting";
      } else if (matchT < 0) {
        enter("COUNTDOWN");
        this.timer.set(String(Math.ceil(-matchT)));
        if (label) label.textContent = "get ready";
      } else {
        // `schedule` always spans exactly `[0, duration)`, so this lookup
        // succeeds for any `matchT` reachable here — the branch above already
        // returns once `matchT >= duration`. The fallback is a complete slot
        // object regardless, so a future change to the schedule's shape
        // cannot turn a missed lookup into a `slot.from` crash.
        const slot = schedule.find((s) => matchT >= s.from && matchT < s.to) || {
          phase: "RED",
          from: schedule.length ? schedule[schedule.length - 1].from : 0,
          to: duration,
        };
        enter(slot.phase);
        if (slot.phase === "RED") {
          armed = matchT - slot.from >= grace;
          this.setPhaseSub(armed ? "Freeze: watching" : "Freeze: grace");
        }
        const left = Math.max(Math.ceil(duration - matchT), 0);
        this.timer.set(String(left));
        if (label) label.textContent = left === 1 ? "second left" : "seconds left";
      }
      if (this.doll) this.doll.setArmed(armed);
      if (armed) this.stage.dataset.armed = "true";
      else delete this.stage.dataset.armed;

      // Recomputed from the time, not latched: a scrub backwards puts a
      // player back in, and only a player going out is announced.
      for (const player of players) {
        const call = demo.eliminations.find(([trackId]) => trackId === player.id);
        const out = call !== undefined && matchT >= call[1];
        if (out === player.out) continue;
        player.out = out;
        player.row.dataset.out = out ? "true" : "false";
        player.row.querySelector(".bib-state").textContent = out
          ? call[2] === "moved"
            ? "moved"
            : "left"
          : "";
        if (out) {
          // The recorded reason, against the player it belongs to: a lost
          // track and a caught movement are different failures, and the
          // replay should not blur them into one.
          this.setStatus(
            "Player " + bib(call[0]) + " out at " + fmt(call[1], 1) + " s: " +
              (call[2] === "moved" ? "moved on a red light." : "left the frame.")
          );
        }
      }

      this.replayTimer = requestAnimationFrame(step);
    };

    Probe.note("replay");
    this.replayTimer = requestAnimationFrame(step);
  }

  stopReplay() {
    if (this.replayTimer !== null) {
      cancelAnimationFrame(this.replayTimer);
      this.replayTimer = null;
    }
    if (this.replayVideo) {
      this.replayVideo.pause();
      this.replayVideo.removeAttribute("src");
      this.replayVideo.remove();
      this.replayVideo = null;
    }
  }
}

/* ------------------------------------------------------------------ boot */

function initArena() {
  const data = window.RL_DATA;
  if (!data || !el("stage")) return;

  const arena = new Arena(data);
  window.rlArena = arena;
  arena.setDifficulty("standard");
  arena.setMode("idle");
  arena.setPhase("LOBBY");

  const scrollToStage = () => {
    const stage = el("console");
    if (!stage) return;
    const rect = stage.getBoundingClientRect();
    // Only move the page if the stage is not already mostly in view.
    if (rect.top < 0 || rect.top > window.innerHeight * 0.45) {
      stage.scrollIntoView({ behavior: lessMotion() ? "auto" : "smooth", block: "center" });
    }
  };

  const play = el("btnPlay");
  if (play) {
    play.addEventListener("click", () => {
      scrollToStage();
      arena.play();
    });
  }
  const again = el("btnAgain");
  if (again) again.addEventListener("click", () => arena.play());

  for (const trigger of document.querySelectorAll("[data-replay]")) {
    trigger.addEventListener("click", () => {
      scrollToStage();
      arena.startReplay();
    });
  }

  // Buttons on the stage that change what it shows carry the mode they
  // switch to, so one listener serves the idle card and the end card.
  el("stage").addEventListener("click", (event) => {
    const button = event.target.closest("[data-action]");
    if (!button) return;
    if (button.dataset.action === "demo") arena.startDemo();
    else if (button.dataset.action === "replay") arena.startReplay();
  });

  const stop = el("btnStop");
  if (stop) {
    stop.addEventListener("click", () => {
      if (arena.autoplay) arena.startDemo();
      else arena.idleDemo();
    });
  }

  const sound = el("btnSound");
  if (sound) {
    // The label always names the state the sound is in, read from the chant
    // itself rather than the button's last attribute, so it matches reality
    // before anyone has clicked it.
    const syncSound = () => {
      const muted = arena.chant.muted;
      sound.setAttribute("aria-pressed", muted ? "false" : "true");
      sound.setAttribute("aria-label", muted ? "Sound off" : "Sound on");
      sound.title = muted ? "Sound is off for your matches" : "Sound is on for your matches";
    };
    sound.addEventListener("click", () => {
      arena.chant.setMuted(!arena.chant.muted);
      syncSound();
    });
    syncSound();
  }

  for (const button of document.querySelectorAll("#difficulty button")) {
    button.addEventListener("click", () => arena.setDifficulty(button.dataset.level));
  }

  // The demo opens the page, unless the visitor has asked for less motion or
  // a headless check is about to drive the arena itself.
  const action = new URLSearchParams(window.location.search).get("action");
  const autoplay = !lessMotion() && (!Probe.on || action === "demo");
  arena.autoplay = autoplay;
  if (arena.scene && autoplay) {
    arena.startDemo();
  } else if (arena.scene && !Probe.on) {
    arena.idleDemo();
  } else if (arena.scene) {
    arena.setMode("demo");
    arena.scene.reset();
  }

  // Pause the demo while the stage is out of sight, and pick it up again
  // when it comes back.
  const consoleNode = el("console");
  let inView = true;
  const sync = () => {
    if (arena.mode !== "demo" || !autoplay) return;
    const visible = inView && !document.hidden;
    if (!visible) arena.pauseDemo();
    else if (arena.demoPaused) arena.startDemo();
  };
  if (consoleNode && "IntersectionObserver" in window) {
    new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          inView = entry.isIntersecting && entry.intersectionRatio >= 0.12;
        }
        sync();
      },
      { threshold: [0, 0.12] }
    ).observe(consoleNode);
  }
  document.addEventListener("visibilitychange", sync);

  // Opening the measurements view hides the stage. A camera or a recording
  // left running there would be a camera on, or a video playing, for nobody.
  window.addEventListener("rl-view", (event) => {
    if (event.detail !== "measurements") return;
    if (arena.mode !== "camera" && arena.mode !== "replay") return;
    if (arena.autoplay) arena.pauseDemo();
    else arena.idleDemo();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initArena();
});
