/**
 * The playable arena: camera in, verdicts out.
 *
 * This file is the wiring, not the rules. The rules live in `judge.js` (the
 * scoring kernel, pinned to Python by golden fixtures) and `game.js` (the
 * phase machine, pinned by the same parity suite). What happens here is the
 * part a browser has to do for itself: open a camera, find the players, cut
 * their boxes into 96x96 windows, pace those windows onto the referee's
 * fixed 0.1 s clock, and turn the events that come back into something a
 * person can read while standing still in front of a laptop.
 *
 * Three details are worth knowing before changing anything.
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
 * red light, so detector wobble cannot be mistaken for a player moving. That
 * mistake is what this arena shipped with, and `site/pose.js` explains the
 * mechanism at length.
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

/** Circumference of the countdown ring's circle (r = 54). */
const RING_CIRCUMFERENCE = 2 * Math.PI * 54;

/**
 * Phases the replay steps through, as fractions of the match duration.
 *
 * Chosen to fit the recorded calls at the 12.0 s duration this demo was
 * baked with. Kept as fractions rather than absolute seconds so the last
 * boundary always lands exactly on whatever `duration_s` the baked demo
 * actually ships — `buildReplaySchedule` scales these against it — and a
 * regenerated demo with a different duration cannot leave a slot lookup
 * with nowhere to land.
 */
const REPLAY_SCHEDULE_FRACTIONS = Object.freeze([
  { phase: "GREEN", from: 0.0 / 12.0, to: 2.1 / 12.0 },
  { phase: "RED", from: 2.1 / 12.0, to: 4.5 / 12.0 },
  { phase: "GREEN", from: 4.5 / 12.0, to: 6.6 / 12.0 },
  { phase: "RED", from: 6.6 / 12.0, to: 8.8 / 12.0 },
  { phase: "GREEN", from: 8.8 / 12.0, to: 10.5 / 12.0 },
  { phase: "RED", from: 10.5 / 12.0, to: 1.0 },
]);

/**
 * The replay schedule for one match, in seconds.
 *
 * The last slot's `to` is exactly `duration`, so any `matchT` inside
 * `[0, duration)` — the only range the caller ever looks up — is covered.
 */
function buildReplaySchedule(duration) {
  return REPLAY_SCHEDULE_FRACTIONS.map((slot) => ({
    phase: slot.phase,
    from: slot.from * duration,
    to: slot.to * duration,
  }));
}

/** How long the arena holds its cold, frozen look at a green-to-red snap. */
const FREEZE_MS = 260;

/**
 * True when the visitor has asked the platform for less movement.
 *
 * The stylesheet flattens every transition and animation on its own. This is
 * for the handful of effects CSS cannot reason about — a full-frame white
 * flash, a burst of thrown paper, a box breaking into shards — which are not
 * worth flattening, only worth skipping.
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

  function label() {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    button.textContent = dark ? "Day game" : "Night game";
    button.setAttribute(
      "aria-label",
      dark ? "Switch to the light theme" : "Switch to the dark theme"
    );
  }

  button.addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
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

/* --------------------------------------------------------------- the hero */

/**
 * The title plays the game with you.
 *
 * The hero doll turns on her own slow schedule and the two words of the
 * title light up with her, so the mechanic is legible before anyone has
 * clicked anything or granted anything.
 *
 * Once a real match starts she stops improvising and follows it. Two dolls
 * on one page disagreeing about the light would be worse than one doll, and
 * a title that says "red light" over a green arena is a bug a reader can
 * see.
 */
function initHeroDemo() {
  const mount = document.getElementById("heroDoll");
  const masthead = document.querySelector(".masthead");
  if (!mount || !window.createDoll) return null;

  const doll = createDoll(mount);
  const caption = document.getElementById("heroDollCaption");
  let phase = "GREEN";
  let timer = null;

  const CAPTIONS = {
    GREEN: "Green light — go",
    RED: "Red light — hold still",
    COUNTDOWN: "Countdown — get ready",
    LOBBY: "Waiting for players",
    VICTORY: "The clock ran out",
    WIPEOUT: "Nobody left standing",
  };

  function show(next) {
    phase = next;
    doll.setPhase(phase);
    doll.setArmed(phase === "RED");
    if (masthead) masthead.dataset.demo = phase === "GREEN" ? "GREEN" : "RED";
    if (caption) caption.textContent = CAPTIONS[phase] || CAPTIONS.LOBBY;
  }

  function tick() {
    show(phase === "GREEN" ? "RED" : "GREEN");
    timer = window.setTimeout(tick, phase === "GREEN" ? 2600 : 2200);
  }

  tick();
  return {
    /** Hand the doll over to a live match. */
    follow(livePhase) {
      if (timer !== null) {
        window.clearTimeout(timer);
        timer = null;
      }
      show(livePhase);
    },
    /** Take the doll back when the match is over. */
    resume() {
      if (timer !== null) return;
      tick();
    },
  };
}

/* -------------------------------------------------------------- utilities */

function el(id) {
  return document.getElementById(id);
}

function fmt(value, digits = 2) {
  return Number(value).toFixed(digits);
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
  const height = 44;
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
      '" stroke="currentColor" stroke-width="1" stroke-dasharray="4 3" opacity="0.55"/>',
    '<polyline points="' + points + '" fill="none" stroke="var(--red-lit)" stroke-width="2" ',
    'stroke-linejoin="round" stroke-linecap="round"/>',
    "</svg>",
  ].join("");
}

/* ---------------------------------------------------------------- probing */

/**
 * A read-out of what the arena did, for a headless browser to assert on.
 *
 * Only active under `?probe=1`. `scripts/verify_site.py` drives the whole
 * play path with a fake camera and reads the result out of the page title,
 * which is the one piece of state a `--dump-dom` run can see without a
 * screenshot or a console scrape.
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

/**
 * Owns the camera, the pose source, the judge, the game, and the HUD.
 *
 * Built once on page load in a dormant state: nothing is requested, nothing
 * is fetched, and no frame is looked at until `play()` is called from a
 * click.
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
    // True once a match or a replay owns the page. Until then the hero doll
    // runs her own demo loop, and nothing here may interrupt it — a page
    // that has not been played yet should still be showing the mechanic.
    this.live = false;
    this.registrationSince = null;
    this.traces = new Map();
    this.rows = new Map();
    this.replayTimer = null;
    /** Cancels a confetti burst still in flight, or null. */
    this.stopConfetti = null;
    this.freezeTimer = null;

    const canvas = document.createElement("canvas");
    canvas.width = 96;
    canvas.height = 96;
    // `willReadFrequently` is the difference between a readback that stays
    // on the CPU and one that stalls the pipeline every frame; the arena
    // does exactly one `getImageData` per player per frame.
    this.cropCtx = canvas.getContext("2d", { willReadFrequently: true });
  }

  /* ------------------------------------------------------------ lifecycle */

  setStatus(text) {
    if (this.status) this.status.textContent = text;
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
    const pill = el("phasePill");
    if (pill) pill.dataset.phase = phase;
    const word = el("phaseWord");
    if (word) {
      word.textContent =
        phase === "GREEN"
          ? "Green light"
          : phase === "RED"
            ? "Red light"
            : phase === "COUNTDOWN"
              ? "Get ready"
              : phase === "VICTORY"
                ? "Victory"
                : phase === "WIPEOUT"
                  ? "Wipeout"
                  : "Lobby";
    }
    if (this.doll) this.doll.setPhase(phase);
    if (this.live && window.rlHeroDemo) window.rlHeroDemo.follow(phase);
    Probe.note(phase);
  }

  setPhaseSub(text) {
    const sub = el("phaseSub");
    if (sub) sub.textContent = text;
  }

  async play() {
    const button = el("btnPlay");
    if (button) button.disabled = true;
    this.stopReplay();
    this.resetStage();
    this.setStatus("Asking for the camera.");

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
    } catch (err) {
      this.fail(
        "No camera",
        "The arena needs a camera to referee anyone. Watch a recorded match instead — every number on this page was measured without one."
      );
      Probe.note("nocamera");
      return;
    }

    this.video.srcObject = this.stream;
    this.stage.dataset.live = "true";
    this.video.play().catch(() => {});
    this.setStatus("Loading the pose runtime.");

    try {
      this.poseSource = await createPoseSource();
    } catch (err) {
      this.fail(
        "No pose runtime",
        "The pose model could not be fetched, so there is nobody to referee. Everything else on this page works without it."
      );
      Probe.note("noposeruntime");
      this.releaseCamera();
      return;
    }

    this.live = true;
    if (this.idle) this.idle.hidden = true;
    this.endCard.dataset.show = "false";
    this.whyCard.dataset.show = "false";

    try {
      this.chant.resume();
    } catch (err) {
      // Audio is a garnish here; a match is perfectly playable in silence.
    }

    this.game = null;
    this.registrationSince = null;
    this.tracker.reset();
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
    this.setPhaseSub("Step into frame");
    this.setStatus("Waiting for a player to stand in frame.");
    Probe.note("camera");

    this.loop();
  }

  fail(title, detail) {
    if (this.idle) {
      this.idle.hidden = false;
      this.idle.querySelector("h3").textContent = title;
      this.idle.querySelector("p").textContent = detail;
    }
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
   * end card.
   *
   * Called both when a match finishes on its own (`endMatch`, where the
   * chant has its own sting to play and the end card has to stay up) and
   * when the visitor walks away from one (`stop`, where everything goes
   * quiet). A match that has ended still has a live camera and a running
   * `requestAnimationFrame` loop until this runs — nothing else in `loop()`
   * checks whether the game is finished.
   */
  stopEngine() {
    if (this.rafId !== null) cancelAnimationFrame(this.rafId);
    this.rafId = null;
    this.releaseCamera();
    if (this.poseSource && this.poseSource.close) {
      try {
        this.poseSource.close();
      } catch (err) {
        // A runtime that will not close is still a runtime we are done with.
      }
    }
    this.poseSource = null;
  }

  stop() {
    this.chant.stop();
    this.stopEngine();
  }

  /* ----------------------------------------------------------- main loop */

  loop() {
    this.rafId = requestAnimationFrame(() => this.loop());
    if (this.video.readyState < 2) return;

    // Monotonic, in seconds — the same unit `Game` and `FrameSampler` take.
    const now = performance.now() / 1000;

    let boxes = [];
    try {
      boxes = this.poseSource.detect(this.video, now * 1000);
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

    this.render(tracks, now);
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
      const window96 = cropWindow(this.video, rect, this.cropCtx);
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
      this.setPhaseSub("Step into frame");
      return;
    }
    if (this.registrationSince === null) this.registrationSince = now;
    const held = now - this.registrationSince;
    if (held < REGISTRATION_HOLD_S) {
      this.setPhaseSub("Registering " + tracks.length);
      return;
    }

    this.game = new Game({
      countdownS: this.config.countdownS,
      durationS: this.config.durationS,
      phaseMinS: this.config.phaseMinS,
      phaseMaxS: this.config.phaseMaxS,
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

        if (event.phase === "GREEN") {
          this.setPhaseSub("Chant running");
          this.safeAudio(() => this.chant.start());
        } else {
          this.safeAudio(() => this.chant.stop());
        }
        if (event.phase === "RED") this.setPhaseSub("Grace");
        if (event.phase === "COUNTDOWN") this.setPhaseSub("Countdown");
        if (this.game && this.game.finished) this.endMatch(event.phase);
      } else if (event.type === "PLAYER_ELIMINATED") {
        this.eliminate(event, now);
      }
    }
  }

  safeAudio(action) {
    try {
      action();
    } catch (err) {
      // Never let a muted or blocked audio context stop a match.
    }
  }

  /**
   * The moment a player stops being a player.
   *
   * Three things land together, because one call is one event and it should
   * not arrive in instalments: the box they were scored through breaks into
   * shards and goes grey, their meter falls to the floor, and a low thump
   * marks it hitting. The buzzer says *what* happened; this says how much it
   * cost. Only the meter and the thump survive reduced motion — a number
   * dropping is information, six shards flying apart is not.
   */
  breakPlayer(trackId) {
    this.safeAudio(() => this.chant.thud());

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
    this.safeAudio(() => this.chant.buzz());
    this.fire(this.flash, "call");
    if (this.doll) this.doll.snap();
    this.breakPlayer(event.trackId);

    const trace = this.traces.get(event.trackId) || [];
    const reason =
      event.reason === "moved"
        ? "moved on an armed red light"
        : "left the arena — the tracker lost them";
    const peak = trace.length ? Math.max.apply(null, trace) : 0;

    this.whyCard.dataset.show = "true";
    this.whyCard.innerHTML =
      "<h4>Player " + event.trackId + " is out</h4>" +
      '<p class="why-line">' + reason + "</p>" +
      '<p class="why-line">peak ' + fmt(peak, 3) + " vs threshold " + fmt(this.threshold, 3) +
      " body-fractions/s</p>" +
      sparkline(trace, this.threshold);

    this.setStatus("Player " + event.trackId + " eliminated: " + reason + ".");
    // The id, not just the fact. `scripts/verify_site.py` runs a walker and a
    // statue side by side and has to be able to tell which of them was called.
    Probe.note("out:" + event.trackId);
  }

  endMatch(phase) {
    this.safeAudio(() => this.chant.stop());
    this.safeAudio(() => this.chant.sting(phase === "VICTORY"));
    // The match is decided; nothing after this point needs a camera frame
    // or a pose detection. Stopping here — not waiting for "Play again" —
    // is what actually releases the camera, since `loop()` has no idea the
    // game is finished and would otherwise keep detecting and rendering
    // against a stage nobody can act on any more. `stopEngine` leaves the
    // chant alone (the sting above is still playing) and leaves the end
    // card and roster exactly as rendered.
    this.stopEngine();
    const button = el("btnPlay");
    if (button) button.disabled = false;
    const survivors = this.game ? this.game.aliveCount : 0;
    const registered = this.game ? this.game.players.size : 0;
    this.showEnd(
      phase,
      survivors,
      registered,
      phase === "VICTORY"
        ? survivors === 1
          ? "One player stood still long enough. The clock ran out first."
          : survivors + " players stood still long enough. The clock ran out first."
        : "Every registered player was called out before the clock ran down."
    );
    Probe.note("end:" + phase);
  }

  /**
   * The end card, and the moment around it.
   *
   * The card leads with the count rather than the word: how many people were
   * left standing out of how many started is the result, and "Victory" is
   * only the name for it. Underneath, the arena reacts once — paper thrown
   * from the corners and a small bow for a win, the lights swept out for a
   * wipeout — and then holds completely still, because a card somebody is
   * reading should not be moving.
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

    // The eyebrow names what ended the match; the count says how it went.
    // Two lines that both said "nobody is left" would be one line twice.
    const eyebrow = el("endEyebrow");
    if (eyebrow) {
      eyebrow.textContent = won ? "The clock ran out" : "The referee got everyone";
    }
    this.setPhaseSub("Match over");
    el("endTitle").textContent = won ? "Victory" : "Wipeout";
    const score = el("endScore");
    if (score) {
      score.innerHTML =
        '<span class="num">' + survivors + "</span>" +
        '<span class="of">of ' + registered + " still standing</span>";
    }
    el("endDetail").textContent = detail;

    if (won) {
      this.celebrate();
      if (this.doll) this.doll.bow();
    } else if (this.lightsOut) {
      this.lightsOut.dataset.on = "true";
    }
  }

  /**
   * Throw paper across the stage, in the arena's own colours.
   *
   * The palette is read live rather than hard-coded, so a burst under the
   * night theme is made of the night theme's green, violet and ink — three
   * colours that are guaranteed to read against whichever background the
   * visitor is on. The fourth is the referee's own paint, the ochre of her
   * dress, which does not change with the theme and should not change here.
   */
  celebrate() {
    if (lessMotion() || !this.fx || typeof burstConfetti !== "function") return;
    if (this.stopConfetti) this.stopConfetti();
    const paint = window.getComputedStyle(document.documentElement);
    const token = (name, fallback) =>
      paint.getPropertyValue(name).trim() || fallback;
    this.stopConfetti = burstConfetti(this.fx, {
      colors: [
        token("--green-lit", "#17a34a"),
        token("--accent", "#4a3aa7"),
        "#d98324",
        token("--ink", "#14201d"),
      ],
    });
  }

  /* -------------------------------------------------------------- render */

  render(tracks, now) {
    const armed = this.game ? this.game.armed(now) : false;
    if (this.doll) this.doll.setArmed(armed);
    if (this.live && window.rlHeroDemo && this.stage.dataset.phase === "RED") {
      window.rlHeroDemo.follow("RED");
    }
    if (this.game && this.stage.dataset.phase === "RED") {
      this.setPhaseSub(armed ? "Armed" : "Grace");
    }

    this.renderBoxes(tracks);
    this.renderRoster(tracks);
    this.renderRing(now);
  }

  renderBoxes(tracks) {
    const live = new Set();
    for (const track of tracks) {
      live.add(track.id);
      let node = this.overlay.querySelector('[data-track="' + track.id + '"]');
      if (!node) {
        node = document.createElement("div");
        node.className = "player-box meter-box";
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

      // The video is mirrored for the player's benefit, so the overlay has to
      // be mirrored with it — but by arithmetic, not by a transform, or every
      // label would come out backwards.
      node.style.left = fmt((1 - drawn.x2) * 100, 2) + "%";
      node.style.top = fmt(drawn.y1 * 100, 2) + "%";
      node.style.width = fmt((drawn.x2 - drawn.x1) * 100, 2) + "%";
      node.style.height = fmt((drawn.y2 - drawn.y1) * 100, 2) + "%";

      const player = this.game ? this.game.players.get(track.id) : null;
      const smoothed = this.judge.smoothed(track.id);
      const over = smoothed > this.threshold;
      node.dataset.state = player && !player.alive ? "out" : over ? "warn" : "safe";
      node.querySelector(".player-tag").textContent =
        "P" + track.id + "  " + fmt(smoothed, 3);
      const meter = node.querySelector(".meter");
      meter.dataset.over = over ? "true" : "false";
      meter.querySelector(".fill").style.width =
        fmt(Math.min(smoothed / (this.threshold * METER_SCALE), 1) * 100, 1) + "%";
    }

    for (const node of Array.from(this.overlay.children)) {
      if (!live.has(Number(node.dataset.track))) node.remove();
    }
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
      let row = this.rows.get(id);
      if (!row) {
        row = document.createElement("div");
        row.className = "meter-row";
        row.innerHTML =
          '<span class="who">P' + id + "</span>" +
          '<div class="meter"><div class="fill"></div><div class="tick"></div></div>' +
          '<span class="val"></span>';
        this.roster.appendChild(row);
        this.rows.set(id, row);
      }
      const player = this.game ? this.game.players.get(id) : null;
      const out = player ? !player.alive : false;
      const smoothed = out ? 0 : this.judge.smoothed(id);
      row.dataset.out = out ? "true" : "false";
      row.querySelector(".val").textContent = out ? "out" : fmt(smoothed, 3);
      const meter = row.querySelector(".meter");
      meter.dataset.over = !out && smoothed > this.threshold ? "true" : "false";
      meter.querySelector(".fill").style.width = out
        ? "0%"
        : fmt(Math.min(smoothed / (this.threshold * METER_SCALE), 1) * 100, 1) + "%";
    }

    if (ids.length === 0 && !this.roster.dataset.empty) {
      this.roster.dataset.empty = "1";
    } else if (ids.length > 0) {
      delete this.roster.dataset.empty;
    }
  }

  renderRing(now) {
    const ring = el("ring");
    if (!ring || !this.game) return;

    let fraction = 0;
    let value = "";
    let unit = "";
    if (this.game.phase === "COUNTDOWN") {
      const left = this.game.countdownLeft(now);
      fraction = left / this.config.countdownS;
      value = String(Math.ceil(left));
      unit = "starting";
    } else if (this.game.inMatch) {
      const left = this.game.timeLeft(now);
      fraction = left / this.config.durationS;
      value = String(Math.ceil(left));
      unit = "seconds left";
    } else {
      fraction = 0;
      value = String(this.game.aliveCount);
      unit = this.game.aliveCount === 1 ? "survivor" : "survivors";
    }

    ring.dataset.phase = this.game.phase;
    el("ringValue").textContent = value;
    el("ringUnit").textContent = unit;
    ring.querySelector(".sweep").style.strokeDashoffset = fmt(
      RING_CIRCUMFERENCE * (1 - Math.max(Math.min(fraction, 1), 0)),
      2
    );
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

  /** The two lines that tell a player what number they are being held to. */
  renderThresholdNotes() {
    const scale = DIFFICULTY[this.difficulty];
    const note = el("difficultyNote");
    if (note) {
      note.textContent =
        "cutoff " + fmt(this.presetThreshold, 4) + " body-fractions/s — " +
        (scale === 1
          ? "the lab-measured midpoint"
          : fmt(scale, 1) + "x the lab-measured midpoint of " +
            fmt(this.config.diffThreshold, 4));
    }

    const live = el("calibrationNote");
    if (!live) return;
    if (!this.calibrationDone) {
      live.textContent =
        "The countdown measures what your camera reads on a player standing " +
        "still, and lifts the cutoff if it has to.";
      return;
    }
    if (this.calibration === null) {
      live.textContent =
        "Nobody held still long enough during the countdown to measure your " +
        "camera. Cutoff " + fmt(this.threshold, 4) + ", straight from the preset.";
      return;
    }
    const raised = this.threshold > this.presetThreshold + 1e-12;
    live.textContent =
      "Calibrated to your camera: cutoff " + fmt(this.threshold, 4) +
      (raised
        ? ", raised from " + fmt(this.presetThreshold, 4) + " to clear a noise floor of " +
          fmt(this.calibration, 4) + "."
        : ". Your noise floor measured " + fmt(this.calibration, 4) +
          ", well under the preset.");
  }

  /* -------------------------------------------------------------- replay */

  /**
   * The recorded match — for anyone without a camera, or unwilling.
   *
   * Two views of the same seeded run share this scene. The primary one is
   * `docs/demo_match.mp4`: real HUD-annotated footage of `redlight play`
   * refereeing the benchmark video, boxes and verdicts and all — not staged,
   * the same run the README's `redlight benchmark` command reproduces. The
   * second is the older animated data view: the players, the eliminations,
   * their times and their reasons, read from `RL_DATA.benchmark.demo_match`.
   * Its light schedule is not recorded, so it is reconstructed here to the
   * only shape consistent with those calls — every "moved" call lands on an
   * armed red light, and every phase length stays inside the 1.5–3.0 s
   * bounds the run was configured with. The page says so where it is shown.
   *
   * Both views are built every time so switching between them is instant;
   * only the video actually costs bytes, and it is fetched once and cached
   * like any other same-origin asset.
   */
  startReplay() {
    this.stopReplay();
    // A replay and a live match must never run at once: one camera loop
    // still writing boxes over a replay would be indistinguishable from a
    // bug. Starting a replay ends any match in progress.
    if (this.rafId !== null) {
      this.stop();
      const button = el("btnPlay");
      if (button) button.disabled = false;
    }
    this.stage.dataset.live = "false";
    this.resetStage();
    this.live = true;
    const demo = this.data.benchmark.demo_match;
    const grace = demo.config.grace_s;
    const countdown = demo.config.countdown_s;
    const duration = demo.config.duration_s;
    const schedule = buildReplaySchedule(duration);

    if (this.idle) this.idle.hidden = true;
    this.endCard.dataset.show = "false";
    this.whyCard.dataset.show = "false";
    this.overlay.innerHTML = "";
    this.roster.innerHTML = "";
    this.rows.clear();

    const tokens = [];
    for (let i = 1; i <= demo.players; i += 1) {
      tokens.push({ id: i, out: false, reason: null });
    }

    const scene = document.createElement("div");
    scene.className = "replay-scene";
    scene.innerHTML =
      '<div class="replay-tabs" role="tablist" aria-label="Recorded match view">' +
      '<button type="button" class="replay-tab" data-view="video" aria-pressed="true">' +
      "Recorded video</button>" +
      '<button type="button" class="replay-tab" data-view="data" aria-pressed="false">' +
      "Data replay</button>" +
      "</div>" +
      '<div class="replay-view" data-view="video">' +
      '<video class="replay-video" src="demo_match.mp4" poster="demo_poster.jpg" ' +
      'muted controls loop playsinline></video>' +
      '<p class="replay-note">The referee calling a real recorded match: ' +
      "public benchmark footage of pedestrians in a courtyard, playing as the " +
      "runners. Every box, meter and verdict here is the engine's, not staged. " +
      "Seeded and reproducible — <code>redlight benchmark</code> from the " +
      "README's install steps replays this exact match.</p>" +
      "</div>" +
      '<div class="replay-view" data-view="data" hidden>' +
      '<div class="replay-field" id="replayField"></div>' +
      '<p class="replay-note">Data replay: ' + demo.players + " players, " +
      demo.eliminations.length + " calls, " + demo.survivors +
      " left standing. Light schedule reconstructed to fit the recorded calls.</p>" +
      "</div>";
    this.overlay.appendChild(scene);

    const tabs = scene.querySelectorAll(".replay-tab");
    const views = scene.querySelectorAll(".replay-view");
    const video = scene.querySelector(".replay-video");
    for (const tab of tabs) {
      tab.addEventListener("click", () => {
        for (const t of tabs) t.setAttribute("aria-pressed", String(t === tab));
        for (const v of views) v.hidden = v.dataset.view !== tab.dataset.view;
        if (tab.dataset.view === "video" && video) video.play().catch(() => {});
        else if (video) video.pause();
      });
    }
    if (video) video.play().catch(() => {});

    const field = scene.querySelector("#replayField");
    for (const token of tokens) {
      const node = document.createElement("div");
      node.className = "replay-token";
      node.dataset.token = String(token.id);
      node.innerHTML =
        '<span class="tk-body"></span><span class="tk-id">' + token.id + "</span>" +
        '<span class="tk-why"></span>';
      field.appendChild(node);
      token.node = node;
    }

    const startedAt = performance.now() / 1000;
    const step = () => {
      const elapsed = performance.now() / 1000 - startedAt;
      const matchT = elapsed - countdown;

      if (matchT < 0) {
        this.setPhase("COUNTDOWN");
        this.setPhaseSub("Countdown");
      } else if (matchT >= duration) {
        const outcome = demo.outcome === "victory" ? "VICTORY" : "WIPEOUT";
        this.setPhase(outcome);
        this.showEnd(
          outcome,
          demo.survivors,
          demo.players,
          "Recorded by the Python engine on the benchmark footage — not staged, "
            + "and reproducible from the README's install steps."
        );
        this.replayTimer = null;
        Probe.note("replay-end");
        return;
      } else {
        // `schedule` always spans exactly `[0, duration)`, so this lookup
        // succeeds for any `matchT` reachable here — the branch above already
        // returns once `matchT >= duration`. The fallback is a complete slot
        // object regardless, so a future change to the schedule's shape
        // cannot turn a missed lookup into a `slot.from` crash.
        const slot = schedule.find((s) => matchT >= s.from && matchT < s.to) || {
          phase: "RED",
          from: schedule.length ? schedule[schedule.length - 1].to : 0,
          to: duration,
        };
        const phase = slot.phase;
        if (this.stage.dataset.phase !== phase) this.setPhase(phase);
        if (phase === "RED") {
          this.setPhaseSub(matchT - slot.from >= grace ? "Armed" : "Grace");
          if (this.doll) this.doll.setArmed(matchT - slot.from >= grace);
        } else {
          this.setPhaseSub("Chant running");
          if (this.doll) this.doll.setArmed(false);
        }
        field.dataset.phase = phase;
      }

      for (const [trackId, at, reason] of demo.eliminations) {
        const token = tokens.find((t) => t.id === trackId);
        if (!token || token.out || matchT < at) continue;
        token.out = true;
        token.node.dataset.out = "true";
        // The recorded reason, shown against the player it belongs to: a
        // lost track and a caught movement are different failures and the
        // replay should not blur them into one.
        token.node.querySelector(".tk-why").textContent =
          reason === "moved" ? "moved" : "lost";
        this.setStatus("Player " + trackId + " out at " + fmt(at, 1) + "s: " + reason + ".");
      }

      const ring = el("ring");
      if (ring) {
        ring.dataset.phase = this.stage.dataset.phase;
        el("ringValue").textContent = String(
          Math.max(Math.ceil(matchT < 0 ? -matchT : duration - matchT), 0)
        );
        el("ringUnit").textContent = matchT < 0 ? "starting" : "seconds left";
        ring.querySelector(".sweep").style.strokeDashoffset = fmt(
          RING_CIRCUMFERENCE *
            (1 - Math.max(Math.min(matchT < 0 ? -matchT / countdown : 1 - matchT / duration, 1), 0)),
          2
        );
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
    const scene = this.overlay.querySelector(".replay-scene");
    if (scene) {
      const video = scene.querySelector(".replay-video");
      if (video) video.pause();
      scene.remove();
    }
  }
}

/* ------------------------------------------------------------------ boot */

/**
 * The three facts in the masthead, read out of the baked config.
 *
 * Typed into the markup they would be three more numbers to keep in step
 * with the engine by hand, and the first one to drift would be the one
 * nobody checks.
 */
function fillFacts(config) {
  const clock = el("factClock");
  if (clock) clock.textContent = "every " + fmt(config.sampleIntervalS, 1) + " s";
  const window96 = el("factWindow");
  if (window96) window96.textContent = config.window + " × " + config.window + " px";
  const cutoff = el("factCutoff");
  if (cutoff) cutoff.textContent = fmt(config.diffThreshold, 4) + " /s";
}

function initArena() {
  const data = window.RL_DATA;
  if (!data || !el("stage")) return;

  const arena = new Arena(data);
  window.rlArena = arena;
  arena.setDifficulty("standard");
  arena.setPhase("LOBBY");

  const play = el("btnPlay");
  if (play) play.addEventListener("click", () => arena.play());

  for (const trigger of document.querySelectorAll("[data-play-scroll]")) {
    trigger.addEventListener("click", () => {
      const stage = el("arena");
      if (stage) stage.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  for (const trigger of document.querySelectorAll("[data-replay]")) {
    trigger.addEventListener("click", () => {
      el("arena").scrollIntoView({ behavior: "smooth", block: "start" });
      arena.startReplay();
    });
  }

  const again = el("btnAgain");
  if (again) {
    again.addEventListener("click", () => {
      arena.stopReplay();
      arena.resetStage();
      arena.live = false;
      if (window.rlHeroDemo) window.rlHeroDemo.resume();
      arena.endCard.dataset.show = "false";
      arena.whyCard.dataset.show = "false";
      arena.stop();
      arena.overlay.innerHTML = "";
      arena.roster.innerHTML = "";
      arena.rows.clear();
      arena.setPhase("LOBBY");
      if (arena.idle) arena.idle.hidden = false;
      const button = el("btnPlay");
      if (button) button.disabled = false;
    });
  }

  const sound = el("btnSound");
  if (sound) {
    // The label always names the state the button is currently in — "Sound
    // on" when the chant is audible — never the action a click would take.
    // Read from `arena.chant.muted` rather than the button's own last
    // attribute, so the label matches reality even before anyone has
    // clicked it: the chant starts unmuted, so the button starts saying so.
    const syncSound = () => {
      const muted = arena.chant.muted;
      sound.setAttribute("aria-pressed", muted ? "false" : "true");
      sound.textContent = muted ? "Sound off" : "Sound on";
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
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  window.rlHeroDemo = initHeroDemo();
  if (window.RL_DATA) fillFacts(window.RL_DATA.config);
  initArena();
});
