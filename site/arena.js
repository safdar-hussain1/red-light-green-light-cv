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
 */

/** Multipliers on the measured diff threshold, offered as difficulty. */
const DIFFICULTY = Object.freeze({
  forgiving: 2.0,
  standard: 1.0,
  ruthless: 0.6,
});

/** How long at least one player must be in frame before the countdown opens. */
const REGISTRATION_HOLD_S = 1.0;

/** Samples kept per player, for the trace shown when they are called out. */
const TRACE_MEMORY = 40;

/** Meter full-scale, as a multiple of the live threshold. */
const METER_SCALE = 2.0;

/** Circumference of the countdown ring's circle (r = 54). */
const RING_CIRCUMFERENCE = 2 * Math.PI * 54;

/** Phases the replay steps through, chosen to fit the recorded calls. */
const REPLAY_SCHEDULE = Object.freeze([
  { phase: "GREEN", from: 0.0, to: 2.1 },
  { phase: "RED", from: 2.1, to: 4.5 },
  { phase: "GREEN", from: 4.5, to: 6.6 },
  { phase: "RED", from: 6.6, to: 8.8 },
  { phase: "GREEN", from: 8.8, to: 10.5 },
  { phase: "RED", from: 10.5, to: 12.0 },
]);

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
    this.idle = el("stageIdle");
    this.whyCard = el("whyCard");
    this.endCard = el("endCard");
    this.roster = el("roster");
    this.status = el("arenaStatus");

    this.doll = window.createDoll ? createDoll(el("arenaDoll")) : null;
    this.chant = new ChantPlayer(data.chant);

    this.difficulty = "standard";
    this.threshold = this.config.diffThreshold;
    this.judge = new MotionJudge(
      this.threshold,
      this.config.confirmFrames,
      this.config.smoothing
    );
    this.sampler = new FrameSampler(this.config.sampleIntervalS);
    this.tracker = new BoxTracker();
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

  setPhase(phase) {
    this.stage.dataset.phase = phase;
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
    this.judge.reset();
    this.sampler.reset();
    this.traces.clear();
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

  stop() {
    if (this.rafId !== null) cancelAnimationFrame(this.rafId);
    this.rafId = null;
    this.chant.stop();
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
    const violations = this.score(tracks, now);

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
   * @returns {Array<number>} Ids the judge has now confirmed are moving.
   */
  score(tracks, now) {
    const crops = {};
    for (const track of tracks) {
      const window96 = cropWindow(this.video, track.box, this.cropCtx);
      if (window96) crops[track.id] = window96;
    }

    const pair = this.sampler.offer(crops, now);
    if (!pair) return [];

    const violations = [];
    for (const key of Object.keys(pair.cur)) {
      const previous = pair.prev[key];
      if (!previous) continue;
      const id = Number(key);
      const score = diffScoreWindow(previous, pair.cur[key], pair.dt);
      if (this.judge.update(id, score)) violations.push(id);

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

  eliminate(event, now) {
    this.safeAudio(() => this.chant.buzz());
    this.flash.dataset.on = "false";
    // Reflow between the two writes, or the animation does not restart when
    // two players go out within a few frames of each other.
    void this.flash.offsetWidth;
    this.flash.dataset.on = "true";

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
    Probe.note("eliminated");
  }

  endMatch(phase) {
    this.safeAudio(() => this.chant.stop());
    this.safeAudio(() => this.chant.sting(phase === "VICTORY"));
    const survivors = this.game ? this.game.aliveCount : 0;
    this.endCard.dataset.show = "true";
    this.endCard.dataset.outcome = phase;
    el("endTitle").textContent = phase === "VICTORY" ? "Victory" : "Wipeout";
    el("endDetail").textContent =
      phase === "VICTORY"
        ? survivors === 1
          ? "One player stood still long enough. The clock ran out first."
          : survivors + " players stood still long enough. The clock ran out first."
        : "Every registered player was called out before the clock ran down.";
    Probe.note("end:" + phase);
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

      // The video is mirrored for the player's benefit, so the overlay has to
      // be mirrored with it — but by arithmetic, not by a transform, or every
      // label would come out backwards.
      node.style.left = fmt((1 - track.box.x2) * 100, 2) + "%";
      node.style.top = fmt(track.box.y1 * 100, 2) + "%";
      node.style.width = fmt((track.box.x2 - track.box.x1) * 100, 2) + "%";
      node.style.height = fmt((track.box.y2 - track.box.y1) * 100, 2) + "%";

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

  /* --------------------------------------------------------- difficulty */

  setDifficulty(name) {
    this.difficulty = name;
    this.threshold = this.config.diffThreshold * DIFFICULTY[name];
    this.judge.threshold = this.threshold;

    const note = el("difficultyNote");
    if (note) {
      const scale = DIFFICULTY[name];
      note.textContent =
        "threshold " + fmt(this.threshold, 4) + " body-fractions/s" +
        (scale === 1
          ? " — the measured default"
          : " (" + fmt(scale, 1) + "x the measured default)");
    }
    for (const button of document.querySelectorAll("#difficulty button")) {
      button.setAttribute("aria-pressed", button.dataset.level === name ? "true" : "false");
    }
  }

  /* -------------------------------------------------------------- replay */

  /**
   * The recorded match, animated — for anyone without a camera, or unwilling.
   *
   * The players, the eliminations, their times and their reasons are the
   * recorded ones from `RL_DATA.benchmark.demo_match`: a real run of the
   * Python engine over the benchmark footage. The light schedule is not
   * recorded, so it is reconstructed here to the only shape consistent with
   * those calls — every "moved" call lands on an armed red light, and every
   * phase length stays inside the 1.5–3.0 s bounds the run was configured
   * with. The page says so where the replay is shown.
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
    this.live = true;
    const demo = this.data.benchmark.demo_match;
    const grace = demo.config.grace_s;
    const countdown = demo.config.countdown_s;
    const duration = demo.config.duration_s;

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
      '<div class="replay-field" id="replayField"></div>' +
      '<p class="replay-note">Recorded match: ' + demo.players + " players, " +
      demo.eliminations.length + " calls, " + demo.survivors +
      " left standing. Light schedule reconstructed to fit the recorded calls.</p>";
    this.overlay.appendChild(scene);
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
        this.setPhase(demo.outcome === "victory" ? "VICTORY" : "WIPEOUT");
        this.endCard.dataset.show = "true";
        this.endCard.dataset.outcome = demo.outcome === "victory" ? "VICTORY" : "WIPEOUT";
        el("endTitle").textContent = demo.outcome === "victory" ? "Victory" : "Wipeout";
        el("endDetail").textContent =
          demo.survivors + " of " + demo.players + " players were still standing when the "
          + "clock ran out. Recorded by the Python engine on the benchmark footage.";
        this.replayTimer = null;
        Probe.note("replay-end");
        return;
      } else {
        const slot = REPLAY_SCHEDULE.find((s) => matchT >= s.from && matchT < s.to);
        const phase = slot ? slot.phase : "RED";
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
    if (scene) scene.remove();
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
    sound.addEventListener("click", () => {
      const muted = sound.getAttribute("aria-pressed") === "true";
      sound.setAttribute("aria-pressed", muted ? "false" : "true");
      sound.textContent = muted ? "Sound on" : "Sound off";
      arena.chant.setMuted(!muted);
    });
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
