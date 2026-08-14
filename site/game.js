/**
 * Game state machine: phases, timing, and the elimination rules, in the browser.
 *
 * A port of `src/redlight/game.py`, held to it by `tests/test_js_parity.py`:
 * the same scenarios are driven through both implementations and the event
 * streams are compared move for move.
 *
 * The referee runs a fixed loop of light phases. Registration happens once,
 * at the start of the countdown: only players who signed up before the
 * countdown began can ever be eliminated or win, so anyone the camera picks
 * up afterward is not part of the match. During red light, a short grace
 * window right after the light turns forgives the motion needed to actually
 * come to a stop — a player already mid-stride when the light changes is not
 * punished for a reaction time nobody has zero of. Once that window closes
 * the phase is armed, and any confirmed motion counts. Losing the tracker's
 * lock on a registered player eliminates them under either light, on the
 * same footing as anyone else who can no longer be seen well enough to judge.
 *
 * This module is pure logic: no video, no wall clock, no timers. Every method
 * that cares about time is handed `now` by the caller as a plain number, and
 * the light schedule is drawn from a single seeded PRNG built at
 * construction, so the whole match is reproducible from that seed and a
 * sequence of `now` values.
 *
 * On the PRNG: the browser uses mulberry32 where Python uses its Mersenne
 * Twister, so the same seed does *not* give the same phase lengths in both
 * languages — and it does not need to. What the parity suite pins is the
 * rules; determinism from a seed is pinned within the browser alone. The
 * scenarios that compare the two languages set `phaseMinS === phaseMaxS`,
 * which makes `uniform(a, a) === a` and takes the PRNGs out of it entirely.
 */

/** Where the match stands right now. */
const Phase = Object.freeze({
  LOBBY: "LOBBY",
  COUNTDOWN: "COUNTDOWN",
  GREEN: "GREEN",
  RED: "RED",
  VICTORY: "VICTORY",
  WIPEOUT: "WIPEOUT",
});

/** What kind of thing just happened. */
const EventType = Object.freeze({
  PHASE_CHANGED: "PHASE_CHANGED",
  PLAYER_ELIMINATED: "PLAYER_ELIMINATED",
  GAME_OVER: "GAME_OVER",
});

/**
 * One thing that happened during `update`, for a caller to react to.
 *
 * `trackId` and `reason` are only ever set on PLAYER_ELIMINATED events.
 * `reason` is `"moved"` — caught by the judge while the current red light was
 * armed — or `"left_arena"` — the tracker lost the player entirely.
 *
 * Frozen, like its Python counterpart: an event is a record of something
 * that already happened, and a HUD should not be able to rewrite history by
 * mutating one.
 */
function makeEvent(type, phase, trackId = null, reason = null) {
  return Object.freeze({ type, phase, trackId, reason });
}

/**
 * A small, fast, seeded PRNG.
 *
 * Chosen because it needs no dependency, is exactly reproducible across
 * every JavaScript engine (all arithmetic is on 32-bit integers via
 * `Math.imul` and `>>>`), and passes well enough for picking light lengths.
 * Cryptographic quality is irrelevant here; reproducibility is the whole
 * requirement.
 *
 * @param {number} seed Any integer.
 * @returns {function(): number} A generator of floats in [0, 1).
 */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Defaults mirroring `GameConfig` in `src/redlight/config.py`. */
const DEFAULT_CONFIG = Object.freeze({
  // null means "pick one at construction", matching GameConfig's own `seed:
  // int | None = None`. A fixed default would hand every visitor the
  // identical run of light lengths, so the second game anyone plays would
  // be the first one again — and a player who noticed could learn the
  // schedule and beat the referee by counting rather than by holding still.
  // The resolved seed is stored back on `config.seed`, so a match that did
  // something interesting can still be replayed exactly by passing it back.
  seed: null,
  countdownS: 3.0,
  durationS: 60.0,
  phaseMinS: 2.0,
  phaseMaxS: 5.0,
  graceS: 0.6,
  threshold: 0.0767,
  diffThreshold: 0.1455,
  confirmFrames: 3,
  smoothing: 0.5,
});

/**
 * Check a config the way `GameConfig.validate` does, reporting every fault.
 *
 * The strictly-positive phase lengths are not a style rule. `_flipPhase`
 * catches up to a late `now` with a while loop, and a zero-length phase
 * would never advance the boundary past it — the tab would hang rather than
 * misbehave visibly. Refusing the config is how that stays impossible.
 *
 * @throws {Error} listing every violation found.
 */
function validateConfig(config) {
  const violations = [];

  if (!(config.countdownS >= 0)) {
    violations.push(`countdownS must be non-negative, got ${config.countdownS}`);
  }
  if (!(config.durationS >= 0)) {
    violations.push(`durationS must be non-negative, got ${config.durationS}`);
  }
  if (!(config.phaseMinS > 0)) {
    violations.push(`phaseMinS must be positive, got ${config.phaseMinS}`);
  }
  if (!(config.phaseMaxS > 0)) {
    violations.push(`phaseMaxS must be positive, got ${config.phaseMaxS}`);
  }
  if (!(config.graceS >= 0)) {
    violations.push(`graceS must be non-negative, got ${config.graceS}`);
  }
  if (config.phaseMinS > config.phaseMaxS) {
    violations.push(
      `phaseMinS (${config.phaseMinS}) must not exceed phaseMaxS (${config.phaseMaxS})`
    );
  }

  if (violations.length > 0) {
    throw new Error(
      "Configuration errors:\n" + violations.map((v) => `  - ${v}`).join("\n")
    );
  }
}

/**
 * Runs the light schedule and applies the elimination rules.
 *
 * Nothing here reads a clock or a camera; every method that needs the current
 * time takes `now` as an argument. The phase schedule comes from one seeded
 * generator built in the constructor and consumed one draw at a time, on
 * every light change — so two games built with the same seed and driven with
 * the same sequence of `now` values produce the identical schedule.
 */
class Game {
  /**
   * @param {object} config Partial config; anything omitted takes its
   *   default. Omit `seed` (or pass null) to get a fresh random match; the
   *   seed actually used is written back to `game.config.seed` so the run
   *   can be reproduced later by passing that value in.
   */
  constructor(config = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    validateConfig(this.config);

    if (this.config.seed === null || this.config.seed === undefined) {
      // A full 32 bits, since that is what mulberry32 consumes.
      this.config.seed = Math.floor(Math.random() * 4294967296);
    }

    this.phase = Phase.LOBBY;
    /** @type {Map<number, {trackId: number, alive: boolean, eliminatedAt: number|null}>} */
    this.players = new Map();
    this._rng = mulberry32(this.config.seed);
    this._phaseStartedAt = null;
    this._phaseEndAt = null;
    this._countdownEndAt = null;
    this._matchStartedAt = null;
  }

  /**
   * Register players and open the countdown.
   *
   * Only the ids passed here are ever eligible to be eliminated or to win the
   * match. Can only be called while the game is in LOBBY, and needs at least
   * one player to register.
   *
   * @throws {Error} If the game is not in LOBBY, or the roster is empty.
   */
  start(now, playerTrackIds) {
    if (this.phase !== Phase.LOBBY) {
      throw new Error(`start() requires phase LOBBY, game is in ${this.phase}`);
    }
    const ids = Array.from(playerTrackIds);
    if (ids.length === 0) {
      throw new Error("start() requires at least one playerTrackId");
    }

    this.players = new Map(
      ids.map((trackId) => [trackId, { trackId, alive: true, eliminatedAt: null }])
    );
    this.phase = Phase.COUNTDOWN;
    this._phaseStartedAt = now;
    this._countdownEndAt = now + this.config.countdownS;
    return [makeEvent(EventType.PHASE_CHANGED, this.phase)];
  }

  /**
   * Advance the match to `now`, applying eliminations and light changes.
   *
   * `violations` are the ids of players the judge has confirmed are moving on
   * this tick; they only cost a player their spot while the current red light
   * is armed — during green, or in red's grace window, motion is forgiven.
   * `missing` are registered players whose track has died; they are
   * eliminated under either light, since a player the camera can no longer
   * see cannot be judged fairly. Ids that are not registered, or already
   * eliminated, are silently ignored in both lists.
   *
   * Before the countdown finishes, or after the match has ended, this is a
   * no-op that returns no events.
   *
   * @returns {Array<object>} Eliminations first, then a match outcome if this
   *   tick ended the match, then a light change if the match is still running.
   */
  update(now, violations = null, missing = null) {
    if (this.phase === Phase.LOBBY || this.finished) {
      return [];
    }
    if (this.phase === Phase.COUNTDOWN) {
      return this._updateCountdown(now);
    }
    return this._updateMatch(now, violations || [], missing || []);
  }

  _updateCountdown(now) {
    if (now < this._countdownEndAt) {
      return [];
    }
    this.phase = Phase.GREEN;
    // The match clock starts when the countdown was *due* to end, not when
    // the caller happened to notice — a dropped frame must not lengthen the
    // match.
    this._matchStartedAt = this._countdownEndAt;
    this._phaseStartedAt = this._countdownEndAt;
    this._scheduleNextPhaseEnd();
    return [makeEvent(EventType.PHASE_CHANGED, this.phase)];
  }

  _updateMatch(now, violations, missing) {
    const events = this._applyEliminations(now, violations, missing);

    // Order matters: eliminations land first, so a final elimination on the
    // same tick the clock expires rules the match a wipeout, not a victory.
    if (this.aliveCount === 0) {
      events.push(...this._endMatch(Phase.WIPEOUT));
      return events;
    }

    if (now - this._matchStartedAt >= this.config.durationS) {
      events.push(...this._endMatch(Phase.VICTORY));
      return events;
    }

    events.push(...this._flipPhase(now));
    return events;
  }

  _applyEliminations(now, violations, missing) {
    const events = [];
    if (this.armed(now)) {
      for (const trackId of violations) {
        events.push(...this._eliminate(trackId, now, "moved"));
      }
    }
    for (const trackId of missing) {
      events.push(...this._eliminate(trackId, now, "left_arena"));
    }
    return events;
  }

  _eliminate(trackId, now, reason) {
    const player = this.players.get(trackId);
    if (player === undefined || !player.alive) {
      return [];
    }
    player.alive = false;
    player.eliminatedAt = now;
    return [makeEvent(EventType.PLAYER_ELIMINATED, this.phase, trackId, reason)];
  }

  _flipPhase(now) {
    const events = [];
    // `validateConfig` requires phaseMinS and phaseMaxS to be strictly
    // positive, so every drawn phase length is > 0 and this loop always
    // advances _phaseEndAt past `now` in a finite number of iterations — it
    // cannot spin forever even on a large time jump, which is what a
    // backgrounded tab hands it on the way back.
    while (now >= this._phaseEndAt) {
      this.phase = this.phase === Phase.GREEN ? Phase.RED : Phase.GREEN;
      this._phaseStartedAt = this._phaseEndAt;
      this._scheduleNextPhaseEnd();
      events.push(makeEvent(EventType.PHASE_CHANGED, this.phase));
    }
    return events;
  }

  _scheduleNextPhaseEnd() {
    const { phaseMinS, phaseMaxS } = this.config;
    // Same form as Python's `random.uniform(a, b)`, which matters for the
    // parity scenarios: at phaseMinS === phaseMaxS this is exactly phaseMinS
    // in both languages, whatever the generator returns.
    const length = phaseMinS + (phaseMaxS - phaseMinS) * this._rng();
    this._phaseEndAt = this._phaseStartedAt + length;
  }

  _endMatch(phase) {
    this.phase = phase;
    return [
      makeEvent(EventType.PHASE_CHANGED, phase),
      makeEvent(EventType.GAME_OVER, phase),
    ];
  }

  /**
   * Whether a red light currently counts motion as an elimination.
   *
   * The light itself switches instantly, but a player mid-stride when it
   * happens needs a moment to stop; for `graceS` after the light turns red,
   * motion is forgiven. After that the phase is armed.
   */
  armed(now) {
    return this.phase === Phase.RED && this.phaseElapsed(now) >= this.config.graceS;
  }

  /**
   * Seconds remaining on the match clock.
   *
   * Outside green/red — before the match has started, or after it has ended —
   * this is just the configured `durationS`, since no match clock is running
   * to count down from it.
   */
  timeLeft(now) {
    if (!this.inMatch) {
      return this.config.durationS;
    }
    return Math.max(this.config.durationS - (now - this._matchStartedAt), 0.0);
  }

  /** Seconds remaining before the countdown ends, or 0 outside it. */
  countdownLeft(now) {
    if (this.phase !== Phase.COUNTDOWN) {
      return 0.0;
    }
    return Math.max(this._countdownEndAt - now, 0.0);
  }

  /** Seconds since the current phase began, or 0 before a match has started. */
  phaseElapsed(now) {
    if (this._phaseStartedAt === null) {
      return 0.0;
    }
    return now - this._phaseStartedAt;
  }

  /** How many registered players have not been eliminated. */
  get aliveCount() {
    let count = 0;
    for (const player of this.players.values()) {
      if (player.alive) count += 1;
    }
    return count;
  }

  /** Whether a light is currently live (green or red). */
  get inMatch() {
    return this.phase === Phase.GREEN || this.phase === Phase.RED;
  }

  /** Whether the match has reached a final outcome (victory or wipeout). */
  get finished() {
    return this.phase === Phase.VICTORY || this.phase === Phase.WIPEOUT;
  }
}

// Dual environment: inlined into one <script> tag by the site build, and
// required directly by the parity suite under node.
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    Phase,
    EventType,
    makeEvent,
    mulberry32,
    validateConfig,
    DEFAULT_CONFIG,
    Game,
  };
}
