/**
 * Motion scoring and the elimination decision, in the browser.
 *
 * This is the desktop engine's scoring kernel, ported so a visitor can play
 * with nothing installed, and the port is tested rather than asserted:
 * `tests/test_js_parity.py` replays golden 96x96 window pairs through this
 * file under node and requires the numbers to match Python bit for bit.
 *
 * Be precise about what that covers. `diff_score_window`, `MotionJudge` and
 * `FrameSampler` are ported here and pinned by fixtures. What comes *before*
 * them is not: `crop_window` in `src/redlight/judge.py` clamps the player's
 * box, resamples with OpenCV's INTER_AREA, and applies a 3x3 Gaussian blur,
 * and the browser reaches its 96x96 windows by its own canvas path. So the
 * proven claim is that both sides score identical windows identically — not
 * that both sides derive identical windows from an identical camera frame.
 * Anything that changes how a window is *produced* is outside what these
 * fixtures can vouch for.
 *
 * The arithmetic is deliberately plain — integer comparisons and two
 * divisions, no blurring, no resizing, no floating-point thresholds. That is
 * what makes bit-exact agreement across two languages a reasonable thing to
 * ask for: an integer count divided by 9216 and then by dt is exactly
 * rounded under IEEE-754 in both.
 *
 * Motion is measured in body-fractions per second. Cropping at the player's
 * box and resampling to a fixed window is what makes a step mean the same
 * thing whether the player is near the camera or far from it — distance and
 * resolution divide out. The diff metric is not made frame-rate independent
 * by dividing by dt, though: a changed-pixel count grows faster than
 * linearly with displacement. `FrameSampler` is what carries that, by
 * holding the measurement interval fixed instead of trying to rescale it.
 */

/** Side length of the fixed analysis window every player crop is resampled to. */
const WINDOW = 96;

/** Pixels in one window. Precomputed because it divides every single score. */
const WINDOW_AREA = WINDOW * WINDOW; // 9216

/** How far a pixel's brightness must swing before it counts as changed. */
const DIFF_PIXEL_DELTA = 25;

/** How often the referee actually scores, regardless of how fast frames arrive. */
const SAMPLE_INTERVAL_S = 0.1;

/**
 * Slack on the sampling interval, so a float division can't hide a frame.
 *
 * A nanosecond is orders of magnitude below any real frame period, so this
 * never lets an early frame through — it only keeps a timestamp that is
 * exactly on the boundary in arithmetic from landing just under it in binary.
 */
const TIMESTAMP_TOLERANCE_S = 1e-9;

/**
 * True for the byte-array types a real 96x96 grayscale window arrives in.
 *
 * This stands in for Python's `dtype != np.uint8` check, and it is a check
 * worth keeping rather than a formality: a plain `Array` of floats would
 * score perfectly happily here and give an answer the desktop engine would
 * never produce.
 *
 * `Uint8ClampedArray` is accepted alongside `Uint8Array` purely as a
 * convenience. It is *not* a zero-copy path from the canvas:
 * `getImageData` returns interleaved RGBA at 4 bytes per pixel, so a caller
 * always has to walk it into a single-channel 96x96 window first. Accepting
 * the clamped type just means that conversion pass can write into the
 * natural array type without a further copy to convert it.
 */
function isByteWindow(value) {
  return value instanceof Uint8Array || value instanceof Uint8ClampedArray;
}

/**
 * Fraction of the window that changed brightness, per second.
 *
 * Counts the pixels whose brightness moved by more than DIFF_PIXEL_DELTA,
 * expresses that as a fraction of the window, and divides by the elapsed
 * time.
 *
 * The absolute difference is taken on Numbers, never on bytes. Reading an
 * element out of a Uint8Array yields a Number in 0..255 and the subtraction
 * happens in double precision, so `100 - 126` is `-26` and not the `230` an
 * unsigned byte subtraction would produce. The golden fixtures pin that
 * behaviour in both directions.
 *
 * @param {Uint8Array|Uint8ClampedArray} prevU8 Earlier 96x96 window, row-major.
 * @param {Uint8Array|Uint8ClampedArray} curU8 Later 96x96 window, row-major.
 * @param {number} dt Seconds between the two windows; must be positive.
 * @returns {number} Changed fraction per second.
 * @throws {Error} If dt is not positive, or either window is not a
 *   9216-element byte array. This is the function the port is held against,
 *   so a wrong-shaped input is a bug to surface, not something to quietly
 *   score.
 */
function diffScoreWindow(prevU8, curU8, dt) {
  if (!(typeof dt === "number" && dt > 0)) {
    throw new Error(`dt must be positive, got ${dt}`);
  }

  for (const [name, win] of [
    ["prevU8", prevU8],
    ["curU8", curU8],
  ]) {
    if (!isByteWindow(win)) {
      throw new Error(
        `${name} must be a Uint8Array or Uint8ClampedArray, got ${
          win === null ? "null" : typeof win
        }`
      );
    }
    if (win.length !== WINDOW_AREA) {
      throw new Error(
        `${name} must be a ${WINDOW}x${WINDOW} window (${WINDOW_AREA} bytes), got ${win.length}`
      );
    }
  }

  let changed = 0;
  for (let i = 0; i < WINDOW_AREA; i += 1) {
    const delta = prevU8[i] - curU8[i];
    if (delta > DIFF_PIXEL_DELTA || delta < -DIFF_PIXEL_DELTA) {
      changed += 1;
    }
  }

  // Two separate divisions, in this order, matching Python exactly. Folding
  // them into one (dividing by WINDOW_AREA * dt) would round differently.
  return changed / WINDOW_AREA / dt;
}

/**
 * Paces scoring on a fixed clock instead of on whatever the camera manages.
 *
 * A capture loop hands over every frame it grabs. This holds them back and
 * releases a pair only once at least `intervalS` has passed since the last
 * pair it released, so comparisons are always made across roughly the same
 * slice of real time. A laptop running at 60 frames a second and a phone
 * managing 24 end up judging the same motion over the same intervals.
 *
 * That is what makes the diff metric fair across frame rates: rather than
 * trying to rescale a changed-pixel count that does not scale linearly, the
 * interval it is measured over is simply held fixed.
 *
 * The comparison is always against the last *released* frame, never the last
 * frame offered, so nothing is silently measured over a shorter gap.
 *
 * The tolerance on the interval is load-bearing rather than cosmetic. A
 * camera running at a clean multiple of the interval — 30 fps against a
 * 0.1 s clock is the obvious case — produces timestamps like `i / 30`, and
 * in binary floating point the gap between three of those lands a fraction
 * of an ulp under 0.1 about half the time. Compared exactly, the sampler
 * would reject every other frame that is precisely on the boundary and pace
 * itself at half the rate it was asked for.
 */
class FrameSampler {
  /** @param {number} intervalS Seconds between released pairs. */
  constructor(intervalS = SAMPLE_INTERVAL_S) {
    this.intervalS = intervalS;
    this._lastFrame = null;
    this._lastTs = null;
  }

  /**
   * Hand the sampler a frame and its timestamp in seconds.
   *
   * @param {*} frame Whatever the caller wants back later — typically the
   *   96x96 window, but the sampler never looks inside it.
   * @param {number} ts Timestamp in seconds.
   * @returns {{prev: *, cur: *, dt: number}|null} The pair to score and the
   *   real interval between them, or null when it is not yet time. The very
   *   first frame only primes the sampler and returns null.
   */
  offer(frame, ts) {
    if (this._lastTs === null) {
      this._lastFrame = frame;
      this._lastTs = ts;
      return null;
    }

    const dt = ts - this._lastTs;
    if (dt < this.intervalS - TIMESTAMP_TOLERANCE_S) {
      return null;
    }

    const prev = this._lastFrame;
    this._lastFrame = frame;
    this._lastTs = ts;
    return { prev, cur: frame, dt };
  }

  /** Drop the held frame so the next offer primes a fresh interval. */
  reset() {
    this._lastFrame = null;
    this._lastTs = null;
  }
}

/**
 * Turns a stream of per-frame scores into eliminate / don't-eliminate calls.
 *
 * Raw scores flicker: an estimate spikes on a lighting change, a detection
 * box wobbles by a few pixels. Calling someone out on a single spiky frame
 * is the fastest way to make the referee feel unfair, so two guards sit in
 * front of the decision.
 *
 * An exponential moving average per player smooths the score, weighting the
 * newest frame by `smoothing`. On top of that, the smoothed score has to
 * stay above the threshold for `confirmFrames` frames in a row before anyone
 * is called out; a single frame back at or below the threshold puts the
 * streak to zero. The player has to actually be moving, not have flickered
 * once.
 */
class MotionJudge {
  /**
   * @param {number} threshold Movement, in body-fractions per second, that counts as moving.
   * @param {number} confirmFrames Consecutive frames above threshold before a call is made.
   * @param {number} smoothing Weight on the newest score in the moving average, in (0, 1].
   */
  constructor(threshold, confirmFrames = 3, smoothing = 0.5) {
    this.threshold = threshold;
    this.confirmFrames = confirmFrames;
    this.smoothing = smoothing;
    this._ema = new Map();
    this._streak = new Map();
  }

  /**
   * Feed one frame's score for one player and get the verdict.
   *
   * @returns {boolean} True if this player has now been above the threshold
   *   for `confirmFrames` frames running.
   */
  update(trackId, score) {
    const previous = this._ema.get(trackId);
    // The first score a player is ever given seeds their average — there is
    // no earlier frame to blend it with, and starting from zero would hand
    // everyone a free frame of movement.
    const ema =
      previous === undefined
        ? score
        : this.smoothing * score + (1 - this.smoothing) * previous;
    this._ema.set(trackId, ema);

    if (ema > this.threshold) {
      this._streak.set(trackId, (this._streak.get(trackId) || 0) + 1);
    } else {
      this._streak.set(trackId, 0);
    }

    return this._streak.get(trackId) >= this.confirmFrames;
  }

  /** The player's current smoothed score, or 0.0 if they have none yet. */
  smoothed(trackId) {
    const ema = this._ema.get(trackId);
    return ema === undefined ? 0.0 : ema;
  }

  /**
   * Forget every player's history, ready for a fresh round.
   *
   * Rounds must not inherit a stale baseline: a player who was sprinting
   * when the last round ended starts the next one as still as everyone else.
   */
  reset() {
    this._ema.clear();
    this._streak.clear();
  }
}

// Dual environment: the arena loads this file inline in one <script> tag,
// while the parity suite requires it under node. Neither should need a
// bundler, so the exports are attached only when a CommonJS module object is
// actually present.
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    WINDOW,
    WINDOW_AREA,
    DIFF_PIXEL_DELTA,
    SAMPLE_INTERVAL_S,
    TIMESTAMP_TOLERANCE_S,
    diffScoreWindow,
    FrameSampler,
    MotionJudge,
  };
}
