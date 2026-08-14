/**
 * Finding the players: pose detection, boxes, crops, and identity over time.
 *
 * Everything between the camera and the scoring kernel lives here. The
 * kernel itself is `judge.js`, which is pinned to Python by golden fixtures;
 * this file is the part that is *not* pinned, and it says so out loud —
 * `crop_window` in the desktop engine clamps the box, resamples with
 * OpenCV's INTER_AREA and applies a 3x3 Gaussian blur, while the browser
 * reaches its 96x96 windows through a canvas. Both then score identically.
 * That distinction is the whole reason this code is a separate file from the
 * judge.
 *
 * ## The runtime dependency, and why it carries no hash
 *
 * MediaPipe's tasks-vision package is fetched from jsdelivr at an exact
 * pinned version, and only after a visitor has clicked play and granted a
 * camera. It carries no subresource-integrity hash and it cannot: the
 * loader resolves its own wasm binary and its model file at run time from a
 * base path it is handed, so there is no fixed set of bytes known at build
 * time to hash. SRI has nothing to attach to for a wasm module fetched that
 * way. What is done instead is to pin the version so the URL cannot drift,
 * to fetch it only on an explicit opt-in, and to make sure nothing else on
 * the page depends on it — every published number, the referee lab, the
 * replay and the selftest all run with this file never loading anything.
 *
 * ## Identity
 *
 * Players are matched frame to frame by box overlap, greedily, best pair
 * first. That is enough for the four players this arena admits, standing in
 * a row facing a camera; it is not a re-identification system and does not
 * pretend to be. Two players who swap places while crossing will swap ids,
 * and the honest consequence is that the referee would judge them as each
 * other — which in a game where everyone is standing still is a situation
 * that does not arise.
 */

/** Exact pinned version of the pose runtime. Never a floating tag. */
const VISION_VERSION = "0.10.14";

const VISION_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@" + VISION_VERSION;

/** The lite pose model: smallest of the three, and the only one a laptop can run four of. */
const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/" +
  "pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

/** How many players the arena admits at once. */
const MAX_POSES = 4;

/**
 * How far each box is grown beyond the landmarks, as a fraction of its size.
 *
 * Landmarks sit on the skeleton, so a box drawn tight to them cuts through
 * the body: hair, shoulders and the outside of a swinging arm all fall
 * outside it. Since the score is a fraction of the *window*, a box that
 * clips the body would leave the most mobile parts of a player permanently
 * out of frame, and holding still would be easier than it should be.
 */
const BOX_PAD = 0.15;

/** Below this, a landmark is a guess rather than an observation. */
const MIN_VISIBILITY = 0.5;

/** Overlap below which two boxes are not the same player from one frame to the next. */
const MIN_MATCH_IOU = 0.15;

/** Side of the analysis window. Must match the judge; the judge enforces it. */
const CROP_SIDE = 96;

/**
 * Load the pose runtime and return something that can find players in a frame.
 *
 * Resolves to an object with `detect(video, timestampMs)` returning boxes in
 * normalised video coordinates. Rejects if the runtime or the model cannot
 * be fetched, which the arena presents as "no pose runtime" rather than as a
 * broken page.
 *
 * `window.RL_POSE_FACTORY` is honoured first, and exists so the headless
 * verification in `scripts/verify_site.py` can drive the whole play path —
 * camera, judge, game, HUD and audio — without a network fetch of a wasm
 * blob. It is a seam, not a feature: nothing in the page sets it.
 */
async function createPoseSource() {
  if (typeof window.RL_POSE_FACTORY === "function") {
    return window.RL_POSE_FACTORY();
  }

  const vision = await import(VISION_BASE + "/vision_bundle.mjs");
  const fileset = await vision.FilesetResolver.forVisionTasks(VISION_BASE + "/wasm");
  const landmarker = await vision.PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numPoses: MAX_POSES,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputSegmentationMasks: false,
  });

  return {
    detect(video, timestampMs) {
      const result = landmarker.detectForVideo(video, timestampMs);
      const boxes = [];
      for (const landmarks of result.landmarks || []) {
        const box = boxFromLandmarks(landmarks);
        if (box) boxes.push(box);
      }
      return boxes;
    },
    close() {
      landmarker.close();
    },
  };
}

/**
 * Bounding box around one pose's landmarks, padded and clamped to the frame.
 *
 * @returns {{x1: number, y1: number, x2: number, y2: number}|null} Normalised
 *   box, or null if the pose has no confidently visible landmarks.
 */
function boxFromLandmarks(landmarks) {
  let x1 = Infinity;
  let y1 = Infinity;
  let x2 = -Infinity;
  let y2 = -Infinity;
  let seen = 0;

  for (const point of landmarks) {
    // `visibility` is absent on some builds; an absent score is treated as
    // visible rather than dropping every landmark and losing the player.
    const visible = point.visibility === undefined || point.visibility >= MIN_VISIBILITY;
    if (!visible) continue;
    seen += 1;
    if (point.x < x1) x1 = point.x;
    if (point.y < y1) y1 = point.y;
    if (point.x > x2) x2 = point.x;
    if (point.y > y2) y2 = point.y;
  }

  if (seen < 4 || !(x2 > x1) || !(y2 > y1)) return null;

  const padX = (x2 - x1) * BOX_PAD;
  const padY = (y2 - y1) * BOX_PAD;
  return {
    x1: Math.max(x1 - padX, 0),
    y1: Math.max(y1 - padY, 0),
    x2: Math.min(x2 + padX, 1),
    y2: Math.min(y2 + padY, 1),
  };
}

/** Intersection over union of two normalised boxes. */
function iou(a, b) {
  const left = Math.max(a.x1, b.x1);
  const top = Math.max(a.y1, b.y1);
  const right = Math.min(a.x2, b.x2);
  const bottom = Math.min(a.y2, b.y2);
  if (!(right > left && bottom > top)) return 0;

  const overlap = (right - left) * (bottom - top);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return overlap / (areaA + areaB - overlap);
}

/**
 * Keeps stable ids on players across frames.
 *
 * A track that goes unmatched is not dropped straight away — a detector
 * blinking for two frames is normal, and losing a track eliminates a player
 * under either light. It survives `graceS` of silence before it is reported
 * as gone.
 */
class BoxTracker {
  /**
   * @param {number} graceS How long an unmatched track survives, in seconds.
   * @param {number} maxTracks Cap on simultaneous players.
   */
  constructor(graceS = 1.2, maxTracks = MAX_POSES) {
    this.graceS = graceS;
    this.maxTracks = maxTracks;
    /** @type {Array<{id: number, box: object, lastSeen: number}>} */
    this.tracks = [];
    this._nextId = 1;
  }

  /**
   * Fold this frame's boxes into the tracks.
   *
   * @returns {{tracks: Array<object>, lost: Array<number>}} The live tracks
   *   after matching, and the ids that just aged out.
   */
  update(boxes, now) {
    const pairs = [];
    this.tracks.forEach((track, ti) => {
      boxes.forEach((box, bi) => {
        const score = iou(track.box, box);
        if (score >= MIN_MATCH_IOU) pairs.push({ ti, bi, score });
      });
    });
    // Best overlap first, so a confident pairing is never stolen by a weaker
    // one that happened to be considered earlier.
    pairs.sort((a, b) => b.score - a.score);

    const usedTrack = new Set();
    const usedBox = new Set();
    for (const pair of pairs) {
      if (usedTrack.has(pair.ti) || usedBox.has(pair.bi)) continue;
      usedTrack.add(pair.ti);
      usedBox.add(pair.bi);
      this.tracks[pair.ti].box = boxes[pair.bi];
      this.tracks[pair.ti].lastSeen = now;
    }

    boxes.forEach((box, bi) => {
      if (usedBox.has(bi)) return;
      if (this.tracks.length >= this.maxTracks) return;
      this.tracks.push({ id: this._nextId++, box, lastSeen: now });
    });

    const lost = [];
    this.tracks = this.tracks.filter((track) => {
      if (now - track.lastSeen <= this.graceS) return true;
      lost.push(track.id);
      return false;
    });

    return { tracks: this.tracks, lost };
  }

  reset() {
    this.tracks = [];
  }
}

/** Weight on the newest box when smoothing a player's scoring rectangle. */
const BOX_SMOOTHING = 0.3;

/** One box eased toward another, corner by corner. */
function easeBox(previous, next, alpha) {
  return {
    x1: previous.x1 + (next.x1 - previous.x1) * alpha,
    y1: previous.y1 + (next.y1 - previous.y1) * alpha,
    x2: previous.x2 + (next.x2 - previous.x2) * alpha,
    y2: previous.y2 + (next.y2 - previous.y2) * alpha,
  };
}

/**
 * Decides which rectangle each player is actually scored through.
 *
 * This exists because of a bug a real player found: standing perfectly still
 * got them called out. The judge was not at fault — on stable crops it flags
 * nothing. The crop was. Landmarks wobble by a couple of pixels every frame
 * even on a statue, the box is derived fresh from those landmarks on every
 * one of them, and a 96x96 window cut from a box that moved is a window
 * whose *contents* moved. The referee was measuring its own detector.
 *
 * Two guards, in order.
 *
 * **Smoothing.** Each player's rectangle is an exponential moving average of
 * the boxes reported for them, so a single frame's wobble moves the window
 * by a fraction of itself instead of all of it.
 *
 * **Freezing.** Smoothing shrinks jitter; it does not remove it. So whenever
 * the players are supposed to be standing still — the countdown and every
 * red light, grace included — each player's scoring rectangle is pinned
 * where it was when that phase began and does not move again until the light
 * turns green. A still player's window is then pixel-identical frame to
 * frame, and the only thing that can change inside it is the player.
 *
 * Freezing does not blind the referee to a player who genuinely walks during
 * red. The rectangle is pinned around where their body *was*, so the moment
 * they move, body pixels leave that region and background arrives in its
 * place — a large change, inside the frozen window, in exactly the direction
 * that gets them called. If anything a pinned rectangle is the more sensitive
 * of the two: a rectangle that follows a walker partly cancels their
 * translation, which is the classic way a box tracker makes walking look
 * like standing.
 */
class BoxStabilizer {
  /** @param {number} alpha Weight on the newest box, in (0, 1]. */
  constructor(alpha = BOX_SMOOTHING) {
    this.alpha = alpha;
    /** @type {Map<number, object>} Smoothed box per player, always current. */
    this._smooth = new Map();
    /** @type {Map<number, object>} Pinned rectangle per player, while held. */
    this._held = new Map();
  }

  /**
   * Fold this frame's tracks in and hand back the rectangle to score each by.
   *
   * @param {Array<{id: number, box: object}>} tracks Live tracks this frame.
   * @param {boolean} hold True while scoring rectangles must not move.
   * @returns {Map<number, object>} Player id to the rectangle to crop.
   */
  update(tracks, hold) {
    const live = new Set();
    for (const track of tracks) {
      live.add(track.id);
      const previous = this._smooth.get(track.id);
      // Smoothing runs even while frozen. The pinned rectangle is what gets
      // scored, but the smoothed one has to be current the moment the light
      // turns green, or the box would snap across the stage.
      this._smooth.set(
        track.id,
        previous === undefined ? { ...track.box } : easeBox(previous, track.box, this.alpha)
      );
    }

    for (const id of Array.from(this._smooth.keys())) {
      if (live.has(id)) continue;
      this._smooth.delete(id);
      this._held.delete(id);
    }

    if (!hold) this._held.clear();

    const rects = new Map();
    for (const id of live) {
      if (hold && !this._held.has(id)) {
        // First frame of the hold, or a player who arrived mid-hold: pin them
        // where they are now.
        this._held.set(id, { ...this._smooth.get(id) });
      }
      rects.set(id, hold ? this._held.get(id) : this._smooth.get(id));
    }
    return rects;
  }

  /** A player's smoothed box — what to draw, which follows them under any light. */
  smoothed(trackId) {
    return this._smooth.get(trackId) || null;
  }

  /** Forget every player, ready for a fresh match. */
  reset() {
    this._smooth.clear();
    this._held.clear();
  }
}

/**
 * Cut one player's box out of the frame as a 96x96 grayscale window.
 *
 * The canvas does the resampling, and the luma weights are the ones OpenCV's
 * BGR2GRAY uses, so the two paths agree on what grey means even though they
 * disagree on how the pixels were resampled. The result is a
 * `Uint8ClampedArray`, which `diffScoreWindow` accepts directly.
 *
 * @returns {Uint8ClampedArray|null} 9216 bytes, or null if the box has
 *   collapsed to nothing — which happens for a frame or two when a player
 *   walks off the edge, and must not be scored as stillness.
 */
function cropWindow(video, box, canvasCtx) {
  const frameW = video.videoWidth;
  const frameH = video.videoHeight;
  if (!frameW || !frameH) return null;

  const sx = box.x1 * frameW;
  const sy = box.y1 * frameH;
  const sw = (box.x2 - box.x1) * frameW;
  const sh = (box.y2 - box.y1) * frameH;
  if (!(sw >= 1 && sh >= 1)) return null;

  canvasCtx.drawImage(video, sx, sy, sw, sh, 0, 0, CROP_SIDE, CROP_SIDE);
  const rgba = canvasCtx.getImageData(0, 0, CROP_SIDE, CROP_SIDE).data;

  const gray = new Uint8ClampedArray(CROP_SIDE * CROP_SIDE);
  for (let i = 0, p = 0; i < gray.length; i += 1, p += 4) {
    gray[i] = Math.round(0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2]);
  }
  return gray;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    VISION_VERSION,
    VISION_BASE,
    POSE_MODEL_URL,
    MAX_POSES,
    BOX_PAD,
    BOX_SMOOTHING,
    boxFromLandmarks,
    iou,
    easeBox,
    BoxTracker,
    BoxStabilizer,
    cropWindow,
    createPoseSource,
  };
}
