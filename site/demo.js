/**
 * The demo match: drawn players, refereed by the real thing.
 *
 * The page opens on a match nobody has to join. Four players are drawn on a
 * canvas, on a sand field in front of a wall painted to look like sky, and
 * they play by the light: they walk while it is green and stop when it turns
 * red. Everything after the drawing is the arena itself. The canvas is the
 * camera frame, the scene's own record of where it drew each player stands
 * in for the pose model, and from there the crop path, the 0.1 s sampler,
 * the judge, the game rules, and every box, meter and call on the stage are
 * the same code a webcam match runs.
 *
 * Two of the four are written to get caught, in the two ways real players
 * are. One keeps walking long after the red light's grace runs out; the
 * other stops in time and then cannot keep still. Neither outcome is
 * written into the referee: this file only decides how the players move,
 * and the judge decides who moved. The other two stop within the grace, the
 * way a careful player does, and are never called.
 *
 * A player who stops, stops dead. A canvas redrawn from the same numbers
 * gives the same pixels, so a frozen drawn player scores exactly zero — what
 * a person holding perfectly still in front of a noiseless camera would
 * score, which no real camera is. That gap is why a live match measures the
 * camera during its countdown, and why the demo has nothing to measure.
 */

/** The drawn frame: the same 4:3 a webcam delivers, at a size that stays sharp. */
const DEMO_W = 960;
const DEMO_H = 720;

/** Where the painted wall meets the sand, as a fraction of the frame height. */
const DEMO_HORIZON = 0.46;

/** Player height at the far end of the field and at the near end, in pixels. */
const DEMO_FAR_HEIGHT = 168;
const DEMO_NEAR_HEIGHT = 300;

/** Where the players' feet are, far and near, as a fraction of the height. */
const DEMO_FAR_FEET = 0.6;
const DEMO_NEAR_FEET = 0.95;

/** How far a player gets in one second of walking, as a fraction of the field. */
const DEMO_WALK_RATE = 0.075;

/** Strides per second. */
const DEMO_STRIDE_HZ = 1.6;

/** When the fidget starts waving on the second red light, in seconds. */
const DEMO_FIDGET_FROM_S = 0.9;

/**
 * The cast, left to right. `react` is how long each one takes to stop when
 * the light turns red, always inside the grace; `plan` is how they play.
 *
 * The lanes are far enough apart that no player's box ever overlaps a
 * neighbour's, even at the end of the field. A neighbour walking through
 * your box while you hold still would be scored as you moving — the
 * referee's one real blind spot, which the measurements page shows — and
 * the demo is not the place to stage it.
 */
const DEMO_CAST = Object.freeze([
  { lane: 0.13, pace: 1.0, react: 0.16, skin: "#f1c7a0", hair: "#1c1714", plan: "steady" },
  { lane: 0.37, pace: 0.95, react: 0.3, skin: "#c98f68", hair: "#241913", plan: "late" },
  { lane: 0.63, pace: 1.06, react: 0.12, skin: "#8a5a3c", hair: "#0f0b09", plan: "steady" },
  { lane: 0.87, pace: 0.97, react: 0.24, skin: "#f3d6bb", hair: "#3b2a1d", plan: "fidget" },
]);

/**
 * Paint for the two arenas: daylight, and the same arena under floodlights.
 *
 * The referee only counts a pixel as changed when its grey level moves by
 * more than 25, so every colour a moving part of a player is painted in has
 * to sit well clear of the ground behind it, in both arenas. The night sand
 * is kept dark and its floodlight faint for exactly that reason: a lit,
 * mid-grey field would swallow the players' trousers, and a walking player
 * would barely score.
 */
const DEMO_PALETTES = Object.freeze({
  day: {
    sky: ["#63b6df", "#c9e8f5"],
    cloud: "rgba(255, 255, 255, 0.96)",
    trim: "#7fb6cf",
    sand: ["#ecd6a6", "#d6b47a"],
    speck: "rgba(120, 88, 38, 0.16)",
    glow: "rgba(255, 255, 255, 0.18)",
    shadow: "rgba(92, 64, 22, 0.28)",
  },
  night: {
    sky: ["#081a29", "#173a52"],
    cloud: "rgba(190, 225, 245, 0.10)",
    trim: "#24485e",
    sand: ["#3a2f1f", "#211b12"],
    speck: "rgba(0, 0, 0, 0.22)",
    glow: "rgba(255, 244, 214, 0.05)",
    shadow: "rgba(0, 0, 0, 0.42)",
  },
});

const SUIT = "#1f8f7d";
const SUIT_DARK = "#1a7a6b";
const SUIT_TRIM = "#5cc4b1";
const SUIT_OUT = "#7a878b";
const SUIT_OUT_DARK = "#5c686b";
const SHOE = "#1d1d1f";

function lerp(a, b, t) {
  return a + (b - a) * t;
}

/** A tiny seeded generator, so the sand looks the same on every visit. */
function demoRandom(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function demoTheme() {
  const forced = document.documentElement.dataset.theme;
  if (forced === "dark" || forced === "light") return forced === "dark" ? "night" : "day";
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "night"
    : "day";
}

function roundRect(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

/** One limb: a rounded bar from a joint, rotated about that joint. */
function limb(ctx, x, y, length, width, angle, colour) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.fillStyle = colour;
  roundRect(ctx, -width / 2, 0, width, length, width / 2);
  ctx.fill();
  ctx.restore();
}

/** The painted arena, drawn once per match and reused for every frame. */
function paintArena(palette) {
  const canvas = document.createElement("canvas");
  canvas.width = DEMO_W;
  canvas.height = DEMO_H;
  const ctx = canvas.getContext("2d");
  const horizon = Math.round(DEMO_H * DEMO_HORIZON);

  const sky = ctx.createLinearGradient(0, 0, 0, horizon);
  sky.addColorStop(0, palette.sky[0]);
  sky.addColorStop(1, palette.sky[1]);
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, DEMO_W, horizon);

  // Flat painted clouds: a long base and two domes, the way a wall painter
  // draws them.
  ctx.fillStyle = palette.cloud;
  for (const [cx, cy, w] of [
    [150, 92, 190],
    [470, 58, 130],
    [770, 120, 220],
    [360, 196, 110],
    [660, 238, 90],
  ]) {
    const h = w * 0.2;
    roundRect(ctx, cx - w / 2, cy, w, h, h / 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx - w * 0.14, cy + 2, w * 0.2, Math.PI, 0);
    ctx.arc(cx + w * 0.14, cy + 2, w * 0.15, Math.PI, 0);
    ctx.fill();
  }

  ctx.fillStyle = palette.trim;
  ctx.fillRect(0, horizon - 4, DEMO_W, 12);

  const sand = ctx.createLinearGradient(0, horizon, 0, DEMO_H);
  sand.addColorStop(0, palette.sand[0]);
  sand.addColorStop(1, palette.sand[1]);
  ctx.fillStyle = sand;
  ctx.fillRect(0, horizon + 8, DEMO_W, DEMO_H - horizon - 8);

  const glow = ctx.createRadialGradient(DEMO_W / 2, horizon, 20, DEMO_W / 2, horizon, DEMO_W * 0.7);
  glow.addColorStop(0, palette.glow);
  glow.addColorStop(1, "rgba(255, 255, 255, 0)");
  ctx.fillStyle = glow;
  ctx.fillRect(0, horizon + 8, DEMO_W, DEMO_H - horizon - 8);

  const random = demoRandom(4242);
  ctx.fillStyle = palette.speck;
  for (let i = 0; i < 260; i += 1) {
    const y = lerp(horizon + 14, DEMO_H - 4, Math.pow(random(), 0.8));
    const x = random() * DEMO_W;
    const size = lerp(1, 3.4, (y - horizon) / (DEMO_H - horizon));
    ctx.beginPath();
    ctx.ellipse(x, y, size * 1.6, size * 0.7, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  return canvas;
}

/**
 * Build the scene on a canvas and hand back what the arena needs from it.
 *
 * The returned object is two things at once. `step(now, game)` moves the
 * players by the light and redraws the frame; `detect()` is the pose-source
 * interface the arena already calls on a camera match, answering with the
 * box around each drawn player.
 *
 * @param {HTMLCanvasElement} canvas Where the match is drawn.
 */
function createDemoScene(canvas) {
  if (!canvas || !canvas.getContext) return null;
  canvas.width = DEMO_W;
  canvas.height = DEMO_H;
  const ctx = canvas.getContext("2d");

  let arena = null;
  let palette = DEMO_PALETTES.day;
  let players = [];
  let lastNow = null;
  let lastPhase = null;
  let redCount = 0;

  /** Where a player is drawn this frame, and what they are doing. */
  function figure(player) {
    const p = player.progress;
    const height = lerp(DEMO_FAR_HEIGHT, DEMO_NEAR_HEIGHT, p);
    const spread = lerp(0.74, 1, p);
    return {
      x: (0.5 + (player.lane - 0.5) * spread) * DEMO_W,
      feet: lerp(DEMO_FAR_FEET, DEMO_NEAR_FEET, p) * DEMO_H,
      h: height,
    };
  }

  /** The angle each arm hangs at: swinging on a walk, flailing on a fidget. */
  function armAngles(player) {
    if (player.waving) {
      const t = player.waveT * 10;
      return [Math.PI - 0.5 - 0.55 * Math.sin(t), -(Math.PI - 0.5) + 0.55 * Math.sin(t + 1.4)];
    }
    const swing = player.walking ? 0.16 * Math.sin(player.stride) : 0.04;
    return [0.14 + swing, -0.14 + swing];
  }

  function drawPlayer(player) {
    const { x, feet, h } = figure(player);
    const out = player.out;
    const suit = out ? SUIT_OUT : SUIT;
    const suitDark = out ? SUIT_OUT_DARK : SUIT_DARK;
    const bob = player.walking ? -0.012 * h * Math.abs(Math.sin(player.stride)) : 0;
    const lift = player.walking ? 0.06 * h : 0;
    const leftLift = lift * Math.max(0, Math.sin(player.stride));
    const rightLift = lift * Math.max(0, -Math.sin(player.stride));

    const hip = feet - 0.44 * h + bob;
    const shoulder = feet - 0.77 * h + bob;
    const legW = 0.11 * h;
    const torsoW = 0.3 * h;
    const headR = 0.1 * h;

    // Shadow on the sand.
    ctx.fillStyle = palette.shadow;
    ctx.beginPath();
    ctx.ellipse(x, feet + 0.01 * h, 0.2 * h, 0.035 * h, 0, 0, Math.PI * 2);
    ctx.fill();

    // Legs and shoes.
    for (const [side, raised] of [
      [-1, leftLift],
      [1, rightLift],
    ]) {
      const legX = x + side * 0.075 * h - legW / 2;
      const legBottom = feet - raised;
      ctx.fillStyle = suitDark;
      roundRect(ctx, legX, hip - 0.02 * h, legW, legBottom - hip, legW * 0.45);
      ctx.fill();
      ctx.fillStyle = SHOE;
      roundRect(ctx, legX - 0.012 * h, legBottom - 0.035 * h, legW + 0.024 * h, 0.05 * h, 0.02 * h);
      ctx.fill();
    }

    // Arms, behind the torso's edges.
    const [left, right] = armAngles(player);
    const armLength = 0.34 * h;
    const armW = 0.085 * h;
    limb(ctx, x - torsoW / 2 + armW * 0.35, shoulder + 0.03 * h, armLength, armW, left, suit);
    limb(ctx, x + torsoW / 2 - armW * 0.35, shoulder + 0.03 * h, armLength, armW, right, suit);

    // Torso, a collar stripe, and the number patch.
    ctx.fillStyle = suit;
    roundRect(ctx, x - torsoW / 2, shoulder, torsoW, hip - shoulder + 0.03 * h, 0.07 * h);
    ctx.fill();
    ctx.fillStyle = out ? "#9aa5a8" : SUIT_TRIM;
    ctx.fillRect(x - 0.05 * h, shoulder, 0.1 * h, 0.03 * h);
    ctx.fillStyle = "#ffffff";
    roundRect(ctx, x - 0.075 * h, shoulder + 0.08 * h, 0.15 * h, 0.1 * h, 0.012 * h);
    ctx.fill();
    ctx.fillStyle = "#2a2a2a";
    ctx.fillRect(x - 0.045 * h, shoulder + 0.112 * h, 0.09 * h, 0.036 * h);

    // Head and hair.
    const headY = shoulder - headR * 0.92;
    ctx.fillStyle = player.skin;
    ctx.beginPath();
    ctx.arc(x, headY, headR, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = player.hair;
    ctx.beginPath();
    ctx.arc(x, headY, headR * 1.04, Math.PI * 1.04, Math.PI * 1.96);
    ctx.quadraticCurveTo(x, headY - headR * 0.35, x - headR * 1.02, headY - headR * 0.1);
    ctx.fill();
  }

  /** The box a pose model would report: every drawn part, padded the same way. */
  function boxFor(player) {
    const { x, feet, h } = figure(player);
    const shoulder = feet - 0.77 * h;
    const headTop = shoulder - 0.1 * h * 1.92;
    const [left, right] = armAngles(player);
    const reach = 0.34 * h + 0.04 * h;
    const leftTip = {
      x: x - 0.15 * h - Math.sin(left) * reach,
      y: shoulder + 0.03 * h + Math.cos(left) * reach,
    };
    const rightTip = {
      x: x + 0.15 * h - Math.sin(right) * reach,
      y: shoulder + 0.03 * h + Math.cos(right) * reach,
    };
    let x1 = Math.min(x - 0.2 * h, leftTip.x, rightTip.x);
    let x2 = Math.max(x + 0.2 * h, leftTip.x, rightTip.x);
    let y1 = Math.min(headTop, leftTip.y, rightTip.y);
    let y2 = feet + 0.02 * h;
    const padX = (x2 - x1) * BOX_PAD;
    const padY = (y2 - y1) * BOX_PAD;
    x1 -= padX;
    x2 += padX;
    y1 -= padY;
    y2 += padY;
    return {
      x1: Math.max(x1 / DEMO_W, 0),
      y1: Math.max(y1 / DEMO_H, 0),
      x2: Math.min(x2 / DEMO_W, 1),
      y2: Math.min(y2 / DEMO_H, 1),
    };
  }

  /**
   * Decide, from the light and each player's plan, who is moving now.
   *
   * The two who get it wrong keep getting it wrong until the referee notices
   * — the late walker for the whole of the first red light, the fidget from
   * just under a second into the second — so how soon they are called is up
   * to the judge, not to a timer here.
   */
  function plan(player, game, now) {
    player.waving = false;
    if (player.out || !game || !game.inMatch) {
      player.walking = false;
      return;
    }
    if (game.phase === "GREEN") {
      player.walking = true;
      return;
    }
    // A red light: everyone reacts, and two of them get it wrong.
    const elapsed = game.phaseElapsed(now);
    if (player.plan === "late" && redCount === 1) {
      player.walking = true;
      return;
    }
    player.walking = elapsed < player.react;
    if (player.plan === "fidget" && redCount === 2 && elapsed >= DEMO_FIDGET_FROM_S) {
      player.waving = true;
    }
  }

  return {
    canvas,

    /** Put everyone back on the start line, in the current theme's arena. */
    reset() {
      palette = DEMO_PALETTES[demoTheme()];
      arena = paintArena(palette);
      players = DEMO_CAST.map((member) => ({
        ...member,
        progress: 0,
        stride: 0,
        waveT: 0,
        walking: false,
        waving: false,
        out: false,
      }));
      lastNow = null;
      lastPhase = null;
      redCount = 0;
      this.draw();
    },

    /**
     * Advance the players by the light, then redraw.
     *
     * Track ids are handed out in the order players are first reported, and
     * the arena starts every match with a fresh tracker, so player `i` here
     * is track `i + 1` there.
     */
    step(now, game) {
      // Capped, so a tab that was asleep does not wake to players who have
      // teleported; a quarter of a second still lets a slow device keep pace.
      const dt = lastNow === null ? 0 : Math.min(Math.max(now - lastNow, 0), 0.25);
      lastNow = now;
      // Red lights are counted as the players see them, frame by frame. If
      // the page stalls through a whole light, the game still replays it,
      // but nobody moved during it and nobody was judged; counting it would
      // spend the late walker's turn on a light no one saw.
      const phase = game ? game.phase : "LOBBY";
      if (phase === "RED" && lastPhase !== "RED") redCount += 1;
      lastPhase = phase;

      players.forEach((player, index) => {
        const record = game ? game.players.get(index + 1) : null;
        player.out = Boolean(record && !record.alive);
        plan(player, game, now);
        if (player.walking) {
          player.progress = Math.min(player.progress + dt * DEMO_WALK_RATE * player.pace, 1);
          player.stride += dt * Math.PI * 2 * DEMO_STRIDE_HZ * player.pace;
        }
        if (player.waving) player.waveT += dt;
      });
      this.draw();
    },

    draw() {
      if (!arena) return;
      ctx.drawImage(arena, 0, 0);
      // Far players first, so nearer ones are painted over them.
      const order = players.slice().sort((a, b) => a.progress - b.progress);
      for (const player of order) drawPlayer(player);
    },

    /** The pose-source interface: one box per drawn player, in cast order. */
    detect() {
      return players.map(boxFor);
    },

    close() {},
  };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { createDemoScene, DEMO_CAST };
}
