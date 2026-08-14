/**
 * The victory burst: paper, thrown twice, falling under gravity.
 *
 * Hand-written canvas particles rather than a library, for the same reason
 * everything else on this page is hand-written: it is forty lines of
 * arithmetic, and a dependency would cost more bytes than the whole effect.
 *
 * Two things keep it from reading as generic celebration filler. The paper
 * is the arena's own palette — the light's green, the interactive violet,
 * the ochre of the referee's dress and the cream of her collar — so the
 * burst is made of the page rather than dropped onto it. And it is thrown
 * from the two bottom corners inward, the way a party popper actually fires,
 * instead of raining from the top of the frame.
 *
 * Each piece is a rectangle drawn on its own spin, and its width is scaled
 * by the cosine of that spin, so it turns edge-on and vanishes to a line
 * twice a rotation. That single trick is what makes flat rectangles read as
 * paper with two sides.
 *
 * Nothing here loops. The burst runs itself out in a couple of seconds and
 * clears the canvas, so a finished match leaves no animation running behind
 * an end card nobody is looking at any more.
 */

/** How long a burst lives before the canvas is wiped, in seconds. */
const CONFETTI_LIFE_S = 2.6;

/** Downward acceleration, in canvas pixels per second squared. */
const CONFETTI_GRAVITY = 900;

/**
 * Fire one burst across a canvas.
 *
 * @param {HTMLCanvasElement} canvas Target; resized to its own layout box.
 * @param {{pieces?: number, colors?: Array<string>}} [options]
 * @returns {function(): void} Stops the burst early and clears the canvas.
 */
function burstConfetti(canvas, options) {
  const settings = options || {};
  const ctx = canvas.getContext("2d");
  if (!ctx) return function () {};

  const width = canvas.clientWidth || canvas.width;
  const height = canvas.clientHeight || canvas.height;
  // Cap the backing store at 2x. Past that a burst costs more to composite
  // than it is worth on a phone, and nobody can see the difference on paper
  // this small.
  const scale = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(Math.round(width * scale), 1);
  canvas.height = Math.max(Math.round(height * scale), 1);
  ctx.setTransform(scale, 0, 0, scale, 0, 0);

  const colors = settings.colors || ["#17a34a", "#4a3aa7", "#d98324", "#f7f2e6"];
  const count = settings.pieces || 90;
  const pieces = [];

  for (let i = 0; i < count; i += 1) {
    // Alternating cannons: even pieces from the left corner, odd from the
    // right, each aimed up and inward with a wide spread.
    const fromLeft = i % 2 === 0;
    const spread = (Math.random() - 0.5) * 0.9;
    const angle = (fromLeft ? -Math.PI / 3 : (-Math.PI * 2) / 3) + spread;
    const speed = 620 + Math.random() * 520;
    pieces.push({
      x: fromLeft ? width * 0.06 : width * 0.94,
      y: height * 0.96,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      size: 5 + Math.random() * 6,
      spin: (Math.random() - 0.5) * 14,
      turn: Math.random() * Math.PI * 2,
      color: colors[i % colors.length],
    });
  }

  let raf = null;
  let last = null;
  let elapsed = 0;

  function stop() {
    if (raf !== null) cancelAnimationFrame(raf);
    raf = null;
    ctx.clearRect(0, 0, width, height);
  }

  function frame(now) {
    if (last === null) last = now;
    // Clamped: a backgrounded tab hands back a dt of several seconds, which
    // would teleport every piece off the canvas in one step.
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    elapsed += dt;

    ctx.clearRect(0, 0, width, height);
    const fade = Math.max(1 - elapsed / CONFETTI_LIFE_S, 0);

    for (const piece of pieces) {
      piece.vy += CONFETTI_GRAVITY * dt;
      // Air drag, so the paper slows into its fall instead of arcing like a
      // thrown stone.
      piece.vx *= 1 - 1.1 * dt;
      piece.x += piece.vx * dt;
      piece.y += piece.vy * dt;
      piece.turn += piece.spin * dt;

      ctx.save();
      ctx.globalAlpha = fade;
      ctx.translate(piece.x, piece.y);
      ctx.rotate(piece.turn * 0.35);
      ctx.fillStyle = piece.color;
      // Edge-on twice a rotation: the whole reason these read as paper.
      const face = Math.abs(Math.cos(piece.turn));
      ctx.fillRect(
        (-piece.size * face) / 2,
        -piece.size / 2,
        Math.max(piece.size * face, 0.6),
        piece.size * 0.6
      );
      ctx.restore();
    }

    if (elapsed >= CONFETTI_LIFE_S) {
      stop();
      return;
    }
    raf = requestAnimationFrame(frame);
  }

  raf = requestAnimationFrame(frame);
  return stop;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { burstConfetti };
}
