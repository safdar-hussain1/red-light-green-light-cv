/**
 * Everything on the page that is not the arena: the two views, the pictures
 * in "How the referee works", and the accuracy figures.
 *
 * **Two views, one file.** The page is a single self-contained HTML file, so
 * the measurements live in it too, in a view of their own that the address
 * bar can point at: `#measurements`, or any section inside it. The game view
 * is everything else. Moving between them is a hash change, so the back
 * button, a shared link and a reload all land where they should.
 *
 * **Numbers are read, never typed.** The accuracy figures come out of the
 * baked benchmark results at load, the same way every chart on the
 * measurements view does, so the page cannot quietly disagree with
 * `reports/benchmark_results.json`.
 */

function pageLessMotion() {
  return (
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

function pageVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* ------------------------------------------------------------------ views */

function viewFor(hash) {
  const id = decodeURIComponent((hash || "").replace(/^#/, ""));
  if (!id) return { view: "home", id: "" };
  if (id === "measurements") return { view: "measurements", id };
  const target = document.getElementById(id);
  if (target && target.closest("#measurements")) return { view: "measurements", id };
  return { view: "home", id };
}

function initViews() {
  const views = Array.from(document.querySelectorAll(".view"));
  if (views.length < 2) return;
  let current = "home";

  const show = (name) => {
    for (const view of views) view.hidden = view.dataset.view !== name;
    for (const link of document.querySelectorAll(".nav a[data-route]")) {
      if (link.dataset.route === "measurements" && name === "measurements") {
        link.setAttribute("aria-current", "page");
      } else {
        link.removeAttribute("aria-current");
      }
    }
    window.dispatchEvent(new CustomEvent("rl-view", { detail: name }));
  };

  const route = () => {
    const { view, id } = viewFor(window.location.hash);
    const changed = view !== current;
    if (changed) {
      show(view);
      current = view;
    }
    // A view that was hidden a moment ago could not be scrolled to by the
    // browser's own anchor handling, so the jump is made here — instantly
    // when the view changed, since smooth-scrolling across a page that has
    // just been swapped out reads as the page jumping twice.
    // "instant", not "auto": auto defers to the stylesheet's smooth scrolling.
    const behavior = changed || pageLessMotion() ? "instant" : "smooth";
    if (view === "measurements" && id === "measurements") {
      window.scrollTo({ top: 0, behavior });
      const title = document.getElementById("measurementsTitle");
      if (title && changed) title.focus({ preventScroll: true });
    } else if (id) {
      const target = document.getElementById(id);
      if (target) target.scrollIntoView({ behavior, block: "start" });
    } else if (changed) {
      window.scrollTo({ top: 0, behavior });
    }
  };

  window.addEventListener("hashchange", route);
  if (window.location.hash) route();
}

/* ------------------------------------------------------- the step pictures */

/**
 * Step two's picture: one player, twice, as the referee sees them — a 12 by
 * 12 stand-in for the real 96 by 96 window. Between the two looks an arm
 * went up and a foot stepped out; the cells that differ are the movement.
 */
const ART_THEN = [
  "....####....",
  "....####....",
  ".....##.....",
  "..########..",
  "..########..",
  "..#.####.#..",
  "..#.####.#..",
  "..#.####.#..",
  "....#..#....",
  "....#..#....",
  "....#..#....",
  "...##..##...",
];

const ART_NOW = [
  "#...####....",
  ".#..####....",
  "..#..##.....",
  "...#######..",
  "...#######..",
  "....####.#..",
  "....####.#..",
  "....####.#..",
  "....#...#...",
  "....#...#...",
  "....#....#..",
  "...##....##.",
];

const SVG_NS = "http://www.w3.org/2000/svg";

function pixelRect(x, y, on) {
  const rect = document.createElementNS(SVG_NS, "rect");
  rect.setAttribute("x", String(x * 8 + 0.5));
  rect.setAttribute("y", String(y * 8 + 0.5));
  rect.setAttribute("width", "7");
  rect.setAttribute("height", "7");
  rect.setAttribute("rx", "1");
  if (on) rect.setAttribute("class", "on");
  return rect;
}

function drawPixelArt() {
  const then = document.getElementById("artThen");
  const now = document.getElementById("artNow");
  const changed = document.getElementById("artChanged");
  if (!then || !now || !changed) return;
  for (let y = 0; y < 12; y += 1) {
    for (let x = 0; x < 12; x += 1) {
      const was = ART_THEN[y][x] === "#";
      const is = ART_NOW[y][x] === "#";
      then.appendChild(pixelRect(x, y, was));
      now.appendChild(pixelRect(x, y, is));
      if (was !== is) changed.appendChild(pixelRect(x, y, false));
    }
  }
}

/** The pictures only play while they are on screen, and never under reduced motion. */
function initStepArt() {
  drawPixelArt();
  const arts = document.querySelectorAll(".step-art");
  if (pageLessMotion() || !("IntersectionObserver" in window)) return;
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        entry.target.classList.toggle("is-playing", entry.isIntersecting);
      }
    },
    { threshold: 0.35 }
  );
  for (const art of arts) observer.observe(art);
}

/* --------------------------------------------------------------- accuracy */

/** Columns in each unit chart: 4,410 samples make a 147 by 30 block. */
const DOT_COLUMNS = 147;

/**
 * One square per sample, so a reader can see the misses rather than take a
 * percentage on trust. `segments` fill the grid in order, each a count of
 * squares in one colour, so the last segment gathers at the end of the
 * block, where it can be counted.
 *
 * Squares are snapped to whole device pixels. At four or five pixels a
 * square, a fractional pitch makes some gaps vanish and others double, and
 * the block shimmers like a moiré instead of reading as squares.
 */
function drawDots(mount, total, segments) {
  if (!mount) return;
  let canvas = mount.querySelector("canvas");
  if (!canvas) {
    canvas = document.createElement("canvas");
    mount.appendChild(canvas);
  }
  const rows = Math.ceil(total / DOT_COLUMNS);
  const available = mount.clientWidth;
  if (!available) return;
  const ratio = window.devicePixelRatio || 1;
  const pitch = Math.max(Math.floor((available * ratio) / DOT_COLUMNS), 2);
  const gap = pitch >= 5 ? 2 : 1;
  canvas.width = pitch * DOT_COLUMNS;
  canvas.height = pitch * rows;
  canvas.style.width = canvas.width / ratio + "px";
  canvas.style.height = canvas.height / ratio + "px";
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  let index = 0;
  for (const segment of segments) {
    ctx.fillStyle = pageVar(segment.colour);
    for (let k = 0; k < segment.count && index < total; k += 1, index += 1) {
      const x = (index % DOT_COLUMNS) * pitch;
      const y = Math.floor(index / DOT_COLUMNS) * pitch;
      ctx.fillRect(x, y, pitch - gap, pitch - gap);
    }
  }
}

function initAccuracy() {
  const data = window.RL_DATA;
  if (!data || !document.getElementById("accuracy")) return;
  const meta = data.benchmark.meta;
  const diff = data.benchmark.classifiers.diff_norm;
  const moving = meta.n_moving;
  const still = meta.n_frozen;
  const caught = Math.round((moving * diff.moving_flagged) / 100);
  const falseCalls = Math.round((still * diff.frozen_flagged) / 100);
  const n = (value) => value.toLocaleString("en-US");

  const set = (id, text) => {
    const node = document.getElementById(id);
    if (node) node.textContent = text;
  };
  set("caughtCount", n(caught));
  set("movingCount", n(moving));
  set("falseCount", n(falseCalls));
  set("stillCount", n(still));
  set("fixtureCount", String(data.fixtures.cases.length));

  const movingMount = document.getElementById("dotsMoving");
  const stillMount = document.getElementById("dotsStill");
  if (movingMount) {
    movingMount.setAttribute(
      "aria-label",
      n(moving) + " squares, one for each moment someone moved: " + n(caught) +
        " filled for the moments the referee caught, " + n(moving - caught) + " empty for the misses."
    );
    movingMount.insertAdjacentHTML(
      "afterend",
      '<p class="dots-key"><span><i style="background:var(--red-lit)"></i>caught, ' + n(caught) +
        '</span><span><i style="background:var(--ink)"></i>missed, ' +
        n(moving - caught) + "</span></p>"
    );
  }
  if (stillMount) {
    stillMount.setAttribute(
      "aria-label",
      n(still) + " squares, one for each moment a player stood still: " +
        (falseCalls === 0 ? "none of them" : n(falseCalls)) + " called out."
    );
    stillMount.insertAdjacentHTML(
      "afterend",
      '<p class="dots-key"><span><i style="background:var(--s1)"></i>left alone, ' +
        n(still - falseCalls) + '</span><span><i style="background:var(--red-lit)"></i>called out, ' +
        n(falseCalls) + "</span></p>"
    );
  }

  const draw = () => {
    drawDots(movingMount, moving, [
      { count: caught, colour: "--red-lit" },
      { count: moving - caught, colour: "--ink" },
    ]);
    drawDots(stillMount, still, [
      { count: still - falseCalls, colour: "--s1" },
      { count: falseCalls, colour: "--red-lit" },
    ]);
  };
  draw();
  window.addEventListener("rl-theme-change", draw);
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  if (media.addEventListener) media.addEventListener("change", draw);
  if ("ResizeObserver" in window) {
    let width = movingMount ? movingMount.clientWidth : 0;
    new ResizeObserver(() => {
      const next = movingMount ? movingMount.clientWidth : 0;
      if (next !== width) {
        width = next;
        draw();
      }
    }).observe(document.getElementById("accuracy"));
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initViews();
  initStepArt();
  initAccuracy();
});
