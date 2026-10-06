/**
 * The referee, drawn — and performed.
 *
 * Original art, built as an inline SVG so it inherits the page's ink colour
 * and stays crisp at any size. The figure is deliberately made of very few
 * shapes: a visitor has to read her state from across a room, in peripheral
 * vision, while concentrating on holding still. Two states carry the whole
 * game — back turned while the light is green, face turned while it is red —
 * and a third, subtler one marks the moment the red light is actually armed:
 * her eyes burn red. That is the page's one piece of anthropomorphism, and it
 * is honest, because the glow appears exactly when the judge starts counting
 * violations and not a moment earlier.
 *
 * **The turn is the performance.** She does not cut between two pictures.
 * The head swings through a profile — a real third drawing, not a crossfade
 * — and the body follows a beat later, because a head leads a turn and a
 * body catches up. Under a red light she then sweeps the arena slowly, head
 * and eyes together, the way somebody actually looks for movement. Those
 * three moments (turn, sweep, and the sharp snap when a player is called
 * out) are the only motion she is allowed; everything else about her holds
 * still, which is what makes them land.
 *
 * Fills are fixed paint: her skin, her hair, her yellow shirt and the orange
 * of her pinafore are the same in either theme, the way a painted figure is
 * the same colour in daylight and under floodlights. Outlines take
 * `currentColor`, and she stands in front of the lamp in the referee panel,
 * so the lit lens is what her silhouette is read against. The one exception
 * is the glow in her eyes, which takes the live red so it matches the light
 * that switched it on.
 *
 * Every movement is a CSS transform driven by an attribute on the root
 * `<svg>`, so `prefers-reduced-motion` flattens the whole performance to
 * instant cuts without any of this code knowing — except for the one thing
 * CSS cannot decide, the intermediate profile frame, which this file skips
 * outright when the visitor has asked for less motion.
 */

/** Paint that does not change with the theme. */
const DOLL_SKIN = "#f4d4b0";
const DOLL_HAIR = "#1f1a17";
const DOLL_DRESS = "#f08a24";
const DOLL_SHIRT = "#ffc531";
const DOLL_TRIM = "#fbf6ec";
const DOLL_CHEEK = "#ec8f86";

/** How long the head holds the profile drawing mid-turn, in ms. */
const PROFILE_MS = 150;

/** How long the body keeps following after the head has arrived, in ms. */
const SETTLE_MS = 380;

/** How long the elimination head-snap runs, in ms. Matches `doll-snap`. */
const SNAP_MS = 520;

/** How long the victory bow runs, in ms. Matches `doll-bow`. */
const BOW_MS = 1500;

/**
 * The whole figure, as one SVG string.
 *
 * All three heads are always in the document; only their opacity changes, so
 * a turn never reflows or re-rasterises anything. The silhouette pieces —
 * her hair buns — sit outside all three, so her outline stays constant
 * through the turn and only the face appears.
 *
 * Two nested groups carry the motion. `.doll-head` is the turn, driven by a
 * transition between three fixed angles. `.doll-gaze` inside it is the
 * sweep, driven by a looping animation. Keeping them apart is what lets her
 * scan the arena and turn away from it without the two transforms fighting
 * over one element.
 */
function dollMarkup() {
  return [
    '<svg class="doll" viewBox="0 0 200 260" role="img"',
    ' aria-labelledby="dolltitle" data-facing="away" data-armed="false"',
    ' fill="none" stroke="currentColor" stroke-width="3"',
    ' stroke-linejoin="round" stroke-linecap="round">',
    '<title id="dolltitle">The referee: back turned on green, watching on red</title>',
    '<defs><filter id="dollGlow" x="-80%" y="-80%" width="260%" height="260%">',
    '<feGaussianBlur stdDeviation="3.2"/></filter></defs>',

    // Ground shadow. No outline — it is light, not an object.
    '<ellipse cx="100" cy="250" rx="50" ry="6" fill="currentColor"',
    ' opacity="0.28" stroke="none"/>',

    '<g class="doll-body">',

    // Knee socks and shoes.
    '<rect x="84" y="196" width="13" height="44" rx="5" fill="' + DOLL_TRIM + '"/>',
    '<rect x="103" y="196" width="13" height="44" rx="5" fill="' + DOLL_TRIM + '"/>',
    '<path d="M78 238h21a4 4 0 0 1 4 4v4H78z" fill="' + DOLL_HAIR + '"/>',
    '<path d="M122 238h-21a4 4 0 0 0-4 4v4h25z" fill="' + DOLL_HAIR + '"/>',

    // Arms, hanging still, under short puffed sleeves.
    '<path d="M74 136 62 186" stroke-width="10.5"/>',
    '<path d="M126 136 138 186" stroke-width="10.5"/>',
    '<path d="M74 136 62 186" stroke="' + DOLL_SKIN + '" stroke-width="5.5"/>',
    '<path d="M126 136 138 186" stroke="' + DOLL_SKIN + '" stroke-width="5.5"/>',
    '<circle cx="61" cy="190" r="6.5" fill="' + DOLL_SKIN + '"/>',
    '<circle cx="139" cy="190" r="6.5" fill="' + DOLL_SKIN + '"/>',
    '<ellipse cx="74" cy="131" rx="11" ry="9" fill="' + DOLL_SHIRT + '"/>',
    '<ellipse cx="126" cy="131" rx="11" ry="9" fill="' + DOLL_SHIRT + '"/>',

    // Yellow shirt, and the orange pinafore over it.
    '<path d="M80 120h40l4 18H76z" fill="' + DOLL_SHIRT + '"/>',
    '<path d="M76 134h48l22 72q-46 12-92 0z" fill="' + DOLL_DRESS + '"/>',
    '<path d="M84 134V122M116 134V122" stroke="' + DOLL_DRESS + '" stroke-width="5"/>',
    '<path d="M56 198q44 11 88 0l2 8q-46 12-92 0z" fill="' + DOLL_TRIM + '" stroke="none"/>',
    '<path d="M100 140v62" stroke="#c96a12" stroke-width="2" opacity="0.6"/>',

    // Neck and a round white collar.
    '<rect x="93" y="104" width="14" height="18" rx="5" fill="' + DOLL_SKIN + '"/>',
    '<path d="M100 118q-12-2-17 6 9 5 17-1 8 6 17 1-5-8-17-6z" fill="' + DOLL_TRIM + '" stroke-width="2.5"/>',

    // Hair buns, tied with yellow: part of the silhouette in every state.
    '<circle cx="56" cy="80" r="17" fill="' + DOLL_HAIR + '"/>',
    '<circle cx="144" cy="80" r="17" fill="' + DOLL_HAIR + '"/>',
    '<rect x="66" y="80" width="9" height="12" rx="3" fill="' + DOLL_SHIRT + '" stroke-width="2"/>',
    '<rect x="125" y="80" width="9" height="12" rx="3" fill="' + DOLL_SHIRT + '" stroke-width="2"/>',

    '<g class="doll-head">',
    '<g class="doll-gaze">',

    // Back of the head: one solid shape and a crown swirl. No feature may
    // read as a face from behind — an arc where a mouth would sit is enough
    // to make the turn ambiguous.
    '<g class="back-head">',
    '<circle cx="100" cy="72" r="40" fill="' + DOLL_HAIR + '"/>',
    '<path d="M90 48q15-7 24 6" stroke="' + DOLL_TRIM + '" stroke-width="2" opacity="0.3"/>',
    '<path d="M74 96q26 12 52 0" stroke="' + DOLL_TRIM + '" stroke-width="2" opacity="0.18"/>',
    "</g>",

    // The profile: the head caught halfway round. One eye, the edge of a
    // nose, and most of the hair still on the far side. It is on screen for
    // a sixth of a second and its whole job is to say the head travelled
    // rather than dissolved.
    '<g class="profile-head">',
    '<circle cx="100" cy="72" r="40" fill="' + DOLL_SKIN + '"/>',
    '<path d="M100 32a40 40 0 0 0 0 80q-17-14-17-40t17-40z" fill="' + DOLL_HAIR + '"/>',
    '<path d="M60 72a40 40 0 0 1 80 0V62H96q-10 6-36 10z" fill="' + DOLL_HAIR + '"/>',
    '<path d="M138 75q6 5 0 9" stroke-width="2.5"/>',
    '<ellipse cx="114" cy="90" rx="7" ry="4.5" fill="' + DOLL_CHEEK + '" opacity="0.55" stroke="none"/>',
    '<ellipse class="eye" cx="122" cy="79" rx="5" ry="6.5" fill="' + DOLL_HAIR + '" stroke="none"/>',
    '<circle cx="123.5" cy="76.5" r="1.8" fill="' + DOLL_TRIM + '" stroke="none"/>',
    '<path d="M117 97q7 3 11-2" stroke-width="2.5"/>',
    "</g>",

    // The face: a blunt fringe, round eyes, and nothing else she needs.
    '<g class="face">',
    '<circle cx="100" cy="72" r="40" fill="' + DOLL_SKIN + '"/>',
    '<path d="M60 94V70a40 40 0 0 1 80 0v24q-3 3-7-1V62H67v31q-4 4-7 1z" fill="' + DOLL_HAIR + '"/>',
    '<ellipse cx="78" cy="92" rx="8" ry="5" fill="' + DOLL_CHEEK + '" opacity="0.55" stroke="none"/>',
    '<ellipse cx="122" cy="92" rx="8" ry="5" fill="' + DOLL_CHEEK + '" opacity="0.55" stroke="none"/>',

    // The eyes are their own group so they can lead the sweep: on a real
    // scan the eyes reach the edge of the room before the head does. While
    // a red light is armed they burn red, and only then.
    '<g class="eyes">',
    '<g class="eye-glow" filter="url(#dollGlow)">',
    '<circle cx="86" cy="80" r="9" stroke="none"/>',
    '<circle cx="114" cy="80" r="9" stroke="none"/>',
    "</g>",
    '<ellipse class="eye" cx="86" cy="80" rx="6.5" ry="8" fill="' + DOLL_HAIR + '" stroke="none"/>',
    '<ellipse class="eye" cx="114" cy="80" rx="6.5" ry="8" fill="' + DOLL_HAIR + '" stroke="none"/>',
    '<circle cx="88.5" cy="76.5" r="2.2" fill="' + DOLL_TRIM + '" stroke="none"/>',
    '<circle cx="116.5" cy="76.5" r="2.2" fill="' + DOLL_TRIM + '" stroke="none"/>',
    "</g>",
    '<path d="M94 99q6 5 12 0" stroke-width="2.5"/>',
    "</g>",

    "</g>",
    "</g>",
    "</g>",
    "</svg>",
  ].join("");
}

/** True when the visitor has asked the platform for less movement. */
function wantsLessMotion() {
  return (
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

/**
 * Mount a doll into an element and hand back the controls she performs with.
 *
 * @param {Element} mount Container; its contents are replaced.
 * @returns {{setPhase: function(string): void, setArmed: function(boolean): void,
 *   snap: function(): void, bow: function(): void, el: Element}}
 */
function createDoll(mount) {
  mount.innerHTML = dollMarkup();
  const svg = mount.querySelector("svg");
  let turnTimers = [];
  let poseTimer = null;

  function clearTurn() {
    for (const timer of turnTimers) window.clearTimeout(timer);
    turnTimers = [];
  }

  /**
   * Swing the head to a new facing, through the profile.
   *
   * `data-turn` is what the stylesheet watches: while it is present the
   * profile drawing is up and the body is running its follow-through, and
   * the looping arena sweep is suspended so two transforms never claim the
   * same group. Reduced motion skips the middle entirely — there is nothing
   * to see in a 150 ms drawing that flashes past without moving.
   */
  function face(next) {
    if (svg.dataset.facing === next) return;
    clearTurn();
    if (wantsLessMotion()) {
      svg.dataset.facing = next;
      delete svg.dataset.turn;
      return;
    }
    svg.dataset.turn = "profile";
    turnTimers.push(
      window.setTimeout(() => {
        svg.dataset.facing = next;
        svg.dataset.turn = "settle";
      }, PROFILE_MS)
    );
    turnTimers.push(
      window.setTimeout(() => {
        delete svg.dataset.turn;
      }, PROFILE_MS + SETTLE_MS)
    );
  }

  /**
   * Green is the phase she turns away for, and so is a wipeout — there is
   * nobody left in the arena to watch. Everything else — the countdown, a
   * red light, a victory — she spends looking straight at the player, which
   * is also what makes the turn at every green feel like a reprieve.
   *
   * The slow sweep is pinned to the red light rather than to her facing, so
   * she watches the room only while it costs a player something.
   */
  function setPhase(phase) {
    face(phase === "GREEN" || phase === "WIPEOUT" ? "away" : "player");
    if (phase === "RED") svg.dataset.scan = "true";
    else delete svg.dataset.scan;
    if (phase !== "RED") svg.dataset.armed = "false";
    if (phase !== "VICTORY") {
      delete svg.dataset.pose;
      if (poseTimer !== null) {
        window.clearTimeout(poseTimer);
        poseTimer = null;
      }
    }
  }

  function setArmed(armed) {
    svg.dataset.armed = armed ? "true" : "false";
  }

  /** One hard look at whoever just moved. */
  function snap() {
    if (wantsLessMotion()) return;
    delete svg.dataset.snap;
    // Reflow between the two writes, or a second call within the animation
    // does nothing — which is exactly the case that matters, two players
    // going out a few frames apart.
    void svg.getBoundingClientRect().width;
    svg.dataset.snap = "true";
    window.setTimeout(() => {
      delete svg.dataset.snap;
    }, SNAP_MS);
  }

  /** Somebody outlasted her. She acknowledges it, once. */
  function bow() {
    if (wantsLessMotion()) return;
    if (poseTimer !== null) window.clearTimeout(poseTimer);
    svg.dataset.pose = "bow";
    poseTimer = window.setTimeout(() => {
      delete svg.dataset.pose;
      poseTimer = null;
    }, BOW_MS);
  }

  return { setPhase, setArmed, snap, bow, el: svg };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { createDoll, dollMarkup };
}
