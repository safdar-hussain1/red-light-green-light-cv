/**
 * The referee, drawn.
 *
 * Original art, built as an inline SVG so it inherits the page's ink colour
 * and stays crisp at any size. The figure is deliberately made of very few
 * shapes: a visitor has to read her state from across a room, in peripheral
 * vision, while concentrating on holding still. Two states carry the whole
 * game — back turned while the light is green, face turned while it is red —
 * and a third, subtler one marks the moment the red light is actually armed:
 * two ranging rings settle over her eyes. That is the page's one piece of
 * anthropomorphism, and it is honest, because the rings appear exactly when
 * the judge starts counting violations and not a moment earlier.
 *
 * Fills are fixed paint: her skin, her hair, and the ochre of her dress are
 * the same in either theme, the way a painted figure is the same colour in
 * daylight and under floodlights. Outlines take `currentColor`, so the whole
 * drawing is drawn in the page's ink and stays legible on both backgrounds.
 * The one exception is the ranging rings, which take the live red so they
 * match the light that switched them on.
 *
 * The head turn is a CSS transform on one group, so `prefers-reduced-motion`
 * flattens it to an instant cut without any of this code knowing.
 */

/** Paint that does not change with the theme. */
const DOLL_SKIN = "#f2d6b3";
const DOLL_HAIR = "#241d19";
const DOLL_DRESS = "#d98324";
const DOLL_TRIM = "#f7f2e6";
const DOLL_CHEEK = "#e08d84";

/**
 * The whole figure, as one SVG string.
 *
 * Both heads are always in the document; only their opacity changes, so a
 * turn never reflows or re-rasterises anything. The silhouette pieces her
 * hair bunches sit outside both, so her outline stays constant through the
 * turn and only the face appears.
 */
function dollMarkup() {
  return [
    '<svg class="doll" viewBox="0 0 200 260" role="img"',
    ' aria-labelledby="dolltitle" data-facing="away" data-armed="false"',
    ' fill="none" stroke="currentColor" stroke-width="3"',
    ' stroke-linejoin="round" stroke-linecap="round">',
    '<title id="dolltitle">The referee: back turned on green, watching on red</title>',

    // Ground shadow. No outline — it is light, not an object.
    '<ellipse cx="100" cy="247" rx="54" ry="7" fill="currentColor"',
    ' opacity="0.12" stroke="none"/>',

    '<g class="doll-body">',

    // Legs and shoes.
    '<rect x="85" y="196" width="12" height="44" rx="5" fill="' + DOLL_HAIR + '"/>',
    '<rect x="103" y="196" width="12" height="44" rx="5" fill="' + DOLL_HAIR + '"/>',
    '<path d="M79 240h20a4 4 0 0 1 4 4v2H79z" fill="' + DOLL_HAIR + '"/>',
    '<path d="M121 240h-20a4 4 0 0 0-4 4v2h24z" fill="' + DOLL_HAIR + '"/>',

    // Arms, hanging still. Stroked, not filled — she is drawn, not modelled.
    '<path d="M76 132 60 186" stroke-width="9"/>',
    '<path d="M124 132 140 186" stroke-width="9"/>',
    '<circle cx="59" cy="191" r="7" fill="' + DOLL_SKIN + '"/>',
    '<circle cx="141" cy="191" r="7" fill="' + DOLL_SKIN + '"/>',

    // Pinafore dress.
    '<path d="M75 122h50l21 84q-46 11-92 0z" fill="' + DOLL_DRESS + '"/>',
    '<path d="M56 199q44 10 88 0l2 7q-46 11-92 0z" fill="' + DOLL_TRIM + '" stroke="none"/>',
    '<path d="M100 126v78" stroke="' + DOLL_TRIM + '" stroke-width="2" opacity="0.5"/>',

    // Collar, and the neck the head turns on.
    '<rect x="92" y="106" width="16" height="20" rx="6" fill="' + DOLL_SKIN + '"/>',
    '<path d="M84 118h32" stroke-width="3"/>',

    // Hair bunches: part of the silhouette in both states.
    '<circle cx="58" cy="88" r="17" fill="' + DOLL_HAIR + '"/>',
    '<circle cx="142" cy="88" r="17" fill="' + DOLL_HAIR + '"/>',

    '<g class="doll-head">',

    // Back of the head: one solid shape, a crown swirl, and a ribbon low on
    // the back. No feature may read as a face from behind — an arc where a
    // mouth would sit is enough to make the turn ambiguous.
    '<g class="back-head">',
    '<circle cx="100" cy="74" r="38" fill="' + DOLL_HAIR + '"/>',
    '<path d="M92 52q14-6 22 6" stroke="' + DOLL_TRIM + '" stroke-width="2" opacity="0.35"/>',
    '<path d="M84 100h32" stroke="' + DOLL_TRIM + '" stroke-width="3" opacity="0.55"/>',
    '<path d="M100 100 88 92v16zM100 100l12-8v16z" fill="' + DOLL_TRIM + '" stroke="none"/>',
    '<circle cx="100" cy="100" r="4" fill="' + DOLL_TRIM + '" stroke="none"/>',
    "</g>",

    // The face.
    '<g class="face">',
    '<circle cx="100" cy="74" r="38" fill="' + DOLL_SKIN + '"/>',
    '<path d="M62 74a38 38 0 0 1 76 0q-9-13-19-7-8-11-19-4-10-7-19 4-9-6-19 7z"',
    ' fill="' + DOLL_HAIR + '"/>',
    '<ellipse cx="76" cy="86" rx="8" ry="5" fill="' + DOLL_CHEEK + '" opacity="0.5" stroke="none"/>',
    '<ellipse cx="124" cy="86" rx="8" ry="5" fill="' + DOLL_CHEEK + '" opacity="0.5" stroke="none"/>',
    '<circle cx="86" cy="78" r="6" fill="' + DOLL_HAIR + '" stroke="none"/>',
    '<circle cx="114" cy="78" r="6" fill="' + DOLL_HAIR + '" stroke="none"/>',
    '<circle cx="88" cy="76" r="2" fill="' + DOLL_TRIM + '" stroke="none"/>',
    '<circle cx="116" cy="76" r="2" fill="' + DOLL_TRIM + '" stroke="none"/>',
    '<path d="M94 95q6 5 12 0" stroke-width="2.5"/>',

    // Armed: the ranging rings. On only while a red light is counting.
    '<g class="scan-ring" stroke="var(--red-lit)" stroke-width="2.5">',
    '<circle cx="86" cy="78" r="12"/>',
    '<circle cx="114" cy="78" r="12"/>',
    '<path d="M86 62v-6M86 94v6M70 78h-6M102 78h6"/>',
    '<path d="M114 62v-6M114 94v6M98 78h-6M130 78h6"/>',
    "</g>",
    "</g>",

    "</g>",
    "</g>",
    "</svg>",
  ].join("");
}

/**
 * Mount a doll into an element and hand back the two controls it needs.
 *
 * @param {Element} mount Container; its contents are replaced.
 * @returns {{setPhase: function(string): void, setArmed: function(boolean): void, el: Element}}
 */
function createDoll(mount) {
  mount.innerHTML = dollMarkup();
  const svg = mount.querySelector("svg");

  /**
   * Green is the only phase she turns away for. Everything else — the
   * countdown, a red light, the end of the match — she spends looking
   * straight at the player, which is also what makes the turn at every green
   * feel like a reprieve.
   */
  function setPhase(phase) {
    svg.dataset.facing = phase === "GREEN" ? "away" : "player";
    if (phase !== "RED") svg.dataset.armed = "false";
  }

  function setArmed(armed) {
    svg.dataset.armed = armed ? "true" : "false";
  }

  return { setPhase, setArmed, el: svg };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { createDoll, dollMarkup };
}
