/**
 * The chant, the buzzer, and the two stings that end a match.
 *
 * The referee's chant is not a recording. It is the same note table the
 * Python engine synthesises its wav from — `RL_DATA.chant`, a tempo and a
 * list of `[midi, beats]` pairs — played here through WebAudio, so the
 * browser and the desktop engine sing the identical tune without the page
 * carrying an audio file.
 *
 * The chant is also a game mechanic rather than a soundtrack, which is why
 * it is cut dead the instant the light turns rather than faded: in the
 * playground game the chant *is* the clock, and its ending is the warning.
 * Everything here is scheduled on the audio clock (`ctx.currentTime`), never
 * on `setTimeout`, so a busy main thread cannot make the tune stutter — a
 * timer that fires late would move the beat a player is timing their last
 * step against.
 *
 * Sound needs a gesture to start, and the arena only ever builds an
 * AudioContext inside the click that starts a match.
 */

/** Concert pitch for MIDI 69, in hertz. */
const A4_HZ = 440.0;

/** MIDI note number of that same A. */
const A4_MIDI = 69;

/** How far ahead of the audio clock notes are queued, in seconds. */
const SCHEDULE_AHEAD_S = 0.25;

function midiToHz(midi) {
  return A4_HZ * Math.pow(2, (midi - A4_MIDI) / 12);
}

/**
 * Plays the chant on a loop, and the short sounds a match punctuates it with.
 *
 * One instance owns one AudioContext. It is built on the first `resume()`,
 * which the arena calls from inside a click, and everything after that is
 * scheduled against that context's clock.
 */
class ChantPlayer {
  /**
   * @param {{bpm: number, notes: Array<Array<number>>}} chant The baked note
   *   table: a tempo, and `[midi, beats]` pairs.
   */
  constructor(chant) {
    this.chant = chant;
    this.muted = false;
    this.ctx = null;
    this._master = null;
    this._playing = false;
    // Bumped by every stop. A pass that was already queued checks this before
    // queueing the next one, so a stop during a pass cannot be undone by a
    // timer that was already in flight.
    this._generation = 0;
    this._timer = null;
  }

  /** Build or wake the AudioContext. Must be called from a user gesture. */
  resume() {
    if (!this.ctx) {
      const Ctor = window.AudioContext || window.webkitAudioContext;
      if (!Ctor) return false;
      this.ctx = new Ctor();
      this._master = this.ctx.createGain();
      this._master.gain.value = this.muted ? 0 : 0.5;
      this._master.connect(this.ctx.destination);
    }
    if (this.ctx.state === "suspended") this.ctx.resume();
    return true;
  }

  setMuted(muted) {
    this.muted = muted;
    if (this._master) {
      this._master.gain.setTargetAtTime(muted ? 0 : 0.5, this.ctx.currentTime, 0.01);
    }
  }

  /**
   * One sung note: three voices under a soft envelope.
   *
   * The same voicing `redlight.synth` renders the wav with, in the same
   * proportions — a triangle fundamental, a second voice eight cents flat so
   * the pair beats about once a second, and a quiet octave above for a
   * music-box edge. A single oscillator is a test tone; this is a child
   * singing slightly out of tune with herself, which is the sound the game
   * is remembered for.
   */
  _note(midi, at, durationS, gain = 0.22) {
    const ctx = this.ctx;
    const hz = midiToHz(midi);
    const env = ctx.createGain();
    env.connect(this._master);

    // Soft in, long out, to leave air between the syllables.
    const attack = Math.min(0.045, durationS * 0.22);
    env.gain.setValueAtTime(0.0001, at);
    env.gain.exponentialRampToValueAtTime(gain, at + attack);
    env.gain.exponentialRampToValueAtTime(gain * 0.55, at + attack + durationS * 0.24);
    env.gain.exponentialRampToValueAtTime(0.0001, at + Math.max(durationS * 0.98, 0.06));

    const voices = [
      { type: "triangle", hz: hz, level: 1 },
      { type: "triangle", hz: hz, level: 0.55, detune: -8 },
      { type: "sine", hz: hz * 2, level: 0.16 },
    ];
    for (const voice of voices) {
      const osc = ctx.createOscillator();
      osc.type = voice.type;
      osc.frequency.value = voice.hz;
      if (voice.detune) osc.detune.value = voice.detune;
      if (voice.level === 1) {
        osc.connect(env);
      } else {
        const level = ctx.createGain();
        level.gain.value = voice.level;
        osc.connect(level);
        level.connect(env);
      }
      osc.start(at);
      osc.stop(at + durationS + 0.08);
    }
  }

  /** Queue one pass of the chant, and a timer to queue the next one. */
  _schedulePass(startAt) {
    const generation = this._generation;
    const beatS = 60 / this.chant.bpm;
    let cursor = startAt;

    for (const [midi, beats] of this.chant.notes) {
      const durationS = beats * beatS;
      this._note(midi, cursor, durationS);
      cursor += durationS;
    }

    const untilNext = (cursor - this.ctx.currentTime - SCHEDULE_AHEAD_S) * 1000;
    this._timer = window.setTimeout(() => {
      if (!this._playing || generation !== this._generation) return;
      this._schedulePass(cursor);
    }, Math.max(untilNext, 20));
  }

  /** Start the loop. Harmless to call while it is already running. */
  start() {
    if (!this.resume() || this._playing) return;
    this._playing = true;
    this._schedulePass(this.ctx.currentTime + 0.05);
  }

  /**
   * Cut the chant.
   *
   * The master gain is not touched — the buzzer and the end stings share it
   * and have to be audible immediately after a light turns red. Instead the
   * loop simply stops queueing, and any notes already queued run out within
   * the lookahead.
   */
  stop() {
    this._playing = false;
    this._generation += 1;
    if (this._timer !== null) {
      window.clearTimeout(this._timer);
      this._timer = null;
    }
  }

  /** The sound of being called out: a short, flat, falling honk. */
  buzz() {
    if (!this.resume()) return;
    const ctx = this.ctx;
    const at = ctx.currentTime;

    const env = ctx.createGain();
    env.gain.setValueAtTime(0.0001, at);
    env.gain.exponentialRampToValueAtTime(0.42, at + 0.01);
    env.gain.exponentialRampToValueAtTime(0.0001, at + 0.5);

    const tone = ctx.createBiquadFilter();
    tone.type = "lowpass";
    tone.frequency.value = 1100;

    env.connect(this._master);
    tone.connect(env);

    for (const detune of [0, 7]) {
      const osc = ctx.createOscillator();
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(150, at);
      osc.frequency.exponentialRampToValueAtTime(96, at + 0.45);
      osc.detune.value = detune;
      osc.connect(tone);
      osc.start(at);
      osc.stop(at + 0.55);
    }
  }

  /**
   * The meter hitting the floor: a body-weight thump, not a note.
   *
   * Pitched far below anything in the chant and gone in a third of a second,
   * so it lands as a physical event rather than as music. It is timed with
   * the eliminated player's meter falling to zero, which is the one moment
   * on the page where a number drops rather than decays.
   */
  thud() {
    if (!this.resume()) return;
    const ctx = this.ctx;
    const at = ctx.currentTime;

    const env = ctx.createGain();
    env.gain.setValueAtTime(0.0001, at);
    env.gain.exponentialRampToValueAtTime(0.5, at + 0.008);
    env.gain.exponentialRampToValueAtTime(0.0001, at + 0.34);
    env.connect(this._master);

    // The weight: a low sine dropping most of an octave as it lands.
    const body = ctx.createOscillator();
    body.type = "sine";
    body.frequency.setValueAtTime(96, at);
    body.frequency.exponentialRampToValueAtTime(38, at + 0.22);
    body.connect(env);
    body.start(at);
    body.stop(at + 0.4);

    // The impact: one very short filtered noise burst, so the thump has an
    // edge to it and does not read as another bass note.
    const frames = Math.floor(ctx.sampleRate * 0.06);
    const buffer = ctx.createBuffer(1, frames, ctx.sampleRate);
    const channel = buffer.getChannelData(0);
    let seed = 20240814;
    for (let i = 0; i < frames; i += 1) {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      channel[i] = (seed / 4294967296) * 2 - 1;
    }
    const noise = ctx.createBufferSource();
    noise.buffer = buffer;
    const muffle = ctx.createBiquadFilter();
    muffle.type = "lowpass";
    muffle.frequency.value = 420;
    const noiseGain = ctx.createGain();
    noiseGain.gain.setValueAtTime(0.35, at);
    noiseGain.gain.exponentialRampToValueAtTime(0.0001, at + 0.09);
    noise.connect(muffle);
    muffle.connect(noiseGain);
    noiseGain.connect(this._master);
    noise.start(at);
  }

  /** How a match ends: up for a win, down for a wipeout. */
  sting(won) {
    if (!this.resume()) return;
    const at = this.ctx.currentTime + 0.02;
    const notes = won ? [67, 72, 76, 79] : [67, 63, 60, 55];
    notes.forEach((midi, index) => {
      this._note(midi, at + index * 0.13, won ? 0.42 : 0.34, 0.26);
    });
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { ChantPlayer, midiToHz };
}
