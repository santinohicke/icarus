// Sound effects, synthesized with the Web Audio API (no audio files).
// Deliberately quiet: everything goes through one low master gain.

const MASTER_VOLUME = 0.12;
const MUTE_KEY = "icarus.muted";

/** Two seconds of white noise, looped by the continuous sounds. */
function noiseBuffer(ctx: AudioContext, seconds = 2): AudioBuffer {
  const buffer = ctx.createBuffer(1, Math.floor(ctx.sampleRate * seconds), ctx.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i += 1) data[i] = Math.random() * 2 - 1;
  return buffer;
}

interface Loop {
  gain: GainNode;
  filter: BiquadFilterNode;
}

export class Sound {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private noise: AudioBuffer | null = null;
  private engine: Loop | null = null;
  private rcs: Loop | null = null;
  private muted: boolean;
  /** Web Audio failed once: stay silent rather than retry on every gesture. */
  private unavailable = false;

  constructor() {
    let stored: string | null = null;
    try {
      stored = window.localStorage.getItem(MUTE_KEY);
    } catch {
      // storage unavailable: default to sound on
    }
    this.muted = stored === "1";
  }

  get isMuted(): boolean {
    return this.muted;
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
    try {
      window.localStorage.setItem(MUTE_KEY, muted ? "1" : "0");
    } catch {
      // ignore
    }
    if (this.ctx && this.master) {
      this.master.gain.setTargetAtTime(muted ? 0 : MASTER_VOLUME, this.ctx.currentTime, 0.05);
    }
  }

  /** Creates or resumes the audio context. Call from a user gesture (click, key). */
  unlock(): void {
    if (this.unavailable) return;
    if (this.ctx !== null) {
      if (this.ctx.state === "suspended") void this.ctx.resume();
      return;
    }
    let ctx: AudioContext | null = null;
    try {
      ctx = new AudioContext();
      const master = ctx.createGain();
      master.gain.value = this.muted ? 0 : MASTER_VOLUME;
      master.connect(ctx.destination);
      this.ctx = ctx;
      this.master = master;
      this.noise = noiseBuffer(ctx);
      // Engine: low rumble. RCS: high hiss.
      this.engine = this.loop("lowpass", 420, 0.7);
      this.rcs = this.loop("highpass", 2_500, 0.4);
      if (ctx.state === "suspended") void ctx.resume();
    } catch {
      // No Web Audio: play silently, and never retry.
      void ctx?.close().catch(() => undefined);
      this.ctx = null;
      this.master = null;
      this.engine = null;
      this.rcs = null;
      this.unavailable = true;
    }
  }

  private loop(type: BiquadFilterType, frequency: number, q: number): Loop | null {
    const { ctx, master, noise } = this;
    if (!ctx || !master || !noise) return null;
    const src = ctx.createBufferSource();
    src.buffer = noise;
    src.loop = true;
    const filter = ctx.createBiquadFilter();
    filter.type = type;
    filter.frequency.value = frequency;
    filter.Q.value = q;
    const gain = ctx.createGain();
    gain.gain.value = 0;
    src.connect(filter).connect(gain).connect(master);
    src.start();
    return { gain, filter };
  }

  /**
   * Continuous sounds, once per frame. `thrust` is the fraction of max thrust
   * (0 = engine off); `rcs` is whether a thruster fires.
   */
  update(thrust: number, rcs: boolean): void {
    const ctx = this.ctx;
    if (!ctx || ctx.state !== "running" || !this.engine || !this.rcs) return;
    const now = ctx.currentTime;
    const t = Number.isFinite(thrust) ? Math.min(1, Math.max(0, thrust)) : 0;
    this.engine.gain.gain.setTargetAtTime(t > 0.01 ? 0.35 + 0.65 * t : 0, now, 0.08);
    this.engine.filter.frequency.setTargetAtTime(250 + 450 * t, now, 0.1);
    this.rcs.gain.gain.setTargetAtTime(rcs ? 0.18 : 0, now, 0.03);
  }

  /** Stops the continuous sounds (pause, menu, end of run). */
  silence(): void {
    this.update(0, false);
  }

  explode(): void {
    const { ctx, master, noise } = this;
    if (!ctx || ctx.state !== "running" || !master || !noise) return;
    this.silence();
    const now = ctx.currentTime;
    // Noise burst with a closing low-pass...
    const src = ctx.createBufferSource();
    src.buffer = noise;
    const filter = ctx.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.setValueAtTime(1_800, now);
    filter.frequency.exponentialRampToValueAtTime(80, now + 1.6);
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.9, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 1.8);
    src.connect(filter).connect(gain).connect(master);
    src.start(now);
    src.stop(now + 1.9);
    // ...over a falling low thump.
    const osc = ctx.createOscillator();
    osc.frequency.setValueAtTime(90, now);
    osc.frequency.exponentialRampToValueAtTime(30, now + 0.6);
    const og = ctx.createGain();
    og.gain.setValueAtTime(0.8, now);
    og.gain.exponentialRampToValueAtTime(0.001, now + 0.7);
    osc.connect(og).connect(master);
    osc.start(now);
    osc.stop(now + 0.8);
  }

  land(): void {
    const { ctx, master } = this;
    if (!ctx || ctx.state !== "running" || !master) return;
    this.silence();
    const now = ctx.currentTime;
    // Soft touchdown thud, then a two-note confirmation chime.
    const thud = ctx.createOscillator();
    thud.frequency.setValueAtTime(70, now);
    thud.frequency.exponentialRampToValueAtTime(40, now + 0.25);
    const tg = ctx.createGain();
    tg.gain.setValueAtTime(0.5, now);
    tg.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
    thud.connect(tg).connect(master);
    thud.start(now);
    thud.stop(now + 0.35);
    [660, 990].forEach((freq, i) => {
      const start = now + 0.35 + i * 0.16;
      const osc = ctx.createOscillator();
      osc.type = "sine";
      osc.frequency.value = freq;
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, start);
      g.gain.linearRampToValueAtTime(0.25, start + 0.02);
      g.gain.exponentialRampToValueAtTime(0.001, start + 0.6);
      osc.connect(g).connect(master);
      osc.start(start);
      osc.stop(start + 0.65);
    });
  }
}
