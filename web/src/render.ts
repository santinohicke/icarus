// Canvas renderer: camera that keeps the stage and the pad in view, the
// stage itself (body, legs, grid fins, gimballed flame), the pad, an altitude
// grid and wind streaks. Purely visual: nothing here feeds the simulation.

import type { Telemetry, VehicleInfo } from "./sim";

interface Camera {
  /** World x at the screen center (m). */
  cx: number;
  /** World y at the bottom edge of the screen (m). */
  bottom: number;
  /** Pixels per meter. */
  scale: number;
}

interface Pose {
  x: number;
  y: number;
  theta: number;
  gimbal: number;
  thrust: number;
  cm: number;
  vx: number;
  vy: number;
  /** RCS torque, N·m (positive = counter-clockwise). */
  rcs: number;
}

const COLORS = {
  skyTop: "#070d1c",
  skyBottom: "#13213d",
  grid: "rgba(226, 232, 244, 0.07)",
  gridLabel: "rgba(226, 232, 244, 0.5)",
  sea: "#06101f",
  seaLine: "rgba(90, 140, 210, 0.35)",
  deck: "#2a3446",
  deckEdge: "#f2b35c",
  deckMark: "rgba(242, 179, 92, 0.55)",
  body: "#e7ecf4",
  bodyShade: "#b9c3d3",
  band: "#1b2230",
  leg: "#9aa6b8",
  fin: "#59647a",
  velocity: "rgba(242, 240, 234, 0.85)",
  streak: "rgba(170, 200, 255, 0.35)",
} as const;

/** Smallest on-screen length of the stage before it is drawn enlarged. */
const MIN_ROCKET_PX = 34;
const NICE_STEPS = [5, 10, 25, 50, 100, 250, 500, 1000];

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

interface Streak {
  x: number;
  y: number;
  len: number;
}

export class Renderer {
  private readonly ctx: CanvasRenderingContext2D;
  private width = 0;
  private height = 0;
  private cam: Camera | null = null;
  private lastFrame: number | null = null;
  private streaks: Streak[] = [];
  private endTime: number | null = null;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D is not supported");
    this.ctx = ctx;
    this.resize();
    window.addEventListener("resize", () => this.resize());
  }

  /** Where the stage was last drawn, in CSS pixels: centre of mass and half its drawn length. */
  rocketOnScreen: { x: number; y: number; half: number } | null = null;

  /** Forgets the camera and effects (new run). */
  reset(): void {
    this.rocketOnScreen = null;
    this.cam = null;
    this.streaks = [];
    this.endTime = null;
  }

  private resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    // Never zero: a minimized window or collapsed frame must not poison the camera.
    this.width = Math.max(1, window.innerWidth);
    this.height = Math.max(1, window.innerHeight);
    this.canvas.width = Math.round(this.width * dpr);
    this.canvas.height = Math.round(this.height * dpr);
    this.canvas.style.width = `${this.width}px`;
    this.canvas.style.height = `${this.height}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  private sx(cam: Camera, x: number): number {
    return this.width / 2 + (x - cam.cx) * cam.scale;
  }

  private sy(cam: Camera, y: number): number {
    return this.height - (y - cam.bottom) * cam.scale;
  }

  private targetCamera(p: Pose, info: VehicleInfo): Camera {
    const pad = info.pad_half_width;
    const top = Math.max(p.y, 0) + info.length;
    const viewHeight = Math.max(110, top * 1.3);
    const minX = Math.min(p.x - 35, -pad - 15);
    const maxX = Math.max(p.x + 35, pad + 15);
    const viewWidth = (maxX - minX) * 1.15;
    const raw = Math.min(this.height / viewHeight, this.width / viewWidth);
    const scale = Number.isFinite(raw) && raw > 0 ? raw : 1;
    const groundMargin = Math.min(110, this.height * 0.16);
    const cx = Number.isFinite(minX + maxX) ? (minX + maxX) / 2 : 0;
    return { cx, bottom: -groundMargin / scale, scale };
  }

  /**
   * Draws one frame. `frozen` (paused, not started or ended) shows the latest
   * state without interpolation and stops the decorative motion.
   */
  draw(prev: Telemetry, cur: Telemetry, alpha: number, info: VehicleInfo, frozen: boolean): void {
    const now = performance.now();
    const frameDt = this.lastFrame === null ? 1 / 60 : Math.min((now - this.lastFrame) / 1000, 0.1);
    this.lastFrame = now;
    const ended = cur.outcome !== 0;

    const t = frozen ? 1 : Math.min(Math.max(alpha, 0), 1);
    const pose: Pose = {
      x: lerp(prev.x, cur.x, t),
      y: lerp(prev.y, cur.y, t),
      theta: lerp(prev.theta, cur.theta, t),
      gimbal: lerp(prev.gimbal, cur.gimbal, t),
      thrust: lerp(prev.thrust_fraction, cur.thrust_fraction, t),
      cm: cur.cm_height,
      vx: cur.vx,
      vy: cur.vy,
      rcs: cur.rcs_torque,
    };

    const target = this.targetCamera(pose, info);
    const finite = (c: Camera | null): c is Camera =>
      c !== null && Number.isFinite(c.cx) && Number.isFinite(c.bottom) && Number.isFinite(c.scale);
    if (!finite(this.cam)) {
      this.cam = target;
    } else {
      const k = 1 - Math.exp(-frameDt * 3);
      this.cam = {
        cx: lerp(this.cam.cx, target.cx, k),
        bottom: lerp(this.cam.bottom, target.bottom, k),
        scale: lerp(this.cam.scale, target.scale, k),
      };
    }
    const cam = this.cam;

    this.drawSky();
    this.drawGrid(cam);
    this.drawStreaks(cam, cur.wind, frameDt, frozen);
    this.drawSeaAndPad(cam, info);
    this.drawRocket(cam, pose, info, cur.engine_on > 0.5, ended);
    if (!ended) this.drawVelocity(cam, pose);
    this.drawOutcome(cam, cur, pose, now);
  }

  private drawSky(): void {
    const g = this.ctx.createLinearGradient(0, 0, 0, this.height);
    g.addColorStop(0, COLORS.skyTop);
    g.addColorStop(1, COLORS.skyBottom);
    this.ctx.fillStyle = g;
    this.ctx.fillRect(0, 0, this.width, this.height);
  }

  private drawGrid(cam: Camera): void {
    const ctx = this.ctx;
    const visible = this.height / cam.scale;
    const step = NICE_STEPS.find((s) => visible / s <= 8) ?? 1000;
    const top = cam.bottom + visible;
    ctx.strokeStyle = COLORS.grid;
    ctx.fillStyle = COLORS.gridLabel;
    ctx.lineWidth = 1;
    ctx.font = '500 11px "JetBrains Mono", ui-monospace, monospace';
    ctx.textBaseline = "bottom";
    // Labels on the right edge: the HUD panel covers the left one.
    ctx.textAlign = "right";
    for (let y = Math.ceil(Math.max(cam.bottom, 0) / step) * step; y <= top; y += step) {
      if (y === 0) continue;
      const py = Math.round(this.sy(cam, y)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, py);
      ctx.lineTo(this.width, py);
      ctx.stroke();
      ctx.fillText(`${y} m`, this.width - 10, py - 2);
    }
    ctx.textAlign = "left";
    const left = cam.cx - this.width / 2 / cam.scale;
    const right = cam.cx + this.width / 2 / cam.scale;
    for (let x = Math.ceil(left / step) * step; x <= right; x += step) {
      const px = Math.round(this.sx(cam, x)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, this.sy(cam, 0));
      ctx.stroke();
    }
  }

  private drawStreaks(cam: Camera, wind: number, dt: number, frozen: boolean): void {
    const ctx = this.ctx;
    const halfW = this.width / 2 / cam.scale;
    const viewH = this.height / cam.scale;
    const left = cam.cx - halfW;
    const right = cam.cx + halfW;
    const respawn = (s: Streak, anywhere: boolean): void => {
      s.x = anywhere ? left + Math.random() * (right - left) : wind >= 0 ? left : right;
      s.y = Math.max(2, cam.bottom) + Math.random() * viewH;
      s.len = 8 + Math.random() * 20;
    };
    while (this.streaks.length < 36) {
      const s = { x: 0, y: 0, len: 0 };
      respawn(s, true);
      this.streaks.push(s);
    }
    ctx.strokeStyle = COLORS.streak;
    ctx.lineWidth = 1;
    for (const s of this.streaks) {
      if (!frozen) s.x += wind * dt * 3;
      if (s.x < left - 40 || s.x > right + 40 || s.y < cam.bottom || s.y > cam.bottom + viewH) {
        respawn(s, false);
      }
      const px = this.sx(cam, s.x);
      const py = this.sy(cam, s.y);
      const len = Math.min(60, (s.len * Math.abs(wind)) / 4);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px - Math.sign(wind) * len, py);
      ctx.stroke();
    }
  }

  private drawSeaAndPad(cam: Camera, info: VehicleInfo): void {
    const ctx = this.ctx;
    const ground = this.sy(cam, 0);
    ctx.fillStyle = COLORS.sea;
    ctx.fillRect(0, ground, this.width, this.height - ground);
    ctx.strokeStyle = COLORS.seaLine;
    ctx.beginPath();
    ctx.moveTo(0, ground + 0.5);
    ctx.lineTo(this.width, ground + 0.5);
    ctx.stroke();

    const pad = info.pad_half_width;
    const x0 = this.sx(cam, -pad);
    const x1 = this.sx(cam, pad);
    const deck = Math.max(4, 3 * cam.scale);
    ctx.fillStyle = COLORS.deck;
    ctx.fillRect(x0, ground, x1 - x0, deck);
    ctx.fillStyle = COLORS.deckEdge;
    ctx.fillRect(x0, ground, Math.max(2, 1.2 * cam.scale), deck);
    ctx.fillRect(x1 - Math.max(2, 1.2 * cam.scale), ground, Math.max(2, 1.2 * cam.scale), deck);
    // Target marker.
    ctx.strokeStyle = COLORS.deckMark;
    ctx.lineWidth = Math.max(1, 0.4 * cam.scale);
    const c = this.sx(cam, 0);
    const r = Math.max(4, 4 * cam.scale);
    ctx.beginPath();
    ctx.moveTo(c - r, ground + 1);
    ctx.lineTo(c + r, ground + 1);
    ctx.moveTo(c, ground - Math.min(r, 6));
    ctx.lineTo(c, ground + 1);
    ctx.stroke();
  }

  private drawRocket(cam: Camera, p: Pose, info: VehicleInfo, engineOn: boolean, ended: boolean): void {
    const ctx = this.ctx;
    const natural = info.length * cam.scale;
    const enlarge = natural < MIN_ROCKET_PX ? MIN_ROCKET_PX / natural : 1;
    const s = cam.scale * enlarge; // pixels per meter for the vehicle itself
    const px = this.sx(cam, p.x);
    const py = this.sy(cam, p.y);

    if (enlarge > 1) {
      ctx.strokeStyle = "rgba(242, 240, 234, 0.4)";
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.arc(px, py, MIN_ROCKET_PX * 0.75, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const len = info.length * s;
    this.rocketOnScreen = { x: px, y: py, half: len / 2 };
    const width = Math.max(3, info.diameter * s);
    const tail = p.cm * s; // CoM → tail distance, downward on screen
    const nose = len - tail;
    const span = info.leg_half_span * s;

    ctx.save();
    ctx.translate(px, py);
    // World counter-clockwise is a negative canvas rotation (y points down).
    ctx.rotate(-p.theta);

    // Flame, along the deflected thrust axis, from the tail.
    if (engineOn && p.thrust > 0.01 && !ended) {
      ctx.save();
      ctx.translate(0, tail);
      ctx.rotate(p.gimbal);
      const flameLen = len * (0.25 + 0.55 * p.thrust) * (0.9 + Math.random() * 0.2);
      const g = ctx.createLinearGradient(0, 0, 0, flameLen);
      g.addColorStop(0, "rgba(255, 245, 210, 0.95)");
      g.addColorStop(0.25, "rgba(255, 179, 71, 0.85)");
      g.addColorStop(1, "rgba(255, 90, 30, 0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.moveTo(-width * 0.32, 0);
      ctx.quadraticCurveTo(-width * 0.5, flameLen * 0.4, 0, flameLen);
      ctx.quadraticCurveTo(width * 0.5, flameLen * 0.4, width * 0.32, 0);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    // Legs: struts from the lower body to the leg tips at tail level.
    ctx.strokeStyle = COLORS.leg;
    ctx.lineWidth = Math.max(1.5, 0.5 * s);
    ctx.beginPath();
    ctx.moveTo(-width / 2, tail - 0.18 * len);
    ctx.lineTo(-span, tail);
    ctx.moveTo(width / 2, tail - 0.18 * len);
    ctx.lineTo(span, tail);
    ctx.stroke();

    // Engine bell.
    ctx.fillStyle = COLORS.band;
    ctx.beginPath();
    ctx.moveTo(-width * 0.3, tail - 0.02 * len);
    ctx.lineTo(width * 0.3, tail - 0.02 * len);
    ctx.lineTo(width * 0.4, tail);
    ctx.lineTo(-width * 0.4, tail);
    ctx.closePath();
    ctx.fill();

    // Body with simple shading.
    const body = ctx.createLinearGradient(-width / 2, 0, width / 2, 0);
    body.addColorStop(0, COLORS.bodyShade);
    body.addColorStop(0.45, COLORS.body);
    body.addColorStop(1, COLORS.bodyShade);
    ctx.fillStyle = body;
    ctx.fillRect(-width / 2, -nose, width, len - 0.02 * len);

    // Interstage band and grid fins near the top.
    ctx.fillStyle = COLORS.band;
    ctx.fillRect(-width / 2, -nose, width, 0.1 * len);
    ctx.fillStyle = COLORS.fin;
    const finY = -nose + 0.03 * len;
    ctx.fillRect(-width / 2 - width * 0.45, finY, width * 0.45, 0.035 * len);
    ctx.fillRect(width / 2, finY, width * 0.45, 0.035 * len);

    // RCS: white cold-gas jets near the nose, exhausting opposite to the push.
    // A counter-clockwise torque pushes the nose left, so the gas goes right.
    if (Math.abs(p.rcs) > 1 && !ended) {
      const dir = p.rcs > 0 ? 1 : -1;
      const y = -nose + 0.06 * len;
      const reach = Math.max(16, 0.3 * len);
      for (let i = 0; i < 3; i += 1) {
        const l = reach * (0.6 + Math.random() * 0.5);
        const spread = l * (0.18 + 0.06 * i);
        const x0 = dir * (width / 2);
        const g = ctx.createLinearGradient(x0, y, x0 + dir * l, y);
        g.addColorStop(0, "rgba(255, 255, 255, 0.85)");
        g.addColorStop(1, "rgba(255, 255, 255, 0)");
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.moveTo(x0, y - 1);
        ctx.lineTo(x0 + dir * l, y - spread);
        ctx.lineTo(x0 + dir * l, y + spread);
        ctx.lineTo(x0, y + 1);
        ctx.closePath();
        ctx.fill();
      }
    }

    ctx.restore();
  }

  private drawVelocity(cam: Camera, p: Pose): void {
    const speed = Math.hypot(p.vx, p.vy);
    if (speed < 0.5) return;
    const ctx = this.ctx;
    const px = this.sx(cam, p.x);
    const py = this.sy(cam, p.y);
    const length = Math.min(90, 18 + 14 * Math.log1p(speed));
    const dx = (p.vx / speed) * length;
    const dy = (-p.vy / speed) * length;
    ctx.strokeStyle = COLORS.velocity;
    ctx.fillStyle = COLORS.velocity;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px + dx, py + dy);
    ctx.stroke();
    const a = Math.atan2(dy, dx);
    ctx.beginPath();
    ctx.moveTo(px + dx, py + dy);
    ctx.lineTo(px + dx - 7 * Math.cos(a - 0.4), py + dy - 7 * Math.sin(a - 0.4));
    ctx.lineTo(px + dx - 7 * Math.cos(a + 0.4), py + dy - 7 * Math.sin(a + 0.4));
    ctx.closePath();
    ctx.fill();
  }

  private drawOutcome(cam: Camera, cur: Telemetry, p: Pose, now: number): void {
    if (cur.outcome === 0) {
      this.endTime = null;
      return;
    }
    this.endTime ??= now;
    const age = (now - this.endTime) / 1000;
    const ctx = this.ctx;
    const px = this.sx(cam, p.x);
    const py = this.sy(cam, Math.max(p.y - p.cm, 0));
    const crashed = cur.outcome >= 2 && cur.outcome <= 7; // ground impact codes
    if (cur.outcome === 1) {
      const pulse = 0.5 + 0.5 * Math.sin(age * 4);
      ctx.strokeStyle = `rgba(80, 230, 150, ${0.35 + 0.4 * pulse})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(px, this.sy(cam, 0), 40 + 6 * pulse, 8, 0, 0, Math.PI * 2);
      ctx.stroke();
    } else if (crashed && age < 1.2) {
      const r = 20 + age * 160;
      const g = ctx.createRadialGradient(px, py, 0, px, py, r);
      g.addColorStop(0, `rgba(255, 230, 160, ${0.9 * (1 - age / 1.2)})`);
      g.addColorStop(0.4, `rgba(255, 120, 40, ${0.6 * (1 - age / 1.2)})`);
      g.addColorStop(1, "rgba(255, 60, 20, 0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
