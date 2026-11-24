// Text HUD in a mission-control style. All dynamic text goes through
// `textContent`, never `innerHTML`.

import { Q_CENTER } from "./controls";
import type { Telemetry, VehicleInfo } from "./sim";

const RAD_TO_DEG = 180 / Math.PI;

type Level = "ok" | "warn" | "bad" | "idle";

interface Row {
  value: HTMLSpanElement;
  row: HTMLDivElement;
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

function signed(v: number, digits: number): string {
  const s = v.toFixed(digits);
  return v > 0 && s !== (0).toFixed(digits) ? `+${s}` : s;
}

/** Wraps an angle to (−π, π]. */
function wrap(a: number): number {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

/** Throttle fraction of max thrust commanded by a quantized level, or null if the engine is commanded off. */
export function commandedFraction(level: number, info: VehicleInfo): number | null {
  const u = (Math.min(level, 254) - Q_CENTER) / Q_CENTER;
  if (u < info.throttle_cutoff) return null;
  const t = Math.min(1, Math.max(0, (u - info.throttle_cutoff) / (1 - info.throttle_cutoff)));
  return info.min_throttle + (1 - info.min_throttle) * t;
}

export class Hud {
  private readonly rows = new Map<string, Row>();
  private readonly fuelBar: HTMLDivElement;
  private readonly gates = new Map<string, HTMLSpanElement>();
  private readonly faults: HTMLDivElement;

  constructor(root: HTMLElement) {
    root.replaceChildren();
    const section = (title: string): HTMLDivElement => {
      const s = el("div", "hud-section");
      s.append(el("div", "hud-title", title));
      root.append(s);
      return s;
    };
    const add = (parent: HTMLElement, key: string, label: string): void => {
      const row = el("div", "hud-row");
      const value = el("span", "hud-value", "—");
      row.append(el("span", "hud-label", label), value);
      parent.append(row);
      this.rows.set(key, { value, row });
    };

    const nav = section("NAVIGATION");
    add(nav, "met", "MET");
    add(nav, "alt", "ALT");
    add(nav, "vs", "V/S");
    add(nav, "hs", "H/S");
    add(nav, "dx", "PAD Δx");
    add(nav, "tilt", "TILT");
    add(nav, "rate", "RATE");

    const prop = section("PROPULSION");
    add(prop, "engine", "ENGINE");
    add(prop, "cmd", "THR CMD");
    add(prop, "thrust", "THRUST");
    add(prop, "gimbal", "GIMBAL");
    add(prop, "rcs", "RCS");
    add(prop, "ign", "IGNITIONS");
    add(prop, "fuel", "FUEL");
    const bar = el("div", "hud-bar");
    this.fuelBar = el("div", "hud-bar-fill");
    bar.append(this.fuelBar);
    prop.append(bar);

    const env = section("ENVIRONMENT");
    add(env, "wind", "WIND");
    this.faults = el("div", "hud-faults");
    env.append(this.faults);

    const gates = section("TOUCHDOWN GATES");
    const chips = el("div", "hud-gates");
    for (const [key, label] of [
      ["vs", "V/S"],
      ["hs", "H/S"],
      ["tilt", "TILT"],
      ["rate", "RATE"],
      ["pad", "PAD"],
    ] as const) {
      const chip = el("span", "gate", label);
      chips.append(chip);
      this.gates.set(key, chip);
    }
    gates.append(chips);
  }

  private set(key: string, text: string, level: Level = "idle"): void {
    const r = this.rows.get(key);
    if (!r) return;
    if (r.value.textContent !== text) r.value.textContent = text;
    r.row.dataset["level"] = level;
  }

  private gate(key: string, label: string, ok: boolean): void {
    const chip = this.gates.get(key);
    if (!chip) return;
    const text = `${label} ${ok ? "✓" : "✗"}`;
    if (chip.textContent !== text) chip.textContent = text;
    chip.dataset["ok"] = ok ? "1" : "0";
    chip.setAttribute("aria-label", `${label} ${ok ? "within" : "outside"} the landing limit`);
  }

  update(t: Telemetry, info: VehicleInfo, throttleLevel: number, rcsLevel: number): void {
    const near = t.clearance < 150;
    const tilt = wrap(t.theta);
    // Speeds are those of the lowest leg tip, exactly what the core judges at
    // touchdown (centre-of-mass speed plus rotation), with the same strict limits.
    const vsOk = Math.abs(t.contact_vy) < info.max_vertical_speed;
    const hsOk = Math.abs(t.contact_vx) < info.max_horizontal_speed;
    const tiltOk = Math.abs(tilt) < info.max_tilt;
    const rateOk = Math.abs(t.omega) < info.max_angular_rate;
    const padOk = t.pad_margin >= 0;
    const judge = (ok: boolean, marginal: boolean): Level => (!near ? "idle" : ok ? "ok" : marginal ? "warn" : "bad");

    this.set("met", `T+${t.time.toFixed(1)} s`);
    this.set("alt", `${Math.max(0, t.clearance).toFixed(1)} m`);
    this.set(
      "vs",
      `${signed(t.contact_vy, 1)} m/s`,
      judge(vsOk, Math.abs(t.contact_vy) < 3 * info.max_vertical_speed),
    );
    this.set(
      "hs",
      `${signed(t.contact_vx, 1)} m/s`,
      judge(hsOk, Math.abs(t.contact_vx) < 3 * info.max_horizontal_speed),
    );
    this.set("dx", `${signed(t.tail_x, 1)} m`, judge(padOk, t.pad_margin > -info.pad_half_width));
    this.set("tilt", `${signed(tilt * RAD_TO_DEG, 1)}°`, judge(tiltOk, Math.abs(tilt) < 2 * info.max_tilt));
    this.set("rate", `${signed(t.omega * RAD_TO_DEG, 1)}°/s`, judge(rateOk, Math.abs(t.omega) < 2 * info.max_angular_rate));

    const on = t.engine_on > 0.5;
    this.set("engine", on ? "RUNNING" : "OFF", on ? "ok" : "idle");
    const frac = commandedFraction(throttleLevel, info);
    const canIgnite = t.ignitions_left > 0 && t.propellant > 0;
    this.set(
      "cmd",
      frac === null ? "CUT-OFF" : !on && !canIgnite ? "NO IGNITION" : `${(frac * 100).toFixed(0)} %`,
      frac !== null && !on && !canIgnite ? "bad" : "idle",
    );
    this.set("thrust", `${(t.thrust_fraction * 100).toFixed(0)} %`);
    this.set("gimbal", `${signed(t.gimbal * RAD_TO_DEG, 1)}°`, t.gimbal_stuck > 0.5 ? "bad" : "idle");
    this.set("rcs", rcsLevel > Q_CENTER ? "◀ CCW" : rcsLevel < Q_CENTER ? "CW ▶" : "—");
    const ign = t.ignitions_left;
    this.set("ign", `${ign}`, ign === 0 && !on ? "bad" : "idle");
    const fuelFrac = info.initial_propellant > 0 ? t.propellant / info.initial_propellant : 0;
    this.set("fuel", `${Math.round(t.propellant).toLocaleString("en-US")} kg`, fuelFrac < 0.15 ? "warn" : "idle");
    this.fuelBar.style.width = `${Math.max(0, Math.min(1, fuelFrac)) * 100}%`;
    this.fuelBar.dataset["low"] = fuelFrac < 0.15 ? "1" : "0";

    const dir = t.wind > 0.05 ? "→" : t.wind < -0.05 ? "←" : "";
    this.set("wind", `${Math.abs(t.wind).toFixed(1)} m/s ${dir}`);
    const faults: string[] = [];
    if (t.gimbal_stuck > 0.5) faults.push(`GIMBAL STUCK ${signed(t.gimbal_stuck_angle * RAD_TO_DEG, 1)}°`);
    if (t.thrust_factor < 1) faults.push(`THRUST LIMITED ${(t.thrust_factor * 100).toFixed(0)} %`);
    const text = faults.join(" · ");
    if (this.faults.textContent !== text) this.faults.textContent = text;

    this.gate("vs", "V/S", vsOk);
    this.gate("hs", "H/S", hsOk);
    this.gate("tilt", "TILT", tiltOk);
    this.gate("rate", "RATE", rateOk);
    this.gate("pad", "PAD", padOk);
  }
}
