// Post-flight debrief and briefing content: what happened, how to fix it, and
// the touchdown numbers as a small table of measured value against limit.
// Pure DOM building with textContent only.

import type { Telemetry, VehicleInfo } from "./sim";

const RAD_TO_DEG = 180 / Math.PI;

export interface Stat {
  label: string;
  value: string;
  unit?: string;
  /** Short context under the value, e.g. "3 ignitions". */
  note?: string;
}

/** One touchdown check: what the stage did against what the judge allows. */
export interface Check {
  label: string;
  value: string;
  limit: string;
  ok: boolean;
}

const TITLES: Record<string, string> = {
  landed: "Landed",
  crashed_tipped_over: "Tipped over",
  crashed_off_pad: "Missed the pad",
  crashed_vertical_speed: "Too fast",
  crashed_horizontal_speed: "Sliding sideways",
  crashed_tilt: "Too tilted",
  crashed_angular_rate: "Spinning",
  out_of_propellant: "Out of fuel",
  out_of_bounds: "Left the landing area",
  time_limit: "Out of time",
  diverged: "Simulation error",
};

/** Headline for an outcome name ("" for a replay that ended before touchdown). */
export function outcomeTitle(outcome: string): string {
  return TITLES[outcome] ?? (outcome === "" ? "Replay ended" : outcome);
}

/** One sentence on how to avoid this outcome next time, or null. */
export function outcomeAdvice(outcome: string, info: VehicleInfo): string | null {
  switch (outcome) {
    case "crashed_vertical_speed":
      return `Brake earlier: be under ${info.max_vertical_speed} m/s down well before the legs touch.`;
    case "crashed_horizontal_speed":
      return "Cancel sideways drift with ← → before the last 50 m.";
    case "crashed_tilt":
      return "Stand the stage upright with Q and E before touchdown.";
    case "crashed_angular_rate":
      return "Stop the rotation with Q and E before touchdown.";
    case "crashed_off_pad":
      return "Steer toward the pad early: drift is hard to fix close to the ground.";
    case "crashed_tipped_over":
      return "Touch down slow and upright: the legs cannot catch a lean.";
    case "out_of_propellant":
      return "Hovering burns fuel fast. Descend steadily and brake near the ground.";
    case "out_of_bounds":
      return "Stay over the landing area and head for the pad.";
    case "time_limit":
      return "Keep descending: the flight has a time limit.";
    default:
      return null;
  }
}

/** Outcomes decided at touchdown, where the touchdown numbers explain the result. */
const TOUCHDOWN = new Set([
  "landed",
  "crashed_tipped_over",
  "crashed_off_pad",
  "crashed_vertical_speed",
  "crashed_horizontal_speed",
  "crashed_tilt",
  "crashed_angular_rate",
]);

/** 3 → "3", 2.5 → "2.5". */
function num(v: number): string {
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
}

/**
 * A value to one decimal that never contradicts its verdict: 2.97 under a
 * limit of 3 reads "2.9", not "3.0".
 */
function shown(v: number, limit: number, ok: boolean): string {
  const s = v.toFixed(1);
  return ok && Number(s) >= limit ? (Math.floor(v * 10) / 10).toFixed(1) : s;
}

/**
 * The touchdown checks for the debrief: descent, sideways and tilt against
 * their limits, plus the miss distance when a leg landed off the pad. Empty
 * when the flight ended before touchdown (fuel, bounds, time).
 */
export function touchdownChecks(outcome: string, s: Telemetry, info: VehicleInfo): Check[] {
  if (!TOUCHDOWN.has(outcome)) return [];
  const vy = Math.abs(s.contact_vy);
  const vx = Math.abs(s.contact_vx);
  const tilt = Math.abs(Math.atan2(Math.sin(s.theta), Math.cos(s.theta))) * RAD_TO_DEG;
  const maxTilt = info.max_tilt * RAD_TO_DEG;
  const check = (label: string, v: number, limit: number, unit: string, limitText = num(limit)): Check => {
    const ok = v < limit;
    return { label, value: `${shown(v, limit, ok)}${unit}`, limit: `${limitText}${unit}`, ok };
  };
  const checks: Check[] = [
    check("Descent", vy, info.max_vertical_speed, " m/s"),
    check("Sideways", vx, info.max_horizontal_speed, " m/s"),
    check("Tilt", tilt, maxTilt, "°", `${Math.round(maxTilt)}`),
  ];
  if (outcome === "crashed_angular_rate") {
    checks.push(check("Spin", Math.abs(s.omega) * RAD_TO_DEG, info.max_angular_rate * RAD_TO_DEG, "°/s", `${Math.round(info.max_angular_rate * RAD_TO_DEG)}`));
  }
  const miss = Math.max(0, -s.pad_margin);
  if (outcome === "crashed_off_pad" || miss >= 0.05) {
    checks.push({ label: "Off the pad", value: `${Math.max(0.1, miss).toFixed(1)} m`, limit: "0 m", ok: false });
  }
  return checks;
}

/** Touchdown limits and engine state for the briefing. */
export function limitStats(info: VehicleInfo, ignitions: number, engineOn: boolean): Stat[] {
  return [
    { label: "Max descent", value: num(info.max_vertical_speed), unit: "m/s" },
    { label: "Max sideways", value: num(info.max_horizontal_speed), unit: "m/s" },
    { label: "Max tilt", value: `${Math.round(info.max_tilt * RAD_TO_DEG)}`, unit: "°" },
    { label: "Engine", value: engineOn ? "On" : "Off", note: `${ignitions} ignition${ignitions === 1 ? "" : "s"}` },
  ];
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

export function statGrid(stats: readonly Stat[]): HTMLElement {
  const grid = el("dl", "stats");
  // Columns divide the count evenly (4 → 4 or 2, 6 → 3 or 2): no orphan tile.
  grid.dataset["count"] = String(stats.length);
  for (const s of stats) {
    const cell = el("div", "stat");
    const value = el("dd", "stat-value");
    value.append(el("span", "stat-number", s.value));
    if (s.unit) value.append(el("span", "stat-unit", s.unit));
    cell.append(el("dt", "stat-label", s.label), value);
    if (s.note) cell.append(el("dd", "stat-note", s.note));
    grid.append(cell);
  }
  return grid;
}

/** The debrief table: one row per check, "You" against "Limit", then OK / Over. */
export function checkTable(checks: readonly Check[]): HTMLElement {
  const table = el("table", "checks");
  table.setAttribute("aria-label", "Touchdown values against their limits");
  const head = el("tr", "");
  const corner = el("td", "");
  const verdict = el("td", "");
  const you = el("th", "", "You");
  const max = el("th", "", "Limit");
  you.scope = "col";
  max.scope = "col";
  head.append(corner, you, max, verdict);
  const thead = el("thead", "");
  thead.append(head);
  const body = el("tbody", "");
  for (const c of checks) {
    const row = el("tr", "");
    row.dataset["state"] = c.ok ? "ok" : "bad";
    const label = el("th", "check-label", c.label);
    label.scope = "row";
    row.append(
      label,
      el("td", "check-value", c.value),
      el("td", "check-limit", c.limit),
      el("td", "check-state", c.ok ? "OK" : "Over"),
    );
    body.append(row);
  }
  table.append(thead, body);
  return table;
}
