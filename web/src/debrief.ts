// Post-flight debrief and briefing content: what happened, how to fix it, and
// the touchdown numbers laid out as measured value against limit. Pure DOM
// building with textContent only.

import type { Telemetry, VehicleInfo } from "./sim";

const RAD_TO_DEG = 180 / Math.PI;

export interface Stat {
  label: string;
  value: string;
  unit?: string;
  /** Short context under the value, e.g. "limit 3.0". */
  note?: string;
  state?: "ok" | "bad";
  /** Bar fill in [0, 1]; no bar when absent. */
  fill?: number;
  /** Position of the limit tick on the bar, in [0, 1]. */
  mark?: number;
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

function speedStat(label: string, v: number, limit: number): Stat {
  const a = Math.abs(v);
  return {
    label,
    value: a.toFixed(1),
    unit: "m/s",
    note: `limit ${limit.toFixed(1)}`,
    state: a < limit ? "ok" : "bad",
    fill: Math.min(1, a / (2 * limit)),
    mark: 0.5,
  };
}

/** The numbers the judge looks at, at the end of a flight. */
export function flightStats(s: Telemetry, info: VehicleInfo): Stat[] {
  const tilt = Math.abs(Math.atan2(Math.sin(s.theta), Math.cos(s.theta)));
  const maxTilt = info.max_tilt;
  const miss = Math.max(0, -s.pad_margin);
  const fuel = info.initial_propellant > 0 ? Math.max(0, s.propellant / info.initial_propellant) : 0;
  return [
    speedStat("Descent speed", s.contact_vy, info.max_vertical_speed),
    speedStat("Sideways speed", s.contact_vx, info.max_horizontal_speed),
    {
      label: "Tilt",
      value: (tilt * RAD_TO_DEG).toFixed(1),
      unit: "°",
      note: `limit ${(maxTilt * RAD_TO_DEG).toFixed(0)}°`,
      state: tilt < maxTilt ? "ok" : "bad",
      fill: Math.min(1, tilt / (2 * maxTilt)),
      mark: 0.5,
    },
    {
      label: "Off the pad",
      value: miss.toFixed(1),
      unit: "m",
      note: miss === 0 ? "both legs on the pad" : "a leg missed the pad",
      state: miss === 0 ? "ok" : "bad",
    },
    {
      label: "Fuel left",
      value: Math.round(s.propellant).toLocaleString("en-US"),
      unit: "kg",
      note: `${Math.round(fuel * 100)}% of the load`,
      fill: Math.min(1, fuel),
    },
    { label: "Flight time", value: s.time.toFixed(1), unit: "s" },
  ];
}

/** Touchdown limits for the briefing. */
export function limitStats(info: VehicleInfo, ignitions: number, engineOn: boolean): Stat[] {
  return [
    { label: "Descent speed", value: `${info.max_vertical_speed}`, unit: "m/s", note: "at most" },
    { label: "Sideways speed", value: `${info.max_horizontal_speed}`, unit: "m/s", note: "at most" },
    { label: "Tilt", value: `${Math.round(info.max_tilt * RAD_TO_DEG)}`, unit: "°", note: "at most" },
    { label: "Engine", value: engineOn ? "On" : "Off", note: `${ignitions} ignition${ignitions === 1 ? "" : "s"} left` },
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
    if (s.state) cell.dataset["state"] = s.state;
    const value = el("dd", "stat-value");
    value.append(el("span", "stat-number", s.value));
    if (s.unit) value.append(el("span", "stat-unit", s.unit));
    cell.append(el("dt", "stat-label", s.label), value);
    if (s.fill !== undefined) {
      const bar = el("dd", "stat-bar");
      bar.setAttribute("aria-hidden", "true");
      const fill = el("span", "stat-fill");
      fill.style.width = `${(Math.max(0, Math.min(1, s.fill)) * 100).toFixed(1)}%`;
      bar.append(fill);
      if (s.mark !== undefined) {
        const mark = el("span", "stat-mark");
        mark.style.left = `${(s.mark * 100).toFixed(1)}%`;
        bar.append(mark);
      }
      cell.append(bar);
    }
    if (s.note) cell.append(el("dd", "stat-note", s.note));
    grid.append(cell);
  }
  return grid;
}
