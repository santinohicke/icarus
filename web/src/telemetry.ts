// Live telemetry charts (uPlot): altitude, vertical speed, throttle and tilt
// over mission time. Sampled a few times per second, redrawn at most once per
// animation frame.

import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

import { SampleBuffer } from "./samples";
import type { Telemetry } from "./sim";

/** Simulation ticks between samples (60 Hz / 6 = 10 samples per second). */
export const SAMPLE_EVERY_TICKS = 6;

const RAD_TO_DEG = 180 / Math.PI;

interface Panel {
  readonly title: string;
  readonly unit: string;
  readonly color: string;
  readonly value: (t: Telemetry) => number;
}

const PANELS: readonly Panel[] = [
  { title: "Altitude", unit: "m", color: "#f2f0ea", value: (t) => Math.max(0, t.clearance) },
  { title: "Vertical speed", unit: "m/s", color: "#f6c453", value: (t) => t.contact_vy },
  { title: "Thrust", unit: "%", color: "#ff9447", value: (t) => t.thrust_fraction * 100 },
  {
    title: "Tilt",
    unit: "°",
    color: "#4fd69c",
    value: (t) => Math.atan2(Math.sin(t.theta), Math.cos(t.theta)) * RAD_TO_DEG,
  },
];

export class TelemetryCharts {
  private readonly buffer = new SampleBuffer(PANELS.length);
  private readonly plots: uPlot[] = [];
  private dirty = false;
  private visible = false;

  constructor(private readonly root: HTMLElement) {
    root.replaceChildren();
    for (const panel of PANELS) {
      const box = document.createElement("div");
      box.className = "chart";
      root.append(box);
      this.plots.push(
        new uPlot(
          {
            width: 300,
            height: 90,
            title: `${panel.title} (${panel.unit})`,
            legend: { show: false },
            cursor: { show: false },
            scales: { x: { time: false } },
            axes: [
              { stroke: "#9aa6bd", grid: { stroke: "rgba(226,232,244,0.08)" }, size: 22, font: "10px \"JetBrains Mono\", monospace" },
              { stroke: "#9aa6bd", grid: { stroke: "rgba(226,232,244,0.08)" }, size: 42, font: "10px \"JetBrains Mono\", monospace" },
            ],
            series: [{}, { stroke: panel.color, width: 1.5, points: { show: false } }],
          },
          [[], []],
          box,
        ),
      );
    }
    new ResizeObserver(() => this.resize()).observe(root);
  }

  private resize(): void {
    const width = Math.max(160, Math.floor(this.root.clientWidth - 16));
    for (const p of this.plots) p.setSize({ width, height: 90 });
  }

  setVisible(visible: boolean): void {
    this.visible = visible;
    this.root.hidden = !visible;
    if (visible) {
      this.resize();
      this.dirty = true;
    }
  }

  get isVisible(): boolean {
    return this.visible;
  }

  reset(t: Telemetry): void {
    this.buffer.clear();
    this.record(t, true);
  }

  /** Call after every simulation tick; keeps one sample every SAMPLE_EVERY_TICKS. */
  record(t: Telemetry, force = false): void {
    if (!force && t.tick % SAMPLE_EVERY_TICKS !== 0 && t.outcome === 0) return;
    if (
      this.buffer.push(
        t.tick,
        t.time,
        PANELS.map((p) => p.value(t)),
      )
    ) {
      this.dirty = true;
    }
  }

  /** Call once per animation frame. */
  draw(): void {
    if (!this.visible || !this.dirty) return;
    this.dirty = false;
    this.plots.forEach((plot, i) => plot.setData([this.buffer.time, this.buffer.series[i] ?? []]));
  }
}
