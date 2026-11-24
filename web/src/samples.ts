// Column store behind the telemetry charts (no DOM, testable).

/** Longest history kept (10 min at 10 Hz, longer than any episode). */
export const MAX_SAMPLES = 6_000;

/** Fixed-capacity column store for the chart data (pure, testable). */
export class SampleBuffer {
  readonly time: number[] = [];
  readonly series: number[][];
  private lastTick = -1;

  constructor(
    readonly columns: number,
    readonly capacity = MAX_SAMPLES,
  ) {
    this.series = Array.from({ length: columns }, () => []);
  }

  clear(): void {
    this.time.length = 0;
    for (const s of this.series) s.length = 0;
    this.lastTick = -1;
  }

  get length(): number {
    return this.time.length;
  }

  /** Adds a sample unless one was already taken at this tick. Returns true if added. */
  push(tick: number, time: number, values: readonly number[]): boolean {
    if (tick === this.lastTick) return false;
    this.lastTick = tick;
    if (this.time.length >= this.capacity) {
      this.time.shift();
      for (const s of this.series) s.shift();
    }
    this.time.push(time);
    this.series.forEach((s, i) => s.push(Number.isFinite(values[i]) ? (values[i] as number) : Number.NaN));
    return true;
  }
}
