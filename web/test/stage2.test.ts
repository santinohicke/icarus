// Stage 2 features without a browser: local scores, share links, the touch
// throttle, the chart buffer, and replays played back through ReplayPlayer.

import { readFileSync } from "node:fs";

import { beforeAll, describe, expect, it } from "vitest";

import { InputState, Q_MAX, minRunningLevel } from "../src/controls";
import { SampleBuffer } from "../src/samples";
import { MAX_SCORES_PER_CONFIG, SCORES_KEY, ScoreBoard, type KeyValueStore, type ScoreEntry } from "../src/scores";
import { MAX_REPLAY_LENGTH, replayFromHash, replayUrl, scenarioUrl } from "../src/share";
import { ReplayPlayer, Simulation, loadWasm } from "../src/sim";

class MemoryStore implements KeyValueStore {
  readonly data = new Map<string, string>();
  failWrites = false;
  getItem(key: string): string | null {
    return this.data.get(key) ?? null;
  }
  setItem(key: string, value: string): void {
    if (this.failWrites) throw new Error("QuotaExceededError");
    this.data.set(key, value);
  }
}

function entry(score: number, overrides: Partial<ScoreEntry> = {}): ScoreEntry {
  return { config: "easy", seed: "42", score, date: 1_700_000_000_000 + score, replay: "SUNSUAAB", physics: 1, ...overrides };
}

describe("ScoreBoard", () => {
  it("ranks scores per config, best first", () => {
    const board = new ScoreBoard(new MemoryStore(), 1);
    expect(board.add(entry(1500))).toBe(1);
    expect(board.add(entry(1700))).toBe(1);
    expect(board.add(entry(1600))).toBe(2);
    expect(board.add(entry(1000, { config: "realistic" }))).toBe(1);
    expect(board.top("easy").map((e) => e.score)).toEqual([1700, 1600, 1500]);
    expect(board.best("realistic")).toBe(1000);
    expect(board.best("easy")).toBe(1700);
  });

  it("keeps only the top entries of each config", () => {
    const board = new ScoreBoard(new MemoryStore(), 1);
    for (let i = 0; i < MAX_SCORES_PER_CONFIG; i += 1) board.add(entry(1100 + i));
    board.add(entry(1000, { config: "realistic" }));
    expect(board.add(entry(1050))).toBeNull(); // below the 10th easy score
    expect(board.add(entry(2000))).toBe(1);
    expect(board.top("easy")).toHaveLength(MAX_SCORES_PER_CONFIG);
    expect(board.top("realistic")).toHaveLength(1);
  });

  it("ignores corrupt or hostile storage", () => {
    const store = new MemoryStore();
    const board = new ScoreBoard(store, 1);
    store.data.set(SCORES_KEY, "{not json");
    expect(board.all()).toEqual([]);
    store.data.set(
      SCORES_KEY,
      JSON.stringify([
        entry(1200),
        { ...entry(1300), config: "hard" },
        { ...entry(1400), seed: "-1" },
        { ...entry(1500), replay: "<script>" },
        { ...entry(1600), score: 1.5 },
        null,
        "x",
      ]),
    );
    expect(board.all().map((e) => e.score)).toEqual([1200]);
  });

  it("hides and then drops entries from another physics version", () => {
    const store = new MemoryStore();
    store.data.set(SCORES_KEY, JSON.stringify([entry(1900, { physics: 0 }), entry(1200)]));
    const board = new ScoreBoard(store, 1);
    expect(board.top("easy").map((e) => e.score)).toEqual([1200]);
    expect(board.add(entry(1300))).toBe(1);
    expect(store.data.get(SCORES_KEY)).not.toContain("1900");
    expect(board.add(entry(1400, { physics: 2 }))).toBeNull();
  });

  it("survives storage that is missing or refuses writes", () => {
    expect(new ScoreBoard(null, 1).add(entry(1500))).toBeNull();
    expect(new ScoreBoard(null, 1).all()).toEqual([]);
    const store = new MemoryStore();
    store.failWrites = true;
    expect(new ScoreBoard(store, 1).add(entry(1500))).toBeNull();
  });
});

describe("share links", () => {
  const base = "https://example.github.io/icarus/?config=easy&seed=5#old";

  it("round-trips a replay through the hash", () => {
    const url = replayUrl(base, "SUNSUAAB_-x");
    expect(url).toBe("https://example.github.io/icarus/#replay=SUNSUAAB_-x");
    expect(replayFromHash(new URL(url).hash)).toBe("SUNSUAAB_-x");
  });

  it("rejects malformed replay hashes", () => {
    expect(replayFromHash("")).toBeNull();
    expect(replayFromHash("#replay=")).toBeNull();
    expect(replayFromHash("#other=abc")).toBeNull();
    expect(replayFromHash("#replay=abc%20def")).toBeNull();
    expect(replayFromHash("#replay=abc<def")).toBeNull();
    expect(replayFromHash(`#replay=${"A".repeat(MAX_REPLAY_LENGTH + 1)}`)).toBeNull();
  });

  it("builds scenario links with seeds above 2^53", () => {
    const seed = (1n << 64n) - 1n;
    const url = new URL(scenarioUrl(base, "realistic", seed));
    expect(url.hash).toBe("");
    expect(url.searchParams.get("config")).toBe("realistic");
    expect(url.searchParams.get("seed")).toBe(seed.toString());
  });
});

describe("touch throttle", () => {
  const minRunning = minRunningLevel(-0.9);

  it("stops at minimum throttle while the engine runs", () => {
    const input = new InputState(200, minRunning);
    input.setThrottle(0);
    expect(input.throttleLevel).toBe(minRunning);
    input.setThrottle(Q_MAX + 50);
    expect(input.throttleLevel).toBe(Q_MAX);
    input.setThrottle(Number.NaN);
    expect(input.throttleLevel).toBe(Q_MAX);
  });

  it("can start from cut-off and only the cut control shuts the engine", () => {
    const input = new InputState(0, minRunning);
    input.setThrottle(minRunning - 1); // still below: stays cut
    expect(input.throttleLevel).toBe(minRunning - 1);
    input.setThrottle(150); // above: commands ignition
    expect(input.throttleLevel).toBe(150);
    input.press("KeyX");
    input.release("KeyX");
    expect(input.sample()[0]).toBe(0);
  });
});

describe("SampleBuffer", () => {
  it("drops duplicate ticks and the oldest samples beyond capacity", () => {
    const b = new SampleBuffer(2, 3);
    expect(b.push(0, 0, [1, 2])).toBe(true);
    expect(b.push(0, 0, [9, 9])).toBe(false);
    b.push(6, 0.1, [3, Number.POSITIVE_INFINITY]);
    b.push(12, 0.2, [5, 6]);
    b.push(18, 0.3, [7, 8]);
    expect(b.time).toEqual([0.1, 0.2, 0.3]);
    expect(b.series[0]).toEqual([3, 5, 7]);
    expect(b.series[1]?.[0]).toBeNaN();
    b.clear();
    expect(b.length).toBe(0);
    expect(b.push(0, 0, [1, 1])).toBe(true);
  });
});

describe("ReplayPlayer", () => {
  beforeAll(async () => {
    await loadWasm(readFileSync(new URL("../src/wasm/icarus_wasm_bg.wasm", import.meta.url)));
  });

  it("replays a recorded run with failures to the identical final state", () => {
    const sim = new Simulation("realistic", 987654321987654321n);
    const inputs: [number, number, number][] = [];
    for (let tick = 0; tick < 2_000 && sim.outcome === 0; tick += 1) {
      if (tick === 90) sim.injectGust(12, 3);
      if (tick === 200) sim.injectThrustLoss(0.7);
      if (tick === 300) sim.injectGimbalStuck(0.017);
      const q: [number, number, number] = [tick < 400 ? 0 : 200, (tick * 13) % 255, tick % 90 < 45 ? 127 : 254];
      inputs.push(q);
      sim.step(...q);
    }
    const encoded = sim.replay();
    const player = new ReplayPlayer(encoded);
    expect(player.config).toBe("realistic");
    expect(player.seed).toBe(987654321987654321n);
    expect(player.totalTicks).toBe(inputs.length);
    expect(player.lastInput).toEqual([player.info.initial_throttle_level, 127, 127]);
    let ticks = 0;
    while (!player.step()) {
      ticks += 1;
      expect(player.lastInput).toEqual(inputs[ticks - 1]);
    }
    expect(player.done).toBe(true);
    expect(player.state).toEqual(sim.state);
    expect(player.outcomeName).toBe(sim.outcomeName);
    expect(player.score).toBe(sim.score);
    expect(player.step()).toBe(true); // stepping past the end is harmless
    player.dispose();
    sim.dispose();
  });

  it("rejects invalid replay strings", () => {
    expect(() => new ReplayPlayer("not-a-replay")).toThrow();
    expect(() => new ReplayPlayer("")).toThrow();
  });
});
