// Local high scores, kept in localStorage (no server).
//
// Everything read back from storage is validated: the data may be stale,
// edited by hand or written by an older version.

import type { Config } from "./sim";
import { isConfig } from "./sim";

export const SCORES_KEY = "icarus.scores.v1";
export const MAX_SCORES_PER_CONFIG = 10;
/** Replays above this length are not stored (keeps localStorage small). */
export const MAX_STORED_REPLAY = 64 * 1024;

export interface ScoreEntry {
  readonly config: Config;
  /** Decimal u64. */
  readonly seed: string;
  readonly score: number;
  /** Milliseconds since the epoch. */
  readonly date: number;
  readonly replay: string;
  /**
   * Physics version the run was flown with. Entries from another version are
   * hidden: their replays no longer decode and their scores do not compare.
   */
  readonly physics: number;
}

/** Storage subset used here (window.localStorage, or a fake in tests). */
export interface KeyValueStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

function isEntry(v: unknown): v is ScoreEntry {
  if (typeof v !== "object" || v === null) return false;
  const e = v as Record<string, unknown>;
  return (
    isConfig(e["config"]) &&
    typeof e["seed"] === "string" &&
    /^\d{1,20}$/.test(e["seed"]) &&
    typeof e["score"] === "number" &&
    Number.isInteger(e["score"]) &&
    e["score"] >= 0 &&
    typeof e["date"] === "number" &&
    Number.isFinite(e["date"]) &&
    typeof e["physics"] === "number" &&
    Number.isInteger(e["physics"]) &&
    typeof e["replay"] === "string" &&
    e["replay"].length <= MAX_STORED_REPLAY &&
    /^[A-Za-z0-9_-]*$/.test(e["replay"])
  );
}

/** Best first; ties go to the earlier run. */
function compare(a: ScoreEntry, b: ScoreEntry): number {
  return b.score - a.score || a.date - b.date;
}

export class ScoreBoard {
  constructor(
    private readonly store: KeyValueStore | null,
    private readonly physics: number,
  ) {}

  /** Every valid entry of this physics version, best first. Never throws. */
  all(): ScoreEntry[] {
    return this.stored().filter((e) => e.physics === this.physics);
  }

  /** Every valid stored entry, any physics version. */
  private stored(): ScoreEntry[] {
    let raw: string | null = null;
    try {
      raw = this.store?.getItem(SCORES_KEY) ?? null;
    } catch {
      return []; // storage disabled (privacy mode, sandboxed iframe)
    }
    if (raw === null) return [];
    try {
      const parsed: unknown = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed.filter(isEntry).sort(compare) : [];
    } catch {
      return [];
    }
  }

  top(config: Config, n = MAX_SCORES_PER_CONFIG): ScoreEntry[] {
    return this.all()
      .filter((e) => e.config === config)
      .slice(0, n);
  }

  best(config: Config): number | null {
    return this.top(config, 1)[0]?.score ?? null;
  }

  /**
   * Records a landed run. Returns its 1-based rank among this config's scores,
   * or null if it did not make the table (or could not be saved).
   */
  add(entry: ScoreEntry): number | null {
    if (!isEntry(entry) || entry.physics !== this.physics) return null;
    // Entries of other physics versions are dropped on the next save.
    const entries = [...this.all(), entry].sort(compare);
    const kept: ScoreEntry[] = [];
    const counts = new Map<Config, number>();
    for (const e of entries) {
      const n = counts.get(e.config) ?? 0;
      if (n < MAX_SCORES_PER_CONFIG) {
        kept.push(e);
        counts.set(e.config, n + 1);
      }
    }
    if (!kept.includes(entry)) return null;
    try {
      this.store?.setItem(SCORES_KEY, JSON.stringify(kept));
    } catch {
      return null; // quota exceeded or storage disabled
    }
    if (this.store === null) return null;
    return kept.filter((e) => e.config === entry.config).indexOf(entry) + 1;
  }
}

/** window.localStorage, or null where it is unavailable. */
export function browserStore(): KeyValueStore | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}
