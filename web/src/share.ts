// Shareable links. A replay travels in the URL hash (`#replay=...`), which is
// never sent to the server; a scenario travels in the query (`?config&seed`).

import type { Config } from "./sim";

const REPLAY_PARAM = "replay";
/** Same limit as the core decoder; longer strings are rejected before decoding. */
export const MAX_REPLAY_LENGTH = 256 * 1024;
const REPLAY_CHARS = /^[A-Za-z0-9_-]+$/;

/** The replay string in a URL hash, or null if there is none or it is malformed. */
export function replayFromHash(hash: string): string | null {
  const params = new URLSearchParams(hash.startsWith("#") ? hash.slice(1) : hash);
  const value = params.get(REPLAY_PARAM);
  if (value === null || value.length === 0 || value.length > MAX_REPLAY_LENGTH) return null;
  return REPLAY_CHARS.test(value) ? value : null;
}

/** A link that opens the replay of a run. */
export function replayUrl(base: string, replay: string): string {
  const url = new URL(base);
  url.search = "";
  url.hash = `${REPLAY_PARAM}=${replay}`;
  return url.toString();
}

/** A link that opens the same scenario, ready to fly. */
export function scenarioUrl(base: string, config: Config, seed: bigint): string {
  const url = new URL(base);
  url.hash = "";
  url.search = "";
  url.searchParams.set("config", config);
  url.searchParams.set("seed", seed.toString());
  return url.toString();
}
