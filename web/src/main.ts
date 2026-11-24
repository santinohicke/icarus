// Icarus web front-end: manual landing (stage 1).

import { KeyboardControls, Q_CENTER, minRunningLevel } from "./controls";
import { Hud } from "./hud";
import { FixedLoop } from "./loop";
import { Renderer } from "./render";
import { type Config, Simulation, isConfig, loadWasm } from "./sim";

type Phase = "ready" | "running" | "paused" | "ended";

const U64_MAX = (1n << 64n) - 1n;

const CONFIG_BLURB: Record<Config, string> = {
  easy: "Engine running, 3 relights. The engine can throttle below the stage's weight, so you can hover.",
  realistic:
    "Engine off, one ignition. Even at minimum throttle the engine is stronger than the stage's weight: you cannot hover. Light it at the right altitude and land in a single burn.",
};

const OUTCOME_TEXT: Record<string, string> = {
  landed: "Touchdown confirmed",
  crashed_tipped_over: "Tipped over",
  crashed_off_pad: "Missed the pad",
  crashed_vertical_speed: "Vertical speed too high at touchdown",
  crashed_horizontal_speed: "Sideways speed too high at touchdown",
  crashed_tilt: "Too tilted at touchdown",
  crashed_angular_rate: "Rotating too fast at touchdown",
  out_of_propellant: "Out of propellant",
  out_of_bounds: "Left the landing zone",
  time_limit: "Ran out of time",
  diverged: "Simulation error",
};

function parseSeed(text: string | null): bigint | null {
  if (text === null || !/^\d{1,20}$/.test(text)) return null;
  const v = BigInt(text);
  return v <= U64_MAX ? v : null;
}

function randomSeed(): bigint {
  const words = crypto.getRandomValues(new Uint32Array(2));
  return (BigInt(words[0] ?? 0) << 32n) | BigInt(words[1] ?? 0);
}

function byId<T extends HTMLElement>(id: string, type: new () => T): T {
  const e = document.getElementById(id);
  if (!(e instanceof type)) throw new Error(`missing or unexpected element #${id}`);
  return e;
}

/** Keyboard shortcuts must not hijack keys meant for a focused control. */
function targetsControl(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLButtonElement ||
    target instanceof HTMLAnchorElement ||
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement
  );
}

type Action = [label: string, handler: (button: HTMLButtonElement) => void];

class App {
  private sim: Simulation;
  private phase: Phase = "ready";
  private readonly controls: KeyboardControls;
  private readonly renderer: Renderer;
  private readonly hud: Hud;
  private readonly loop: FixedLoop;
  private readonly overlay = byId("overlay", HTMLDivElement);
  private readonly card = byId("overlay-card", HTMLDivElement);
  private readonly seedLabel = byId("seed", HTMLSpanElement);
  private readonly pauseButton = byId("btn-pause", HTMLButtonElement);
  private lastRcs = Q_CENTER;

  constructor(config: Config, seed: bigint) {
    this.sim = new Simulation(config, seed);
    this.controls = new KeyboardControls(
      this.sim.info.initial_throttle_level,
      minRunningLevel(this.sim.info.throttle_cutoff),
    );
    this.renderer = new Renderer(byId("view", HTMLCanvasElement));
    this.hud = new Hud(byId("hud", HTMLElement));
    this.loop = new FixedLoop(
      () => this.tick(),
      (alpha) => this.frame(alpha),
      (error) => this.fail(error),
    );
    this.bindUi();
    this.newRun(config, seed);
    this.loop.start();
  }

  private bindUi(): void {
    for (const button of document.querySelectorAll<HTMLButtonElement>("[data-config]")) {
      button.addEventListener("click", () => {
        button.blur();
        const c = button.dataset["config"];
        if (isConfig(c)) this.newRun(c, this.sim.seed);
      });
    }
    byId("btn-restart", HTMLButtonElement).addEventListener("click", (e) => {
      (e.currentTarget as HTMLButtonElement).blur();
      this.newRun(this.sim.config, this.sim.seed);
    });
    byId("btn-new", HTMLButtonElement).addEventListener("click", (e) => {
      (e.currentTarget as HTMLButtonElement).blur();
      this.newRun(this.sim.config, randomSeed());
    });
    this.pauseButton.addEventListener("click", () => {
      this.pauseButton.blur();
      this.togglePause();
    });

    // Losing the page or the window pauses a running descent.
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") this.pause();
    });
    window.addEventListener("blur", () => this.pause());

    window.addEventListener("keydown", (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey || e.repeat || targetsControl(e.target)) return;
      const flying = this.phase === "running";
      switch (e.code) {
        case "Space":
        case "Enter":
          if (this.phase === "ready" || this.phase === "paused") this.run();
          else if (this.phase === "ended") this.newRun(this.sim.config, this.sim.seed);
          break;
        case "KeyP":
        case "Escape":
          this.togglePause();
          break;
        // Restarts only outside a live descent (R sits next to the RCS keys).
        case "KeyR":
          if (!flying) this.newRun(this.sim.config, this.sim.seed);
          break;
        case "KeyN":
          if (!flying) this.newRun(this.sim.config, randomSeed());
          break;
        case "Digit1":
          if (!flying) this.newRun("easy", this.sim.seed);
          break;
        case "Digit2":
          if (!flying) this.newRun("realistic", this.sim.seed);
          break;
        default:
          return;
      }
      e.preventDefault();
    });
  }

  private newRun(config: Config, seed: bigint): void {
    this.sim.dispose();
    this.sim = new Simulation(config, seed);
    this.controls.input.reset(
      this.sim.info.initial_throttle_level,
      minRunningLevel(this.sim.info.throttle_cutoff),
    );
    this.lastRcs = Q_CENTER;
    this.renderer.reset();
    this.phase = "ready";
    this.pauseButton.textContent = "Pause";
    this.seedLabel.textContent = seed.toString();
    for (const b of document.querySelectorAll<HTMLButtonElement>("[data-config]")) {
      b.setAttribute("aria-pressed", String(b.dataset["config"] === config));
    }
    const url = new URL(window.location.href);
    url.searchParams.set("config", config);
    url.searchParams.set("seed", seed.toString());
    window.history.replaceState(null, "", url);
    this.showReady();
  }

  private run(): void {
    if (this.phase === "ready") {
      // Presses made while reading the briefing do not count (held keys do).
      this.controls.input.reset(this.sim.info.initial_throttle_level);
    }
    this.phase = "running";
    this.hideOverlay();
    this.pauseButton.textContent = "Pause";
  }

  private pause(): void {
    if (this.phase !== "running") return;
    this.phase = "paused";
    this.controls.input.clearPending();
    this.pauseButton.textContent = "Resume";
    this.showCard("PAUSED", [], [["Resume (Space)", () => this.run()]], "Press P or Space to resume.");
  }

  private togglePause(): void {
    if (this.phase === "running") this.pause();
    else if (this.phase === "paused") this.run();
  }

  private tick(): void {
    if (this.phase !== "running") return;
    const [throttle, gimbal, rcs] = this.controls.input.sample();
    this.lastRcs = rcs;
    const code = this.sim.step(throttle, gimbal, rcs);
    if (code !== 0) this.end();
  }

  private frame(alpha: number): void {
    this.renderer.draw(this.sim.previousState, this.sim.state, alpha, this.sim.info, this.phase !== "running");
    this.hud.update(this.sim.state, this.sim.info, this.controls.input.throttleLevel, this.lastRcs);
  }

  private end(): void {
    this.phase = "ended";
    const s = this.sim.state;
    const landed = this.sim.outcome === 1;
    const title = landed ? `LANDED · ${this.sim.score} pts` : "MISSION FAILED";
    const lines = [
      OUTCOME_TEXT[this.sim.outcomeName] ?? this.sim.outcomeName,
      `Leg contact: vertical ${s.contact_vy.toFixed(2)} m/s · sideways ${s.contact_vx.toFixed(2)} m/s · ${Math.round(s.propellant)} kg left`,
    ];
    this.showCard(
      title,
      lines,
      [
        ["Retry (R)", () => this.newRun(this.sim.config, this.sim.seed)],
        ["New scenario (N)", () => this.newRun(this.sim.config, randomSeed())],
        ["Copy replay", (button) => void this.copyReplay(button)],
      ],
      undefined,
      landed ? "success" : "failure",
    );
  }

  private async copyReplay(button: HTMLButtonElement): Promise<void> {
    let replay: string;
    try {
      replay = this.sim.replay();
    } catch (error) {
      button.textContent = "Replay unavailable";
      console.error(error);
      return;
    }
    try {
      await navigator.clipboard.writeText(replay);
      button.textContent = "Copied";
    } catch {
      // No clipboard access (insecure context, denied permission): show the
      // text so it can be copied by hand.
      const area = document.createElement("textarea");
      area.readOnly = true;
      area.className = "replay-text";
      area.value = replay;
      area.setAttribute("aria-label", "Replay string");
      this.card.append(area);
      area.select();
      button.textContent = "Select and copy";
    }
  }

  private showReady(): void {
    const touch = window.matchMedia("(pointer: coarse)").matches;
    const info = this.sim.info;
    const lines = [
      CONFIG_BLURB[this.sim.config],
      `Land on the pad with the legs moving slower than ${info.max_vertical_speed} m/s down and ${info.max_horizontal_speed} m/s sideways, tilt under ${Math.round((info.max_tilt * 180) / Math.PI)}°.`,
    ];
    if (touch) lines.push("A keyboard is needed for now. Touch controls are coming in the next version.");
    this.showCard(
      this.sim.config === "easy" ? "EASY · HOVER-CAPABLE" : "REALISTIC · HOVERSLAM",
      lines,
      [["Launch (Space)", () => this.run()]],
    );
  }

  private showCard(title: string, lines: string[], actions: Action[], hint?: string, tone?: "success" | "failure"): void {
    const h = document.createElement("h2");
    h.id = "overlay-title";
    h.textContent = title;
    const children: HTMLElement[] = [h];
    for (const line of lines) {
      const p = document.createElement("p");
      p.textContent = line;
      children.push(p);
    }
    const row = document.createElement("div");
    row.className = "card-actions";
    for (const [label, handler] of actions) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = label;
      b.addEventListener("click", () => handler(b));
      row.append(b);
    }
    children.push(row);
    if (hint) {
      const p = document.createElement("p");
      p.className = "hint";
      p.textContent = hint;
      children.push(p);
    }
    this.card.replaceChildren(...children);
    if (tone) this.card.dataset["tone"] = tone;
    else delete this.card.dataset["tone"];
    this.overlay.hidden = false;
    row.querySelector("button")?.focus({ preventScroll: true });
  }

  private hideOverlay(): void {
    this.overlay.hidden = true;
    if (document.activeElement instanceof HTMLElement && this.overlay.contains(document.activeElement)) {
      document.activeElement.blur();
    }
  }

  private fail(error: unknown): void {
    console.error(error);
    this.showCard("SIMULATION ERROR", [String(error)], [["Reload", () => window.location.reload()]]);
  }
}

async function main(): Promise<void> {
  try {
    await loadWasm();
  } catch (err) {
    document.body.textContent = `Failed to load the simulation: ${String(err)}`;
    return;
  }
  const params = new URLSearchParams(window.location.search);
  const configParam = params.get("config");
  const config: Config = isConfig(configParam) ? configParam : "easy";
  const seed = parseSeed(params.get("seed")) ?? randomSeed();
  new App(config, seed);
}

void main();
