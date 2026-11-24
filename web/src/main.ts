// Icarus web front-end: manual landing, replays, failures and local scores.

import { KeyboardControls, Q_CENTER, minRunningLevel } from "./controls";
import { Hud } from "./hud";
import { FixedLoop } from "./loop";
import { Renderer } from "./render";
import { ScoreBoard, type ScoreEntry, browserStore } from "./scores";
import { replayFromHash, replayUrl, scenarioUrl } from "./share";
import {
  type Config,
  type FlightView,
  ReplayPlayer,
  Simulation,
  getPhysicsVersion,
  isConfig,
  loadWasm,
} from "./sim";
import { TelemetryCharts } from "./telemetry";
import { TouchControls } from "./touch";

type Phase = "ready" | "running" | "paused" | "ended";
type Mode = "fly" | "replay";

const U64_MAX = (1n << 64n) - 1n;
const DEG = Math.PI / 180;
const REPLAY_SPEEDS = [1, 2, 4] as const;

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

/**
 * Keyboard shortcuts must not hijack keys meant for a focused control: text
 * fields keep every key; buttons and links keep only the keys that press them.
 */
function targetsControl(target: EventTarget | null, code: string): boolean {
  if (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement
  ) {
    return true;
  }
  const pressKey = code === "Space" || code === "Enter" || code === "NumpadEnter";
  return pressKey && (target instanceof HTMLButtonElement || target instanceof HTMLAnchorElement);
}

function formatDate(ms: number): string {
  return new Date(ms).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

type Action = [label: string, handler: (button: HTMLButtonElement) => void];

class App {
  private mode: Mode = "fly";
  private phase: Phase = "ready";
  /** The live run (fly mode) or null. */
  private sim: Simulation | null = null;
  /** The replay being watched (replay mode) or null. */
  private player: ReplayPlayer | null = null;
  private speedIndex = 0;
  /** Scenario to return to when leaving a replay. */
  private lastScenario: { config: Config; seed: bigint };
  private readonly controls: KeyboardControls;
  private readonly renderer: Renderer;
  private readonly hud: Hud;
  private readonly charts: TelemetryCharts;
  private readonly touch: TouchControls;
  private readonly scores = new ScoreBoard(browserStore(), getPhysicsVersion());
  private readonly loop: FixedLoop;
  private readonly overlay = byId("overlay", HTMLDivElement);
  private readonly card = byId("overlay-card", HTMLDivElement);
  private readonly seedLabel = byId("seed", HTMLSpanElement);
  private readonly bestLabel = byId("best", HTMLSpanElement);
  private readonly pauseButton = byId("btn-pause", HTMLButtonElement);
  private readonly chartsButton = byId("btn-charts", HTMLButtonElement);
  private readonly faultPanel = byId("faults", HTMLElement);
  private readonly replayBar = byId("replay-bar", HTMLElement);
  private readonly replayLabel = byId("replay-label", HTMLSpanElement);
  private readonly speedButton = byId("btn-speed", HTMLButtonElement);
  private lastRcs = Q_CENTER;

  constructor(config: Config, seed: bigint, replay: string | null) {
    this.lastScenario = { config, seed };
    this.controls = new KeyboardControls(0);
    this.renderer = new Renderer(byId("view", HTMLCanvasElement));
    this.hud = new Hud(byId("hud", HTMLElement));
    this.charts = new TelemetryCharts(byId("charts", HTMLElement));
    this.touch = new TouchControls(byId("touch", HTMLElement), () => this.controls.input);
    this.loop = new FixedLoop(
      () => this.tick(),
      (alpha) => this.frame(alpha),
      (error) => this.fail(error),
    );
    this.bindUi();
    const coarse = window.matchMedia("(pointer: coarse)");
    this.setTouch(coarse.matches);
    coarse.addEventListener("change", (e) => this.setTouch(e.matches));
    // Charts start hidden (they distract from flying); T or the button opens them.
    this.setCharts(false);
    // On failure, watch() loads the scenario itself and shows the error.
    if (replay === null) this.newRun(config, seed);
    else this.watch(replay);
    this.loop.start();
  }

  /** The run on screen, live or replayed. */
  private get view(): FlightView {
    const v = this.mode === "replay" ? this.player : this.sim;
    if (v === null) throw new Error("no flight loaded");
    return v;
  }

  private bindUi(): void {
    const click = (id: string, handler: () => void): void => {
      byId(id, HTMLButtonElement).addEventListener("click", (e) => {
        (e.currentTarget as HTMLButtonElement).blur();
        handler();
      });
    };
    for (const button of document.querySelectorAll<HTMLButtonElement>("[data-config]")) {
      button.addEventListener("click", () => {
        button.blur();
        const c = button.dataset["config"];
        if (isConfig(c)) this.newRun(c, this.lastScenario.seed);
      });
    }
    click("btn-restart", () => this.newRun(this.lastScenario.config, this.lastScenario.seed));
    click("btn-new", () => this.newRun(this.lastScenario.config, randomSeed()));
    click("btn-pause", () => this.togglePause());
    click("btn-charts", () => this.setCharts(!this.charts.isVisible));
    click("btn-speed", () => this.cycleSpeed());
    click("btn-exit-replay", () => this.exitReplay());

    for (const button of this.faultPanel.querySelectorAll<HTMLButtonElement>("[data-fault]")) {
      // Never take keyboard focus away from the game keys.
      button.tabIndex = -1;
      button.addEventListener("mousedown", (e) => e.preventDefault());
      button.addEventListener("click", () => {
        button.blur();
        this.injectFault(button.dataset["fault"] ?? "");
      });
    }

    // Losing the page or the window pauses a running descent.
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") this.pause();
    });
    window.addEventListener("blur", () => this.pause());
    window.addEventListener("hashchange", () => {
      const replay = replayFromHash(window.location.hash);
      if (replay !== null) this.watch(replay);
    });
    // A finger on the screen switches the on-screen controls on.
    window.addEventListener("pointerdown", (e) => {
      if (e.pointerType === "touch") this.setTouch(true);
    });

    window.addEventListener("keydown", (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey || e.repeat || targetsControl(e.target, e.code)) return;
      const flying = this.phase === "running" && this.mode === "fly";
      switch (e.code) {
        case "Space":
        case "Enter":
          if (this.phase === "ready" || this.phase === "paused") this.run();
          else if (this.phase === "ended" && this.mode === "fly") this.restart();
          else if (this.phase === "ended") this.replayAgain();
          break;
        case "KeyP":
        case "Escape":
          this.togglePause();
          break;
        case "KeyT":
          this.setCharts(!this.charts.isVisible);
          break;
        // Restarts only outside a live descent (R sits next to the RCS keys).
        case "KeyR":
          if (!flying) this.restart();
          break;
        case "KeyN":
          if (!flying) this.newRun(this.lastScenario.config, randomSeed());
          break;
        case "Digit1":
          if (!flying) this.newRun("easy", this.lastScenario.seed);
          break;
        case "Digit2":
          if (!flying) this.newRun("realistic", this.lastScenario.seed);
          break;
        case "Digit7":
          if (flying) this.injectFault("gimbal");
          break;
        case "Digit8":
          if (flying) this.injectFault("thrust");
          break;
        case "Digit9":
          if (flying) this.injectFault("gust-left");
          break;
        case "Digit0":
          if (flying) this.injectFault("gust-right");
          break;
        default:
          return;
      }
      e.preventDefault();
    });
  }

  private setTouch(on: boolean): void {
    document.body.classList.toggle("touch-mode", on);
  }

  private setCharts(on: boolean): void {
    this.charts.setVisible(on);
    this.chartsButton.setAttribute("aria-pressed", String(on));
  }

  private disposeFlights(): void {
    this.sim?.dispose();
    this.sim = null;
    this.player?.dispose();
    this.player = null;
  }

  private restart(): void {
    if (this.mode === "replay") this.replayAgain();
    else this.newRun(this.lastScenario.config, this.lastScenario.seed);
  }

  private newRun(config: Config, seed: bigint): void {
    this.disposeFlights();
    const sim = new Simulation(config, seed);
    this.sim = sim;
    this.mode = "fly";
    this.lastScenario = { config, seed };
    this.controls.input.reset(sim.info.initial_throttle_level, minRunningLevel(sim.info.throttle_cutoff));
    this.lastRcs = Q_CENTER;
    this.renderer.reset();
    this.charts.reset(sim.state);
    this.phase = "ready";
    this.updateChrome();
    const url = new URL(scenarioUrl(window.location.href, config, seed));
    window.history.replaceState(null, "", url);
    this.showReady();
  }

  /** Starts watching a replay string. Returns false (with a message) if it is invalid. */
  private watch(encoded: string): boolean {
    let player: ReplayPlayer;
    try {
      player = new ReplayPlayer(encoded);
    } catch (error) {
      console.error(error);
      if (this.mode === "replay" || this.sim === null) {
        this.newRun(this.lastScenario.config, this.lastScenario.seed);
      } else if (this.phase === "running") {
        this.pause();
      }
      this.showCard(
        "INVALID REPLAY",
        ["This replay link is damaged or was recorded with a different version of the physics."],
        [
          [
            "Fly instead",
            () => {
              if (this.phase === "ready" || this.phase === "paused") this.run();
              else this.newRun(this.lastScenario.config, this.lastScenario.seed);
            },
          ],
        ],
      );
      return false;
    }
    this.disposeFlights();
    this.player = player;
    this.mode = "replay";
    this.speedIndex = 0;
    this.lastScenario = { config: player.config, seed: player.seed };
    this.renderer.reset();
    this.charts.reset(player.state);
    this.phase = "running";
    this.hideOverlay();
    this.updateChrome();
    window.history.replaceState(null, "", replayUrl(window.location.href, encoded));
    return true;
  }

  private replayAgain(): void {
    const url = replayFromHash(window.location.hash);
    if (url !== null) this.watch(url);
  }

  private exitReplay(): void {
    this.newRun(this.lastScenario.config, this.lastScenario.seed);
  }

  private cycleSpeed(): void {
    this.speedIndex = (this.speedIndex + 1) % REPLAY_SPEEDS.length;
    this.updateChrome();
  }

  /** Top-bar state that depends on the mode and the scenario. */
  private updateChrome(): void {
    const { config, seed } = this.lastScenario;
    const replay = this.mode === "replay";
    this.seedLabel.textContent = seed.toString();
    for (const b of document.querySelectorAll<HTMLButtonElement>("[data-config]")) {
      b.setAttribute("aria-pressed", String(b.dataset["config"] === config));
    }
    const best = this.scores.best(config);
    this.bestLabel.textContent = best === null ? "—" : String(best);
    this.pauseButton.textContent = this.phase === "paused" ? "Resume" : "Pause";
    this.replayBar.hidden = !replay;
    this.speedButton.textContent = `${REPLAY_SPEEDS[this.speedIndex] ?? 1}×`;
    this.faultPanel.hidden = replay || this.phase !== "running";
    document.body.dataset["mode"] = this.mode;
  }

  private run(): void {
    if (this.phase === "ready" && this.sim !== null) {
      // Presses made while reading the briefing do not count (held keys do).
      this.controls.input.reset(this.sim.info.initial_throttle_level);
    }
    if (this.phase !== "ready" && this.phase !== "paused") return;
    // Taps made while paused (X, Z, short presses) do not leak into the run.
    if (this.phase === "paused") this.controls.input.clearPending();
    this.phase = "running";
    this.hideOverlay();
    this.updateChrome();
  }

  private pause(): void {
    if (this.phase !== "running") return;
    this.phase = "paused";
    this.controls.input.clearPending();
    this.updateChrome();
    this.showCard("PAUSED", [], [["Resume (Space)", () => this.run()]], "Press P or Space to resume.");
  }

  private togglePause(): void {
    if (this.phase === "running") this.pause();
    else if (this.phase === "paused") this.run();
  }

  private injectFault(kind: string): void {
    const sim = this.sim;
    if (sim === null || this.mode !== "fly" || this.phase !== "running") return;
    const s = sim.state;
    switch (kind) {
      case "gimbal": {
        // Jams the gimbal where it is, or at 1° if it is centred.
        const angle = Math.abs(s.gimbal) > 0.2 * DEG ? s.gimbal : 1 * DEG;
        sim.injectGimbalStuck(angle);
        break;
      }
      case "thrust":
        sim.injectThrustLoss(0.7 * s.thrust_factor);
        break;
      case "gust-left":
        sim.injectGust(-12, 3);
        break;
      case "gust-right":
        sim.injectGust(12, 3);
        break;
      default:
        return;
    }
  }

  private tick(): void {
    if (this.phase !== "running") return;
    if (this.mode === "replay") {
      const player = this.player;
      if (player === null) return;
      const steps = REPLAY_SPEEDS[this.speedIndex] ?? 1;
      for (let i = 0; i < steps && !player.done; i += 1) {
        player.step();
        this.charts.record(player.state);
      }
      if (player.done) this.end();
      return;
    }
    const sim = this.sim;
    if (sim === null) return;
    const [throttle, gimbal, rcs] = this.controls.input.sample();
    this.lastRcs = rcs;
    const code = sim.step(throttle, gimbal, rcs);
    this.charts.record(sim.state);
    if (code !== 0) this.end();
  }

  private frame(alpha: number): void {
    const view = this.view;
    this.renderer.draw(view.previousState, view.state, alpha, view.info, this.phase !== "running");
    if (this.player !== null && this.mode === "replay") {
      const [throttle, , rcs] = this.player.lastInput;
      this.hud.update(view.state, view.info, throttle, rcs);
      const total = (this.player.totalTicks * view.info.dt).toFixed(1);
      this.replayLabel.textContent = `REPLAY · T+${view.state.time.toFixed(1)} / ${total} s`;
    } else {
      const level = this.controls.input.throttleLevel;
      this.hud.update(view.state, view.info, level, this.lastRcs);
      this.touch.sync(level);
    }
    this.charts.draw();
  }

  private end(): void {
    this.phase = "ended";
    this.updateChrome();
    const view = this.view;
    const s = view.state;
    const landed = view.outcome === 1;
    const lines = [
      OUTCOME_TEXT[view.outcomeName] ?? (view.outcomeName || "Replay ended before touchdown"),
      `Leg contact: vertical ${s.contact_vy.toFixed(2)} m/s · sideways ${s.contact_vx.toFixed(2)} m/s · ${Math.round(s.propellant)} kg left`,
    ];

    if (this.mode === "replay") {
      this.showCard(
        landed ? `REPLAY · LANDED · ${view.score} pts` : view.outcome === 0 ? "REPLAY · ENDED" : "REPLAY · MISSION FAILED",
        lines,
        [
          ["Watch again (R)", () => this.replayAgain()],
          ["Fly this scenario", () => this.exitReplay()],
        ],
        undefined,
        landed ? "success" : "failure",
      );
      return;
    }

    const sim = this.sim;
    if (sim === null) return;
    let replay: string | null = null;
    try {
      replay = sim.replay();
    } catch (error) {
      console.error(error);
    }
    if (landed && replay !== null) {
      const rank = this.scores.add({
        config: sim.config,
        seed: sim.seed.toString(),
        score: sim.score,
        date: Date.now(),
        replay,
        physics: getPhysicsVersion(),
      });
      if (rank === 1) lines.push(`New personal best on ${sim.config}!`);
      else if (rank !== null) lines.push(`#${rank} on your ${sim.config} leaderboard.`);
      this.updateChrome();
    }
    const actions: Action[] = [
      ["Retry (R)", () => this.restart()],
      ["New scenario (N)", () => this.newRun(sim.config, randomSeed())],
    ];
    if (replay !== null) {
      const link = replayUrl(window.location.href, replay);
      actions.push(["Share replay link", (button) => void this.copy(button, link, "Replay link")]);
    }
    actions.push([
      "Challenge link",
      (button) => void this.copy(button, scenarioUrl(window.location.href, sim.config, sim.seed), "Scenario link"),
    ]);
    this.showCard(
      landed ? `LANDED · ${sim.score} pts` : "MISSION FAILED",
      lines,
      actions,
      undefined,
      landed ? "success" : "failure",
    );
  }

  private async copy(button: HTMLButtonElement, text: string, label: string): Promise<void> {
    try {
      await navigator.clipboard.writeText(text);
      if (button.isConnected) button.textContent = "Copied";
    } catch {
      // No clipboard access (insecure context, denied permission): show the
      // text so it can be copied by hand, in the card that was clicked.
      if (!button.isConnected) return;
      this.card.querySelector(".replay-text")?.remove();
      const area = document.createElement("textarea");
      area.readOnly = true;
      area.className = "replay-text";
      area.value = text;
      area.setAttribute("aria-label", label);
      this.card.append(area);
      area.select();
      button.textContent = "Select and copy";
    }
  }

  private scoreTable(config: Config): HTMLElement | null {
    const top = this.scores.top(config, 5);
    if (top.length === 0) return null;
    const box = document.createElement("div");
    box.className = "scores";
    const title = document.createElement("div");
    title.className = "scores-title";
    title.textContent = `YOUR BEST LANDINGS · ${config.toUpperCase()}`;
    const list = document.createElement("ol");
    top.forEach((entry: ScoreEntry) => {
      const li = document.createElement("li");
      const text = document.createElement("span");
      text.textContent = `${entry.score} pts · seed ${entry.seed} · ${formatDate(entry.date)}`;
      const watch = document.createElement("button");
      watch.type = "button";
      watch.textContent = "Watch";
      watch.setAttribute("aria-label", `Watch the ${entry.score}-point landing`);
      watch.addEventListener("click", () => this.watch(entry.replay));
      li.append(text, watch);
      list.append(li);
    });
    box.append(title, list);
    return box;
  }

  private showReady(): void {
    const sim = this.sim;
    if (sim === null) return;
    const info = sim.info;
    const touch = document.body.classList.contains("touch-mode");
    const lines = [
      CONFIG_BLURB[sim.config],
      `Land on the pad with the legs moving slower than ${info.max_vertical_speed} m/s down and ${info.max_horizontal_speed} m/s sideways, tilt under ${Math.round(info.max_tilt / DEG)}°.`,
    ];
    if (touch) lines.push("Drag the throttle bar on the right; hold the arrow buttons to steer.");
    this.showCard(
      sim.config === "easy" ? "EASY · HOVER-CAPABLE" : "REALISTIC · HOVERSLAM",
      lines,
      [["Launch (Space)", () => this.run()]],
      "Failures can be triggered mid-flight with the buttons on the right (keys 7–0).",
      undefined,
      this.scoreTable(sim.config),
    );
  }

  private showCard(
    title: string,
    lines: string[],
    actions: Action[],
    hint?: string,
    tone?: "success" | "failure",
    extra?: HTMLElement | null,
  ): void {
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
    if (extra) children.push(extra);
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
  new App(config, seed, replayFromHash(window.location.hash));
}

void main();
