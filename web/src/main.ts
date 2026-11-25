// Icarus web front-end: main menu, manual landing, replays and local scores.

import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";
import "@fontsource/jetbrains-mono/700.css";
import "@fontsource/jetbrains-mono/800.css";

import { Sound } from "./audio";
import { KeyboardControls, Q_CENTER, minRunningLevel } from "./controls";
import { type Check, checkTable, limitStats, outcomeAdvice, outcomeTitle, type Stat, statGrid, touchdownChecks } from "./debrief";
import { Hud } from "./hud";
import { FixedLoop } from "./loop";
import { MainMenu } from "./menu";
import { Renderer } from "./render";
import { ScoreBoard, browserStore } from "./scores";
import { replayFromHash, replayUrl, scenarioUrl } from "./share";
import {
  Autopilot,
  CONFIGS,
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
import { TUTORIAL_SEED, TutorialScript } from "./tutorial";
import { TutorialView } from "./tutorial-view";

type Phase = "menu" | "ready" | "running" | "paused" | "ended";
/** Who flies: the player, a recorded replay, the autopilot, or the player guided by the tutorial. */
type Mode = "fly" | "replay" | "auto" | "tutorial";
/** A menu entry: a level, or the tutorial (level 0). */
type LevelId = Config | "tutorial";

const VERSION = "v0.2.0";
const U64_MAX = (1n << 64n) - 1n;
const REPLAY_SPEEDS = [1, 2, 4] as const;
/** Outcome codes that end in an explosion (crashes), as opposed to landing or timing out. */
const CRASH_CODES = new Set([2, 3, 4, 5, 6, 7]);

const MISSION: Record<Config, { title: string; blurb: string }> = {
  training: {
    title: "Training",
    blurb: "Your first landing. The engine is running and can hover, the air is calm and the pad is wide.",
  },
  easy: {
    title: "Easy",
    blurb: "The engine is running and can hover, with 3 relights if you cut it.",
  },
  crosswind: {
    title: "Crosswind",
    blurb: "Strong wind and gusts push you off a narrower pad. Lean into the wind.",
  },
  realistic: {
    title: "Realistic",
    blurb:
      "Engine off and one ignition. Even at minimum throttle it out-pushes the stage's weight, so you cannot hover: light it at the right height and land in one burn.",
  },
  storm: {
    title: "Storm",
    blurb: "One-burn landing in a gale, with sensors that lag twice as much.",
  },
  engine_out: {
    title: "Engine out",
    blurb:
      "One-burn landing where something breaks on the way down: the gimbal jams (fight the spin with Q and E) or thrust drops (light earlier).",
  },
};

const LANDED_KEY = "icarus.landed.v1";
const TUTORIAL_KEY = "icarus.tutorial.v1";

/**
 * Levels landed at least once. Kept apart from the scores (which are tied to
 * the physics version) and also in memory, so it works without storage. Every
 * level can be played; this only marks progress and suggests the next one.
 */
class Progress {
  private readonly landed = new Set<Config>();

  constructor() {
    try {
      const parsed: unknown = JSON.parse(window.localStorage.getItem(LANDED_KEY) ?? "[]");
      if (Array.isArray(parsed)) for (const c of parsed) if (isConfig(c)) this.landed.add(c);
    } catch {
      // storage unavailable or corrupt: start from nothing
    }
  }

  has(config: Config): boolean {
    return this.landed.has(config);
  }

  add(config: Config): void {
    this.landed.add(config);
    try {
      window.localStorage.setItem(LANDED_KEY, JSON.stringify([...this.landed]));
    } catch {
      // keep the in-memory record
    }
  }

  private tutorial: boolean | null = null;

  /** The tutorial was finished at least once. */
  get tutorialDone(): boolean {
    if (this.tutorial === null) {
      try {
        this.tutorial = window.localStorage.getItem(TUTORIAL_KEY) === "1";
      } catch {
        this.tutorial = false;
      }
    }
    return this.tutorial;
  }

  finishTutorial(): void {
    this.tutorial = true;
    try {
      window.localStorage.setItem(TUTORIAL_KEY, "1");
    } catch {
      // keep the in-memory record
    }
  }
}

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

/** A card button: label, handler and an optional key shown beside the label. */
type Action = [label: string, handler: () => void, key?: string | undefined];

interface CardOptions {
  title: string;
  tone?: "success" | "failure";
  /** Large figure under the title (the score of a landing). */
  hero?: { value: string; label: string } | undefined;
  lead?: string | undefined;
  stats?: readonly Stat[];
  /** Touchdown checks (debrief): measured value against limit. */
  checks?: readonly Check[];
  actions: readonly Action[];
  extra?: HTMLElement | null;
  /** Keyboard hint at the bottom, e.g. "Space launches". */
  hint?: string | undefined;
}

class App {
  private mode: Mode = "fly";
  private phase: Phase = "menu";
  /** The live run (fly mode) or null. */
  private sim: Simulation | null = null;
  /** The replay being watched (replay mode) or null. */
  private player: ReplayPlayer | null = null;
  /** The autopilot flying the live run (auto mode) or null. */
  private autopilot: Autopilot | null = null;
  /** The lesson script (tutorial mode) or null. */
  private tutorial: TutorialScript | null = null;
  private readonly tutorialView = new TutorialView(document.body);
  /** Throttle level last written back to the keyboard state in tutorial mode. */
  private syncedThrottle = 0;
  /** The autopilot's last commands, for the HUD. */
  private autoInput: [number, number, number] = [0, Q_CENTER, Q_CENTER];
  private speedIndex = 0;
  /** Scenario to return to when leaving a replay. */
  private lastScenario: { config: Config; seed: bigint };
  private readonly controls: KeyboardControls;
  private readonly renderer: Renderer;
  private readonly hud: Hud;
  private readonly charts: TelemetryCharts;
  private readonly touch: TouchControls;
  private readonly menu: MainMenu<LevelId>;
  private readonly sound = new Sound();
  private readonly progress = new Progress();
  private readonly scores = new ScoreBoard(browserStore(), getPhysicsVersion());
  private readonly loop: FixedLoop;
  private readonly overlay = byId("overlay", HTMLDivElement);
  private readonly card = byId("overlay-card", HTMLDivElement);
  private readonly bestLabel = byId("best", HTMLSpanElement);
  private readonly missionLabel = byId("mission", HTMLSpanElement);
  private readonly modeLabel = byId("mode", HTMLSpanElement);
  private readonly pauseButton = byId("btn-pause", HTMLButtonElement);
  private readonly chartsButton = byId("btn-charts", HTMLButtonElement);
  private readonly soundButton = byId("btn-sound", HTMLButtonElement);
  private readonly replayBar = byId("replay-bar", HTMLElement);
  private readonly replayLabel = byId("replay-label", HTMLSpanElement);
  private readonly speedButton = byId("btn-speed", HTMLButtonElement);
  private lastRcs = Q_CENTER;
  /** Set after an unrecoverable error: the loop is stopped and only Reload works. */
  private failed = false;

  constructor(start: { config: Config; seed: bigint } | null, replay: string | null) {
    this.lastScenario = start ?? { config: "easy", seed: randomSeed() };
    this.controls = new KeyboardControls(0);
    this.renderer = new Renderer(byId("view", HTMLCanvasElement));
    this.hud = new Hud(byId("hud", HTMLElement));
    this.charts = new TelemetryCharts(byId("charts", HTMLElement));
    this.touch = new TouchControls(byId("touch", HTMLElement), () => this.controls.input);
    this.menu = new MainMenu<LevelId>(
      byId("menu", HTMLElement),
      (id) => {
        this.sound.unlock();
        if (id === "tutorial") this.startTutorial();
        else this.newRun(id, randomSeed());
      },
      () => this.toggleSound(),
      !this.sound.isMuted,
      VERSION,
    );
    this.loop = new FixedLoop(
      () => this.tick(),
      (alpha) => this.frame(alpha),
      (error) => this.fail(error),
    );
    this.bindUi();
    const coarse = window.matchMedia("(pointer: coarse)");
    this.setTouch(coarse.matches);
    coarse.addEventListener("change", (e) => this.setTouch(e.matches));
    // Charts start hidden: they distract from flying.
    this.setCharts(false);
    this.updateSoundButton();
    // A replay link plays at once; a challenge link opens its scenario;
    // otherwise the game starts at the main menu.
    if (replay !== null) this.watch(replay);
    else if (start !== null) this.newRun(start.config, start.seed);
    else this.showMenu();
    this.loop.start();
  }

  /** The run on screen, live or replayed, or null at the menu. */
  private get view(): FlightView | null {
    return this.mode === "replay" ? this.player : this.sim;
  }

  private bindUi(): void {
    const click = (id: string, handler: () => void): void => {
      byId(id, HTMLButtonElement).addEventListener("click", (e) => {
        (e.currentTarget as HTMLButtonElement).blur();
        handler();
      });
    };
    // Mid-flight the brand button pauses instead of discarding the run.
    click("btn-menu", () => (this.phase === "running" ? this.pause() : this.showMenu()));
    click("btn-pause", () => this.togglePause());
    click("btn-charts", () => this.setCharts(!this.charts.isVisible));
    click("btn-sound", () => this.toggleSound());
    click("btn-speed", () => this.cycleSpeed());
    click("btn-exit-replay", () => this.exitReplay());

    // Losing the page or the window pauses a running descent.
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") this.pause();
    });
    window.addEventListener("blur", () => this.pause());
    window.addEventListener("hashchange", () => {
      const replay = replayFromHash(window.location.hash);
      if (replay !== null) this.watch(replay);
    });
    // Browsers only allow audio after a user gesture.
    // Touch only grants audio on pointerup/click, so unlock there too.
    window.addEventListener("click", () => this.sound.unlock(), { capture: true });
    window.addEventListener("pointerdown", (e) => {
      this.sound.unlock();
      // A finger on the screen switches the on-screen controls on.
      if (e.pointerType === "touch") this.setTouch(true);
    });

    window.addEventListener("keydown", (e) => {
      if (this.failed || e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
      this.sound.unlock();
      if (this.phase === "menu") {
        if (e.code === "ArrowDown" || e.code === "ArrowUp") {
          this.menu.moveFocus(e.code === "ArrowDown" ? 1 : -1);
          e.preventDefault();
        }
        return;
      }
      if (targetsControl(e.target, e.code)) return;
      // The key a frozen lesson asks for resumes time (and flies, via the controls).
      // Other keys pressed during the freeze are dropped, so X cannot cut the engine on resume.
      if (this.phase === "running" && this.tutorial?.press(e.code) === true) this.controls.input.clearPending();
      switch (e.code) {
        case "Space":
        case "Enter":
          if (this.phase === "ready" || this.phase === "paused") this.run();
          else if (this.phase === "ended") this.card.querySelector<HTMLButtonElement>(".card-actions .primary")?.click();
          break;
        case "Escape":
          if (this.phase === "running" || this.phase === "paused") this.togglePause();
          else if (this.phase === "ready") this.showMenu();
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

  private toggleSound(): boolean {
    this.sound.unlock();
    this.sound.setMuted(!this.sound.isMuted);
    this.updateSoundButton();
    return !this.sound.isMuted;
  }

  private updateSoundButton(): void {
    const on = !this.sound.isMuted;
    this.soundButton.textContent = on ? "Sound" : "Muted";
    this.soundButton.setAttribute("aria-pressed", String(on));
  }

  private disposeFlights(): void {
    this.sim?.dispose();
    this.sim = null;
    this.player?.dispose();
    this.player = null;
    this.autopilot?.dispose();
    this.autopilot = null;
    this.tutorial = null;
    this.tutorialView.hide();
  }

  private setPhase(phase: Phase): void {
    this.phase = phase;
    document.body.dataset["phase"] = phase;
    if (phase !== "running") this.sound.silence();
  }

  private showMenu(): void {
    this.disposeFlights();
    this.mode = "fly";
    this.setPhase("menu");
    this.hideOverlay();
    this.updateChrome();
    window.history.replaceState(null, "", new URL(window.location.pathname, window.location.href));
    const tutorialDone = this.progress.tutorialDone;
    this.menu.show([
      { id: "tutorial", title: "Tutorial", status: tutorialDone ? "Done" : null, done: tutorialDone },
      ...CONFIGS.map((config) => {
        const best = this.scores.best(config);
        const landed = best !== null || this.progress.has(config);
        return {
          id: config,
          title: MISSION[config].title,
          status: best !== null ? `Best ${best.toLocaleString("en-US")}` : landed ? "Landed" : null,
          done: landed,
        };
      }),
    ]);
  }

  /** Retry: the same scenario again, straight into the flight (no briefing). */
  private restart(): void {
    if (this.mode === "replay") this.replayAgain();
    else if (this.mode === "auto") this.watchAutopilot();
    else if (this.mode === "tutorial") this.startTutorial(true);
    else this.newRun(this.lastScenario.config, this.lastScenario.seed, false, true);
  }

  /** The tutorial: a Training flight with three guided lessons. */
  private startTutorial(launch = false): void {
    this.newRun("training", TUTORIAL_SEED, false, launch, true);
  }

  /** A new scenario of the current level, straight into the flight. */
  private newScenario(): void {
    this.newRun(this.lastScenario.config, randomSeed(), false, true);
  }

  /**
   * The autopilot flies a fresh scenario of the current level. A new seed each
   * time, since the same seed would replay exactly the same flight.
   */
  private watchAutopilot(): void {
    this.newRun(this.lastScenario.config, randomSeed(), true);
  }

  /**
   * A new run of `config` with `seed`, flown by the player or (auto) the
   * autopilot. `launch` skips the briefing and starts flying at once;
   * `tutorial` adds the guided lessons.
   */
  private newRun(config: Config, seed: bigint, auto = false, launch = false, tutorial = false): void {
    this.menu.hide();
    this.disposeFlights();
    const sim = new Simulation(config, seed);
    // Built before anything is switched to auto mode, so a failure cannot
    // leave a run labelled AUTOPILOT that the keyboard actually flies.
    const autopilot = auto ? new Autopilot(config) : null;
    this.sim = sim;
    this.autopilot = autopilot;
    this.tutorial = tutorial && autopilot === null ? new TutorialScript() : null;
    this.mode = autopilot !== null ? "auto" : this.tutorial !== null ? "tutorial" : "fly";
    this.autoInput = [sim.info.initial_throttle_level, Q_CENTER, Q_CENTER];
    this.lastScenario = { config, seed };
    this.controls.input.reset(sim.info.initial_throttle_level, minRunningLevel(sim.info.throttle_cutoff));
    this.syncedThrottle = sim.info.initial_throttle_level;
    this.lastRcs = Q_CENTER;
    this.renderer.reset();
    this.charts.reset(sim.state);
    this.setPhase("ready");
    this.updateChrome();
    // A tutorial link would open plain Training, so the tutorial keeps a bare URL.
    const url = this.tutorial !== null ? new URL(window.location.pathname, window.location.href) : scenarioUrl(window.location.href, config, seed);
    window.history.replaceState(null, "", url);
    if (auto || launch) this.run();
    else this.showReady();
  }

  /** Starts watching a replay string. Returns false (with a message) if it is invalid. */
  private watch(encoded: string): boolean {
    let player: ReplayPlayer;
    try {
      player = new ReplayPlayer(encoded);
    } catch (error) {
      console.error(error);
      if ((this.mode !== "fly" && this.mode !== "tutorial") || this.sim === null) {
        this.newRun(this.lastScenario.config, this.lastScenario.seed);
      } else if (this.phase === "running") {
        this.pause();
      }
      this.showCard({
        title: "Replay unavailable",
        lead: "This replay link is damaged or was recorded with an older version of the physics.",
        actions: [
          [
            "Fly instead",
            () => {
              if (this.phase === "ready" || this.phase === "paused") this.run();
              else this.newRun(this.lastScenario.config, this.lastScenario.seed);
            },
          ],
          ["Main menu", () => this.showMenu()],
        ],
      });
      return false;
    }
    this.menu.hide();
    this.disposeFlights();
    this.player = player;
    this.mode = "replay";
    this.speedIndex = 0;
    this.lastScenario = { config: player.config, seed: player.seed };
    this.renderer.reset();
    this.charts.reset(player.state);
    this.setPhase("running");
    this.hideOverlay();
    this.updateChrome();
    window.history.replaceState(null, "", replayUrl(window.location.href, encoded));
    return true;
  }

  private replayAgain(): void {
    const encoded = replayFromHash(window.location.hash);
    if (encoded !== null) this.watch(encoded);
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
    this.missionLabel.textContent = this.mode === "tutorial" ? "Tutorial" : MISSION[config].title;
    this.missionLabel.title = `Scenario ${seed.toString()}`;
    this.modeLabel.textContent = replay ? "Replay" : this.mode === "auto" ? "Autopilot" : "";
    this.modeLabel.hidden = this.mode === "fly" || this.mode === "tutorial";
    const best = this.scores.best(config);
    this.bestLabel.textContent = best === null ? "—" : best.toLocaleString("en-US");
    this.pauseButton.textContent = this.phase === "paused" ? "Resume" : "Pause";
    this.replayBar.hidden = !replay || this.phase === "menu";
    this.speedButton.textContent = `${REPLAY_SPEEDS[this.speedIndex] ?? 1}×`;
    document.body.dataset["mode"] = this.mode;
  }

  private run(): void {
    if (this.failed) return;
    if (this.phase === "ready" && this.sim !== null) {
      // Presses made while reading the briefing do not count (held keys do).
      this.controls.input.reset(this.sim.info.initial_throttle_level);
      this.syncedThrottle = this.sim.info.initial_throttle_level;
    }
    if (this.phase !== "ready" && this.phase !== "paused") return;
    // Taps made while paused (X, short presses) do not leak into the run.
    if (this.phase === "paused") this.controls.input.clearPending();
    this.sound.unlock();
    this.setPhase("running");
    this.hideOverlay();
    this.updateChrome();
  }

  private pause(): void {
    if (this.failed || this.phase !== "running") return;
    this.setPhase("paused");
    this.controls.input.clearPending();
    this.updateChrome();
    const actions: Action[] = [["Resume", () => this.run(), "Esc"]];
    if (this.mode === "fly") {
      actions.push(["Retry", () => this.restart()], ["New scenario", () => this.newScenario()]);
    } else if (this.mode === "auto") {
      actions.push(["Watch again", () => this.watchAutopilot()], ["Fly it yourself", () => this.flyYourself()]);
    } else if (this.mode === "tutorial") {
      actions.push(["Restart tutorial", () => this.startTutorial(true)]);
    } else {
      actions.push(["Watch again", () => this.replayAgain()]);
    }
    actions.push(["Main menu", () => this.showMenu()]);
    this.showCard({ title: "Paused", actions });
  }

  private togglePause(): void {
    if (this.phase === "running") this.pause();
    else if (this.phase === "paused") this.run();
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
    let throttle: number;
    let gimbal: number;
    let rcs: number;
    if (this.mode === "auto" && this.autopilot !== null) {
      this.autoInput = this.autopilot.act(sim);
      [throttle, gimbal, rcs] = this.autoInput;
    } else if (this.mode === "tutorial" && this.tutorial !== null) {
      const commands = this.tutorialTick(this.tutorial, sim);
      if (commands === null) {
        // Frozen: the HUD shows the side thrusters idle, not the setup's last push.
        this.lastRcs = Q_CENTER;
        return;
      }
      [throttle, gimbal, rcs] = commands;
    } else {
      [throttle, gimbal, rcs] = this.controls.input.sample();
    }
    this.lastRcs = rcs;
    const code = sim.step(throttle, gimbal, rcs);
    this.charts.record(sim.state);
    if (code !== 0) this.end();
  }

  /**
   * One tutorial tick: the player's keys, except where a lesson setup pushes.
   * Returns null while time is frozen (nothing steps).
   */
  private tutorialTick(script: TutorialScript, sim: Simulation): readonly [number, number, number] | null {
    const input = this.controls.input;
    const lesson = script.lesson;
    if (script.frozen && lesson !== null) {
      // Touch controls send no key events: a held button or a raised throttle bar answers too.
      const held = lesson.codes.find((c) => input.isHeld(c));
      if (held !== undefined) script.press(held);
      else if (lesson.channel === "throttle" && input.throttleLevel > this.syncedThrottle) script.answer("throttle");
      if (script.frozen) return null;
      // Presses queued during the freeze do not leak into the first tick.
      input.clearPending();
    }
    const player = input.sample();
    const commands = script.command(sim.state, sim.info, player);
    // When a setup cuts the thrust, ↑ then starts from the level actually flown.
    if (commands[0] !== player[0]) input.setThrottle(commands[0]);
    this.syncedThrottle = input.throttleLevel;
    return commands;
  }

  private frame(alpha: number): void {
    const view = this.view;
    if (this.phase === "menu" || view === null) return;
    const frozen = this.tutorial?.frozen === true;
    this.renderer.draw(view.previousState, view.state, alpha, view.info, this.phase !== "running" || frozen);
    if (this.player !== null && this.mode === "replay") {
      const [throttle, , rcs] = this.player.lastInput;
      this.hud.update(view.state, view.info, throttle, rcs);
      const total = (this.player.totalTicks * view.info.dt).toFixed(1);
      this.replayLabel.textContent = `Replay ${view.state.time.toFixed(1)} / ${total} s`;
    } else {
      const level = this.mode === "auto" ? this.autoInput[0] : this.controls.input.throttleLevel;
      this.hud.update(view.state, view.info, level, this.lastRcs);
      if (this.mode === "fly" || this.mode === "tutorial") this.touch.sync(level);
    }
    if (this.phase === "running") {
      const s = view.state;
      // Frozen time is silent.
      if (frozen) this.sound.update(0, false);
      else this.sound.update(s.engine_on > 0.5 ? s.thrust_fraction : 0, Math.abs(s.rcs_torque) > 1);
    }
    this.updateTutorialView(frozen);
    this.charts.draw();
  }

  private updateTutorialView(frozen: boolean): void {
    const script = this.tutorial;
    const lesson = script?.lesson ?? null;
    if (script === null || lesson === null || this.phase !== "running") {
      this.tutorialView.hide();
      return;
    }
    const touch = document.body.classList.contains("touch-mode");
    const targets = lesson.highlight.map((key) => this.hud.element(key));
    if (touch) targets.push(this.touch.target(lesson.codes[0] ?? "") ?? undefined);
    this.tutorialView.update(
      lesson,
      frozen,
      targets.filter((t): t is HTMLElement => t !== undefined),
      this.renderer.rocketOnScreen,
      touch,
    );
  }

  private end(): void {
    const view = this.view;
    if (view === null) return;
    this.setPhase("ended");
    this.updateChrome();
    const s = view.state;
    const landed = view.outcome === 1;
    if (landed) this.sound.land();
    else if (CRASH_CODES.has(view.outcome)) this.sound.explode();
    const debrief = {
      title: outcomeTitle(view.outcomeName),
      tone: landed ? ("success" as const) : ("failure" as const),
      checks: touchdownChecks(view.outcomeName, s, view.info),
      lead: landed ? undefined : (outcomeAdvice(view.outcomeName, view.info) ?? undefined),
    };

    if (this.mode === "replay") {
      this.showCard({
        ...debrief,
        hero: landed ? { value: view.score.toLocaleString("en-US"), label: "points" } : undefined,
        actions: [
          ["Watch again", () => this.replayAgain()],
          ["Fly this scenario", () => this.exitReplay()],
          ["Main menu", () => this.showMenu()],
        ],
      });
      return;
    }

    if (this.mode === "tutorial") {
      this.tutorialView.hide();
      if (landed) {
        this.progress.finishTutorial();
        this.showCard({
          title: "Tutorial complete",
          tone: "success",
          lead: "You used the side thrusters, the gimbal and the throttle. Now land Training on your own.",
          actions: [
            ["Fly Training", () => this.newRun("training", randomSeed()), "Space"],
            ["Repeat tutorial", () => this.startTutorial(true)],
            ["Main menu", () => this.showMenu()],
          ],
        });
      } else {
        this.showCard({
          ...debrief,
          actions: [
            ["Try again", () => this.startTutorial(true), "Space"],
            ["Main menu", () => this.showMenu()],
          ],
        });
      }
      return;
    }

    const sim = this.sim;
    if (sim === null) return;
    const hero = landed ? { value: sim.score.toLocaleString("en-US"), label: "points" } : undefined;
    if (this.mode === "auto") {
      // The autopilot's landings never count for the player's scores or progress.
      this.showCard({
        ...debrief,
        hero,
        actions: [
          ["Watch again", () => this.watchAutopilot()],
          ["Fly it yourself", () => this.flyYourself()],
          ["Main menu", () => this.showMenu()],
        ],
      });
      return;
    }

    if (landed) {
      this.progress.add(sim.config);
      let replay: string | null = null;
      try {
        replay = sim.replay();
      } catch (error) {
        console.error(error);
      }
      if (replay !== null) {
        const rank = this.scores.add({
          config: sim.config,
          seed: sim.seed.toString(),
          score: sim.score,
          date: Date.now(),
          replay,
          physics: getPhysicsVersion(),
        });
        if (hero !== undefined && rank === 1) hero.label = "points, a new personal best";
        else if (hero !== undefined && rank !== null) hero.label = `points, your #${rank} on this level`;
        this.updateChrome();
      }
    }
    this.showCard({
      ...debrief,
      hero,
      actions: [
        ["Retry", () => this.restart(), "Space"],
        ["New scenario", () => this.newScenario()],
        ["Main menu", () => this.showMenu()],
      ],
    });
  }

  /** The same scenario, flown by the player. */
  private flyYourself(): void {
    this.newRun(this.lastScenario.config, this.lastScenario.seed);
  }

  private showReady(): void {
    const sim = this.sim;
    if (sim === null) return;
    const touch = document.body.classList.contains("touch-mode");
    if (this.mode === "tutorial") {
      this.showCard({
        title: "Tutorial",
        lead: "Three short lessons on the controls. Time stops whenever you need to act and shows you what to press.",
        actions: [
          ["Start", () => this.run(), touch ? undefined : "Space"],
          ["Main menu", () => this.showMenu()],
        ],
      });
      return;
    }
    const s = sim.state;
    this.showCard({
      title: MISSION[sim.config].title,
      lead: MISSION[sim.config].blurb,
      stats: limitStats(sim.info, s.ignitions_left, s.engine_on > 0.5),
      actions: [
        ["Launch", () => this.run(), touch ? undefined : "Space"],
        ["Watch autopilot", () => this.watchAutopilot()],
        ["Main menu", () => this.showMenu()],
      ],
      hint: touch ? "Drag the throttle bar on the right and hold the arrows to steer." : undefined,
    });
  }

  private showCard(o: CardOptions): void {
    const h = document.createElement("h2");
    h.id = "overlay-title";
    h.textContent = o.title;
    const children: HTMLElement[] = [h];
    if (o.hero) {
      const hero = document.createElement("p");
      hero.className = "card-hero";
      const value = document.createElement("span");
      value.className = "card-hero-value";
      value.textContent = o.hero.value;
      const label = document.createElement("span");
      label.className = "card-hero-label";
      label.textContent = o.hero.label;
      hero.append(value, label);
      children.push(hero);
    }
    if (o.lead) {
      const p = document.createElement("p");
      p.className = "card-lead";
      p.textContent = o.lead;
      children.push(p);
    }
    if (o.stats && o.stats.length > 0) children.push(statGrid(o.stats));
    if (o.checks && o.checks.length > 0) children.push(checkTable(o.checks));
    if (o.extra) children.push(o.extra);
    const row = document.createElement("div");
    row.className = "card-actions";
    o.actions.forEach(([label, handler, key], i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = i === 0 ? "primary" : "secondary";
      b.textContent = label;
      if (key !== undefined) {
        const k = document.createElement("kbd");
        k.textContent = key;
        k.setAttribute("aria-hidden", "true");
        b.append(k);
      }
      b.addEventListener("click", () => handler());
      row.append(b);
    });
    children.push(row);
    if (o.hint) {
      const p = document.createElement("p");
      p.className = "card-hint";
      p.textContent = o.hint;
      children.push(p);
    }
    this.card.replaceChildren(...children);
    if (o.tone) this.card.dataset["tone"] = o.tone;
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
    this.failed = true;
    this.loop.stop();
    this.sound.silence();
    this.showCard({ title: "Simulation error", lead: String(error), actions: [["Reload", () => window.location.reload()]] });
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
  const seed = parseSeed(params.get("seed"));
  const start = isConfig(configParam) && seed !== null ? { config: configParam, seed } : null;
  new App(start, replayFromHash(window.location.hash));
}

void main();
