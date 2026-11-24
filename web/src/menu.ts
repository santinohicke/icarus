// Main menu: the Earth seen from orbit with a small rocket circling it, and
// the mission list on top. All markup is built with DOM APIs, no innerHTML.

import { geoDistance, geoOrthographic, geoPath } from "d3-geo";
import type { GeometryObject, Topology } from "topojson-specification";
import { feature } from "topojson-client";
import landTopology from "world-atlas/land-110m.json";

import type { Config } from "./sim";

const SVG_NS = "http://www.w3.org/2000/svg";
const W = 680;
const H = 383;
/** Earth disc: centre below the bottom edge so only its limb shows. */
const EARTH = { cx: 500, cy: 780, r: 560 } as const;
/** Visible orbit ellipse (tilted). */
const ORBIT = { cx: 500, cy: 300, rx: 300, ry: 64, tilt: (-10 * Math.PI) / 180 } as const;

export interface MenuLevel {
  config: Config;
  title: string;
  best: number | null;
  /** Landed at least once (even if the score has since been reset). */
  landed: boolean;
}

function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number>,
  parent?: Element,
): SVGElementTagNameMap[K] {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  parent?.append(e);
  return e;
}

function html<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

function gradient(defs: Element, id: string, stops: [number, string, number?][], cx = 0.5, cy = 0.5, r = 0.5): void {
  const g = svg("radialGradient", { id, cx, cy, r }, defs);
  for (const [offset, color, opacity] of stops) {
    svg("stop", { offset, "stop-color": color, "stop-opacity": opacity ?? 1 }, g);
  }
}

/** Deterministic pseudo-random numbers, so the scene looks the same every visit. */
function lcg(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807) % 2147483647;
    return s / 2147483647;
  };
}

function buildScene(): { root: SVGSVGElement; rocket: SVGGElement } {
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "xMidYMid slice", "aria-hidden": "true" });
  root.classList.add("menu-scene");
  const defs = svg("defs", {}, root);
  gradient(
    defs,
    "m-ocean",
    [
      [0, "#2A5A8C"],
      [0.18, "#14304F"],
      [0.5, "#08162A"],
      [1, "#03070F"],
    ],
    0.62,
    0.04,
    0.75,
  );
  gradient(
    defs,
    "m-shade",
    [
      [0, "#000", 0],
      [0.16, "#000", 0.1],
      [0.32, "#000", 0.55],
      [0.6, "#000", 0.8],
    ],
    0.62,
    0.04,
    0.75,
  );
  gradient(defs, "m-glare", [
    [0, "#FFF6E0", 0.9],
    [0.25, "#FFD9A0", 0.35],
    [1, "#FFB070", 0],
  ]);
  gradient(defs, "m-atm", [
    [0.93, "#4E9BFF", 0],
    [0.965, "#5AA8FF", 0.45],
    [1, "#5AA8FF", 0],
  ]);
  const clip = svg("clipPath", { id: "m-earth" }, defs);
  svg("circle", { cx: EARTH.cx, cy: EARTH.cy, r: EARTH.r }, clip);

  const rnd = lcg(7);
  const stars = svg("g", {}, root);
  for (let i = 0; i < 110; i += 1) {
    svg(
      "circle",
      {
        cx: (rnd() * W).toFixed(1),
        cy: (rnd() * 300).toFixed(1),
        r: rnd() < 0.1 ? 1.3 : 0.7,
        fill: "#fff",
        opacity: (0.15 + rnd() * 0.6).toFixed(2),
      },
      stars,
    );
  }

  // The orbit's far half is drawn behind the Earth, the near half in front.
  const orbitPath = svg("ellipse", {
    cx: ORBIT.cx,
    cy: ORBIT.cy,
    rx: ORBIT.rx,
    ry: ORBIT.ry,
    fill: "none",
    stroke: "#3A4452",
    "stroke-width": 1,
    "stroke-dasharray": "3 5",
    transform: `rotate(${(ORBIT.tilt * 180) / Math.PI} ${ORBIT.cx} ${ORBIT.cy})`,
  });

  svg("circle", { cx: EARTH.cx, cy: EARTH.cy, r: EARTH.r + 40, fill: "url(#m-atm)" }, root);
  svg("circle", { cx: EARTH.cx, cy: EARTH.cy, r: EARTH.r, fill: "url(#m-ocean)" }, root);
  const surface = svg("g", { "clip-path": "url(#m-earth)" }, root);
  const land = svg("g", {}, surface);
  const clouds = svg("g", {}, surface);
  svg("circle", { cx: EARTH.cx, cy: EARTH.cy, r: EARTH.r, fill: "url(#m-shade)" }, surface);
  const lights = svg("g", {}, surface);
  svg(
    "circle",
    { cx: EARTH.cx, cy: EARTH.cy, r: EARTH.r + 1, fill: "none", stroke: "#7DB8FF", "stroke-width": 1.2, "stroke-opacity": 0.7 },
    root,
  );
  svg("circle", { cx: 618, cy: 233, r: 70, fill: "url(#m-glare)" }, root);
  svg("circle", { cx: 618, cy: 233, r: 2.5, fill: "#fff" }, root);
  root.append(orbitPath);

  // Continents, clouds and city lights from the bundled 110m world atlas.
  const projection = geoOrthographic().scale(EARTH.r).translate([EARTH.cx, EARTH.cy]).rotate([28, 58]).clipAngle(90);
  const path = geoPath(projection);
  const centre: [number, number] = [-28, -58];
  const topo = landTopology as unknown as Topology<{ land: GeometryObject }>;
  const landShape = feature(topo, topo.objects.land);
  const d = path(landShape) ?? "";
  svg("path", { d, fill: "#4A5E3E", stroke: "#6E8058", "stroke-width": 0.5, "stroke-opacity": 0.6 }, land);
  svg("path", { d, fill: "#8C7A55", opacity: 0.25 }, land);
  const visible = (lon: number, lat: number, margin = 0): boolean => geoDistance([lon, lat], centre) <= Math.PI / 2 - margin;
  for (let i = 0; i < 26; i += 1) {
    const lon = -75 + rnd() * 80;
    const lat = -15 + rnd() * 45;
    const p = projection([lon, lat]);
    if (!p || !visible(lon, lat)) continue;
    const [x, y] = p;
    svg(
      "ellipse",
      {
        cx: x.toFixed(1),
        cy: y.toFixed(1),
        rx: (10 + rnd() * 26).toFixed(1),
        ry: (1.5 + rnd() * 3).toFixed(1),
        fill: "#fff",
        opacity: (0.08 + rnd() * 0.14).toFixed(2),
        transform: `rotate(${(-8 + rnd() * 10).toFixed(1)} ${x.toFixed(1)} ${y.toFixed(1)})`,
      },
      clouds,
    );
  }
  const cities: [number, number, number][] = [
    [-46.6, -23.5, 22],
    [-43.2, -22.9, 16],
    [-47.9, -15.8, 8],
    [-38.5, -13, 8],
    [-34.9, -8, 8],
    [-48.5, -1.4, 6],
    [-60, -3.1, 5],
    [-66.9, 10.5, 10],
    [-74, 4.7, 10],
    [-80.2, 25.8, 14],
    [-82.4, 23.1, 8],
    [3.4, 6.5, 14],
    [-0.2, 5.6, 8],
    [-17.4, 14.7, 7],
    [-7.6, 33.6, 8],
    [-58.4, -34.6, 14],
    [-56.2, -34.9, 6],
  ];
  for (const [lon, lat, n] of cities) {
    const p = projection([lon, lat]);
    if (!p || !visible(lon, lat, 0.02)) continue;
    for (let i = 0; i < n; i += 1) {
      const a = rnd() * Math.PI * 2;
      const dist = rnd() ** 1.6 * 7;
      svg(
        "circle",
        {
          cx: (p[0] + Math.cos(a) * dist * 1.6).toFixed(1),
          cy: (p[1] + Math.sin(a) * dist * 0.5).toFixed(1),
          r: rnd() < 0.25 ? 1.1 : 0.6,
          fill: "#FFD58A",
          opacity: (0.45 + rnd() * 0.5).toFixed(2),
        },
        lights,
      );
    }
  }

  // The rocket: a tiny first stage with a flame, pointing along its orbit.
  const rocket = svg("g", {}, root);
  svg("path", { d: "M -1.6 3 L 1.6 3 L 1.6 -6 L 0 -9 L -1.6 -6 Z", fill: "#f2f2ef" }, rocket);
  svg("path", { d: "M -1.6 3 L -3.6 5.5 M 1.6 3 L 3.6 5.5", stroke: "#c9ced6", "stroke-width": 0.7 }, rocket);
  svg("path", { d: "M -1 3.4 L 0 8 L 1 3.4 Z", fill: "#ffb347", opacity: 0.85 }, rocket);
  return { root, rocket };
}

export class MainMenu {
  private readonly scene: SVGSVGElement;
  private readonly rocket: SVGGElement;
  private readonly list: HTMLElement;
  private readonly soundItem: HTMLButtonElement;
  private frame: number | null = null;
  private angle = 2.6;
  private lastTime: number | null = null;

  constructor(
    private readonly root: HTMLElement,
    private readonly onPick: (config: Config) => void,
    private readonly onToggleSound: () => boolean,
    soundOn: boolean,
    version: string,
  ) {
    const { root: scene, rocket } = buildScene();
    this.scene = scene;
    this.rocket = rocket;

    const panel = html("div", "menu-panel");
    const title = html("h1", "menu-title", "Icarus");
    const tagline = html("p", "menu-tagline", "Land the rocket");
    this.list = html("nav", "menu-levels");
    this.list.setAttribute("aria-label", "Choose a mission");
    panel.append(title, tagline, this.list);

    const footer = html("div", "menu-footer");
    this.soundItem = html("button", "menu-sound");
    this.soundItem.type = "button";
    this.soundItem.addEventListener("click", () => this.setSoundLabel(this.onToggleSound()));
    this.setSoundLabel(soundOn);
    footer.append(this.soundItem, html("span", "menu-version", version));

    root.replaceChildren(scene, panel, footer);
  }

  private setSoundLabel(on: boolean): void {
    this.soundItem.textContent = on ? "Sound on" : "Sound off";
    this.soundItem.setAttribute("aria-pressed", String(on));
  }

  show(levels: readonly MenuLevel[]): void {
    const items = levels.map((level, i) => {
      const b = html("button", "menu-level");
      b.type = "button";
      const index = html("span", "menu-index", String(i + 1));
      const title = html("span", "menu-level-title", level.title);
      const status = html("span", "menu-level-best");
      if (level.best !== null) status.textContent = `Best ${level.best.toLocaleString("en-US")}`;
      else if (level.landed) status.textContent = "Landed";
      b.append(index, title, status);
      b.addEventListener("click", () => this.onPick(level.config));
      b.addEventListener("mouseenter", () => b.focus({ preventScroll: true }));
      return b;
    });
    this.list.replaceChildren(...items);
    this.root.hidden = false;
    this.lastTime = null;
    // Every level is open; the cursor starts on the first one not landed yet
    // (the suggested next mission), or the last one once all are done.
    const next = levels.findIndex((level) => !level.landed);
    items[next === -1 ? items.length - 1 : next]?.focus({ preventScroll: true });
    if (this.frame === null) this.frame = requestAnimationFrame((t) => this.animate(t));
  }

  hide(): void {
    this.root.hidden = true;
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.frame = null;
  }

  get visible(): boolean {
    return !this.root.hidden;
  }

  /** Arrow keys move between menu items while the menu is open. */
  moveFocus(delta: number): void {
    const items = [...this.root.querySelectorAll<HTMLButtonElement>(".menu-level, .menu-sound")];
    if (items.length === 0) return;
    const i = items.indexOf(document.activeElement as HTMLButtonElement);
    const next = i < 0 ? items[delta > 0 ? 0 : items.length - 1] : items[(i + delta + items.length) % items.length];
    next?.focus({ preventScroll: true });
  }

  private animate(now: number): void {
    const dt = this.lastTime === null ? 0 : Math.min((now - this.lastTime) / 1000, 0.1);
    this.lastTime = now;
    this.angle += dt * 0.2;
    const ex = ORBIT.rx * Math.cos(this.angle);
    const ey = ORBIT.ry * Math.sin(this.angle);
    const c = Math.cos(ORBIT.tilt);
    const s = Math.sin(ORBIT.tilt);
    const x = ORBIT.cx + ex * c - ey * s;
    const y = ORBIT.cy + ex * s + ey * c;
    // Heading along the orbit (derivative of the ellipse), nose first.
    const tx = -ORBIT.rx * Math.sin(this.angle);
    const ty = ORBIT.ry * Math.cos(this.angle);
    const heading = (Math.atan2(tx * s + ty * c, tx * c - ty * s) * 180) / Math.PI + 90;
    // Behind the Earth on the far side of the orbit: dimmed.
    const behind = Math.sin(this.angle) < -0.2 && Math.cos(this.angle) > -0.3;
    this.rocket.setAttribute("transform", `translate(${x.toFixed(2)} ${y.toFixed(2)}) rotate(${heading.toFixed(1)}) scale(1.4)`);
    this.rocket.setAttribute("opacity", behind ? "0.25" : "1");
    this.frame = requestAnimationFrame((t) => this.animate(t));
  }
}
