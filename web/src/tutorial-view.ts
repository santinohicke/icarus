// Tutorial overlay. While time is frozen the screen turns gray and dark
// except the HUD items the lesson is about, which stay in colour, and white
// text floats next to the rocket: "Press [key picture] to ...". textContent only.

import type { Lesson } from "./tutorial";

/** Space around a highlighted element left out of the gray, in CSS pixels. */
const PAD = 6;
const RADIUS = 10;
/** Gap between the rocket and the text. */
const GAP = 48;
const EDGE = 16;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

/** SVG path of a rounded rectangle (a hole in the gray layer). */
function roundedRect(x: number, y: number, w: number, h: number, r: number): string {
  const q = Math.min(r, w / 2, h / 2);
  return (
    `M${x + q} ${y}H${x + w - q}A${q} ${q} 0 0 1 ${x + w} ${y + q}V${y + h - q}` +
    `A${q} ${q} 0 0 1 ${x + w - q} ${y + h}H${x + q}A${q} ${q} 0 0 1 ${x} ${y + h - q}V${y + q}A${q} ${q} 0 0 1 ${x + q} ${y}Z`
  );
}

/** "Press [key] to ...": the key as a picture, or the touch control's name. */
function instruction(lesson: Lesson, touch: boolean): HTMLElement {
  const line = el("p", "tutorial-line");
  line.append(touch ? `${lesson.touchVerb} ` : "Press ");
  if (touch) {
    line.append(el("strong", "tutorial-touch-key", lesson.touchKey));
  } else {
    const img = el("img", "tutorial-key");
    img.src = lesson.image;
    img.alt = lesson.key;
    img.draggable = false;
    line.append(img);
  }
  line.append(` ${lesson.action}`);
  return line;
}

export class TutorialView {
  private readonly dim = el("div", "tutorial-dim");
  private readonly text = el("div", "tutorial-text");
  private shown: { lesson: Lesson; touch: boolean } | null = null;

  constructor(root: HTMLElement) {
    this.text.setAttribute("role", "status");
    root.append(this.dim, this.text);
    this.hide();
  }

  hide(): void {
    this.dim.hidden = true;
    this.text.hidden = true;
    this.shown = null;
  }

  /**
   * Shows the frozen `lesson`: the gray screen with `targets` left in colour,
   * and the instruction beside `rocket` (where the stage is drawn). Call once
   * per frame: the text and the holes follow the layout.
   */
  update(lesson: Lesson, targets: readonly HTMLElement[], rocket: { x: number; y: number; half: number } | null, touch: boolean): void {
    if (this.shown?.lesson !== lesson || this.shown.touch !== touch) {
      this.text.replaceChildren(instruction(lesson, touch));
      this.shown = { lesson, touch };
    }
    this.text.hidden = false;
    this.place(rocket);
    const holes = targets
      .map((t) => t.getBoundingClientRect())
      .filter((r) => r.width > 0 && r.height > 0)
      .map((r) => roundedRect(r.left - PAD, r.top - PAD, r.width + 2 * PAD, r.height + 2 * PAD, RADIUS))
      .join("");
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.dim.style.clipPath = `path(evenodd, "M0 0H${w}V${h}H0Z${holes}")`;
    this.dim.hidden = false;
  }

  /** Beside the rocket (right, else left); under it when neither side has room. */
  private place(rocket: { x: number; y: number; half: number } | null): void {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const box = this.text.getBoundingClientRect();
    const r = rocket ?? { x: w / 2, y: h / 2, half: 40 };
    let x: number;
    let y = r.y - box.height / 2;
    if (r.x + GAP + box.width <= w - EDGE) x = r.x + GAP;
    else if (r.x - GAP - box.width >= EDGE) x = r.x - GAP - box.width;
    else {
      x = (w - box.width) / 2;
      y = r.y + r.half + 24;
    }
    // Clear of the touch controls along the bottom, which the text may point at.
    const bottom = document.body.classList.contains("touch-mode") ? Math.min(0.38 * h, 240) + 24 : EDGE;
    y = Math.min(Math.max(y, 70), h - box.height - bottom);
    this.text.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
  }
}
