import { describe, expect, it } from "vitest";

import { isPhone } from "../src/device";

describe("isPhone", () => {
  it("is true for touch-only phones in either orientation, large ones included", () => {
    expect(isPhone(390, 844, true)).toBe(true);
    expect(isPhone(932, 430, true)).toBe(true);
    expect(isPhone(448, 998, true)).toBe(true);
    // A large phone with a smaller display size setting.
    expect(isPhone(540, 1200, true)).toBe(true);
    expect(isPhone(599, 1300, true)).toBe(true);
  });

  it("is false for tablets, open foldables and anything with a mouse", () => {
    expect(isPhone(600, 960, true)).toBe(false);
    expect(isPhone(700, 830, true)).toBe(false);
    expect(isPhone(744, 1133, true)).toBe(false);
    expect(isPhone(1024, 768, true)).toBe(false);
    expect(isPhone(390, 844, false)).toBe(false);
    expect(isPhone(1920, 1080, false)).toBe(false);
  });
});
