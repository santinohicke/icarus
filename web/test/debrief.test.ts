import { describe, expect, it } from "vitest";

import { limitStats, touchdownChecks } from "../src/debrief";
import type { Telemetry, VehicleInfo } from "../src/sim";

const DEG = Math.PI / 180;
const info = { max_vertical_speed: 3, max_horizontal_speed: 2, max_tilt: 10 * DEG } as unknown as VehicleInfo;

function state(vy: number, vx: number, thetaDeg: number, padMargin = 5): Telemetry {
  return { contact_vy: vy, contact_vx: vx, theta: thetaDeg * DEG, pad_margin: padMargin } as unknown as Telemetry;
}

describe("touchdownChecks", () => {
  it("compares each touchdown value with its limit", () => {
    const checks = touchdownChecks("crashed_vertical_speed", state(-3.2, 2.4, 1.2), info);
    expect(checks).toEqual([
      { label: "Descent", value: "3.2 m/s", limit: "3 m/s", ok: false },
      { label: "Sideways", value: "2.4 m/s", limit: "2 m/s", ok: false },
      { label: "Tilt", value: "1.2°", limit: "10°", ok: true },
    ]);
  });

  it("adds the miss distance, never shown as 0.0, when a leg missed the pad", () => {
    const checks = touchdownChecks("crashed_off_pad", state(-1, 0, 0, -0.01), info);
    expect(checks.at(-1)).toEqual({ label: "Off the pad", value: "0.1 m", limit: "0 m", ok: false });
  });

  it("never rounds a passing value up to its limit", () => {
    expect(touchdownChecks("landed", state(-2.97, 0, 0), info)[0]).toMatchObject({ value: "2.9 m/s", ok: true });
  });

  it("adds the spin rate when the stage crashed spinning", () => {
    const spin = { ...info, max_angular_rate: 20 * DEG } as unknown as VehicleInfo;
    const s = { ...state(-1, 0, 0), omega: -30 * DEG } as unknown as Telemetry;
    expect(touchdownChecks("crashed_angular_rate", s, spin).at(-1)).toEqual({ label: "Spin", value: "30.0°/s", limit: "20°/s", ok: false });
  });

  it("wraps the tilt angle", () => {
    expect(touchdownChecks("landed", state(-1, 0, 361), info)[2]?.value).toBe("1.0°");
  });

  it("is empty when the flight ended before touchdown", () => {
    expect(touchdownChecks("out_of_propellant", state(-20, 0, 0), info)).toEqual([]);
    expect(touchdownChecks("", state(-20, 0, 0), info)).toEqual([]);
  });
});

describe("limitStats", () => {
  it("names the limits without symbols", () => {
    const [descent, sideways, tilt] = limitStats(info, 3, true);
    expect([descent?.label, descent?.value, descent?.unit]).toEqual(["Max descent", "3", "m/s"]);
    expect([sideways?.label, sideways?.value]).toEqual(["Max sideways", "2"]);
    expect([tilt?.label, tilt?.value, tilt?.unit]).toEqual(["Max tilt", "10", "°"]);
  });
});
