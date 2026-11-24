// Verifies the WebAssembly build against the golden trajectories, bit for bit.
//
// Usage (after `npm run wasm` in web/):  node parity/verify_wasm.mjs
// Exits with status 1 on any mismatch.

import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { WasmController, WasmReplay, WasmSim, initSync, physicsVersion, telemetryFields } from "../web/src/wasm/icarus_wasm.js";

const here = dirname(fileURLToPath(import.meta.url));
const goldenDir = join(here, "golden");
initSync({ module: readFileSync(join(here, "..", "web", "src", "wasm", "icarus_wasm_bg.wasm")) });

const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const MASK64 = (1n << 64n) - 1n;

/** Bit patterns of a Float64Array (copied, so the WASM memory can move). */
function bitsOf(values) {
  return new BigUint64Array(Float64Array.from(values).buffer);
}

function hex(bits) {
  return Array.from(bits, (b) => b.toString(16).padStart(16, "0"));
}

function fnv1a(h, bits) {
  for (const word of bits) {
    for (let i = 0n; i < 8n; i += 1n) {
      h ^= (word >> (8n * i)) & 0xffn;
      h = (h * FNV_PRIME) & MASK64;
    }
  }
  return h;
}

function fromHex(text) {
  const view = new DataView(new ArrayBuffer(8));
  view.setBigUint64(0, BigInt(`0x${text}`), true);
  return view.getFloat64(0, true);
}

function applyEvent(sim, e) {
  const a = fromHex(e.a);
  const b = fromHex(e.b);
  switch (e.kind) {
    case "gimbal_stuck":
      return sim.injectGimbalStuck(a);
    case "thrust_loss":
      return sim.injectThrustLoss(a);
    case "gust":
      return sim.injectGust(a, b);
    default:
      throw new Error(`unknown event kind ${e.kind}`);
  }
}

function check(golden) {
  const errors = [];
  if (golden.physics_version !== physicsVersion()) {
    return [`physics version ${golden.physics_version} != build ${physicsVersion()}`];
  }
  if (JSON.stringify(golden.telemetry_fields) !== JSON.stringify(telemetryFields())) {
    return ["telemetry layout differs from the build"];
  }

  if (golden.format !== 1) return [`unsupported golden format ${golden.format}`];
  if (typeof golden.replay !== "string") return ["golden has no replay string"];
  const lastSample = golden.samples[golden.samples.length - 1];
  if (lastSample.tick !== golden.ticks) return ["last sample is not at the final tick"];

  const sim = new WasmSim(golden.config, BigInt(golden.seed));
  // Goldens flown by a controller: it must make the recorded decision every tick.
  const controller = golden.controller ? new WasmController(golden.controller, golden.config) : null;
  const samples = new Map(golden.samples.map((s) => [s.tick, s.telemetry]));
  let bits = bitsOf(sim.telemetry());
  let h = fnv1a(FNV_OFFSET, bits);
  const compareSample = (tick) => {
    const expected = samples.get(tick);
    if (expected === undefined) return;
    const actual = hex(bits);
    const i = actual.findIndex((v, k) => v !== expected[k]);
    if (i >= 0) {
      errors.push(`tick ${tick}: ${golden.telemetry_fields[i]} = ${actual[i]}, expected ${expected[i]}`);
    }
  };
  compareSample(0);

  let tick = 0;
  for (const [count, q0, q1, q2] of golden.inputs) {
    for (let n = 0; n < count; n += 1) {
      for (const e of golden.events) if (e.tick === tick) applyEvent(sim, e);
      if (controller !== null) {
        const d = controller.act(sim);
        if (d[0] !== q0 || d[1] !== q1 || d[2] !== q2) {
          errors.push(`tick ${tick}: controller decided [${Array.from(d)}], recorded [${q0},${q1},${q2}]`);
          break;
        }
      }
      sim.step(q0, q1, q2);
      tick += 1;
      const telemetry = sim.telemetry();
      if (!telemetry.every(Number.isFinite)) errors.push(`tick ${tick}: non-finite telemetry`);
      bits = bitsOf(telemetry);
      h = fnv1a(h, bits);
      compareSample(tick);
      if (errors.length > 5) break;
    }
    if (errors.length > 5) break;
  }
  if (tick !== golden.ticks) errors.push(`ran ${tick} ticks, expected ${golden.ticks}`);
  if (sim.outcomeName() !== (golden.outcome ?? "")) {
    errors.push(`outcome ${sim.outcomeName()}, expected ${golden.outcome}`);
  }
  if (sim.score() !== golden.score) errors.push(`score ${sim.score()}, expected ${golden.score}`);
  const hash = h.toString(16).padStart(16, "0");
  if (hash !== golden.hash) errors.push(`trajectory hash ${hash}, expected ${golden.hash}`);

  // WebAssembly must encode exactly the replay string the Python module wrote,
  // and that string must replay to the same final state.
  if (sim.replay() !== golden.replay) errors.push("WASM encodes a different replay string");
  const player = new WasmReplay(golden.replay);
  while (!player.step()) {
    /* advance */
  }
  if (hex(bitsOf(player.telemetry())).join() !== lastSample.telemetry.join()) {
    errors.push("replay string does not reproduce the final state");
  }
  player.free();
  controller?.free();
  sim.free();
  return errors;
}

const manifest = JSON.parse(readFileSync(join(goldenDir, "manifest.json"), "utf8"));
const expected = [...manifest.cases].sort();
const files = readdirSync(goldenDir)
  .filter((f) => f.endsWith(".json") && f !== "manifest.json")
  .map((f) => f.slice(0, -5))
  .sort();
if (expected.length === 0 || JSON.stringify(files) !== JSON.stringify(expected)) {
  console.error(`golden files ${JSON.stringify(files)} do not match the manifest ${JSON.stringify(expected)}`);
  process.exit(1);
}
let failed = 0;
for (const name of files) {
  const golden = JSON.parse(readFileSync(join(goldenDir, `${name}.json`), "utf8"));
  let errors;
  try {
    errors = golden.name === name ? check(golden) : [`file ${name}.json holds case ${golden.name}`];
  } catch (error) {
    errors = [`threw: ${error}`];
  }
  if (errors.length === 0) {
    console.log(`ok       ${golden.name} (${golden.ticks} ticks, ${golden.outcome ?? "running"})`);
  } else {
    failed += 1;
    console.log(`MISMATCH ${golden.name}`);
    for (const e of errors.slice(0, 6)) console.log(`         ${e}`);
  }
}
if (failed > 0) {
  console.error(`${failed} of ${files.length} golden trajectories differ in WebAssembly`);
  process.exit(1);
}
console.log(`WebAssembly matches all ${files.length} golden trajectories bit for bit.`);
