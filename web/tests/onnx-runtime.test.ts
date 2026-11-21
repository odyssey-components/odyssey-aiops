import { afterEach, expect, mock, test } from "bun:test";
import { createHash } from "node:crypto";
import maze from "../maze.json";

mock.module("odymaze-ort-wasm?url", () => ({ default: "/assets/runtime.wasm" }));
mock.module("odymaze-ort-loader?url", () => ({ default: "/assets/runtime.mjs" }));
const binary = new Uint8Array([1, 2, 3]);
const manifest = {
  schema: "odymaze-onnx/v1", filename: "model.onnx", sha256: createHash("sha256").update(binary).digest("hex"), opset: 17,
  policy_format: "odymaze-policy/v1", lesson: "fixed-map-v1", encoding: "one-hot-xy-direction/v1",
  maze, actions: ["left", "right", "forward"], outputs_are_logits: true,
  inputs: [{ name: "observations", dtype: "float32", shape: ["batch", 18] }],
  outputs: [{ name: "logits", dtype: "float32", shape: ["batch", 3] }],
  metadata: { seed: 7, epochs: 0, samples: 0, accuracy: 0, success: false },
};
class FakeTensor {
  constructor(public type: string, public data: Float32Array, public dims: number[]) {}
  dispose() {}
}
let releases = 0, creations = 0;
let inputWidth = 18;
let run = async () => ({ logits: new FakeTensor("float32", Float32Array.from([1000, 999, 998]), [1, 3]) });
mock.module("onnxruntime-web/wasm", () => ({
  env: { wasm: {} }, Tensor: FakeTensor,
  InferenceSession: { async create() {
    creations++;
    return {
      inputNames: ["observations"], outputNames: ["logits"],
      inputMetadata: [{ name: "observations", isTensor: true, type: "float32", shape: ["batch", inputWidth] }],
      outputMetadata: [{ name: "logits", isTensor: true, type: "float32", shape: ["batch", 3] }],
      run: () => run(), release: async () => { releases++; },
    };
  } },
}));
const { loadOnnxModel } = await import("../src/utilities/onnx-runtime");
const originalFetch = globalThis.fetch;
const originalWindow = globalThis.window;
function serve(badHash = false) {
  globalThis.window = { location: { href: "http://127.0.0.1:4175/" } } as Window & typeof globalThis;
  globalThis.fetch = mock(async (url: string | URL | Request) => new Response(
    String(url).endsWith("artifact=manifest") ? JSON.stringify({ ...manifest, ...(badHash ? { sha256: "0".repeat(64) } : {}) }) : binary,
  )) as unknown as typeof fetch;
}
afterEach(() => {
  globalThis.fetch = originalFetch;
  globalThis.window = originalWindow;
  inputWidth = 18;
});

test("runtime checks digest before creating a session", async () => {
  serve(true);
  const before = creations;
  await expect(loadOnnxModel("test")).rejects.toThrow("校验失败");
  expect(creations).toBe(before);
});
test("runtime rejects and releases mismatched graph metadata", async () => {
  serve(); inputWidth = 40;
  const before = releases;
  await expect(loadOnnxModel("test")).rejects.toThrow("形状");
  expect(releases).toBe(before + 1);
});
test("runtime validates observations and applies stable logits softmax", async () => {
  serve();
  const model = await loadOnnxModel("test");
  try {
    await expect(model.predict(Array(18).fill(NaN))).rejects.toThrow("有限数值");
    await expect(model.predict(Array(18).fill(1e100))).rejects.toThrow("float32");
    const result = await model.predict(Array(18).fill(0));
    expect(result.values[0]).toBeCloseTo(0.6652409558);
    expect(result.values.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(result.hidden).toBeNull();
  } finally { await model.dispose(); }
});
test("dispose waits for active inference and closes immediately", async () => {
  serve();
  let finish!: () => void;
  const originalRun = run;
  run = () => new Promise(resolve => { finish = () => resolve({ logits: new FakeTensor("float32", Float32Array.from([1, 2, 3]), [1, 3]) }); });
  const model = await loadOnnxModel("test");
  const before = releases;
  const prediction = model.predict(Array(18).fill(0));
  const disposal = model.dispose();
  await expect(model.predict(Array(18).fill(0))).rejects.toThrow("已释放");
  expect(releases).toBe(before);
  finish();
  await prediction; await disposal; await model.dispose();
  expect(releases).toBe(before + 1);
  run = originalRun;
});
