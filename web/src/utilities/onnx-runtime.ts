import {fetchModelArtifact} from '../apis/models';
import type { InferenceSession, Tensor } from "onnxruntime-web/wasm";
import wasmUrl from "odymaze-ort-wasm?url";
import loaderUrl from "odymaze-ort-loader?url";
import type { Model } from "../../core";
import { policyFromManifest, validateOnnxManifest, type OnnxManifest } from "./onnx-contract";

export interface OnnxRuntimeModel {
  policy: Model;
  manifest: OnnxManifest;
  predict(observation: number[], hidden?: number[]): Promise<{ values: number[]; hidden: number[] | null }>;
  dispose(): Promise<void>;
}


function checkSession(session: InferenceSession, manifest: OnnxManifest) {
  for (const [names, metadata, expected] of [
    [session.inputNames, session.inputMetadata, manifest.inputs],
    [session.outputNames, session.outputMetadata, manifest.outputs],
  ] as const) {
    if (names.length !== expected.length || metadata.length !== expected.length) throw new Error("ONNX 输入输出数量与清单不符");
    for (let index = 0; index < expected.length; index++) {
      const item = metadata[index], contract = expected[index];
      if (names[index] !== contract.name || item.name !== contract.name || !item.isTensor
        || item.type !== "float32" || item.shape.length !== 2
        || (item.shape[0] !== "batch" && item.shape[0] !== 1) || item.shape[1] !== contract.shape[1]) {
        throw new Error("ONNX 输入输出名称、类型或形状与清单不符");
      }
    }
  }
}

function finiteVector(value: number[], width: number, label: string) {
  if (value.length !== width || !value.every(Number.isFinite)) throw new Error(`${label}必须包含 ${width} 个有限数值`);
  const data = Float32Array.from(value);
  if (!data.every(Number.isFinite)) throw new Error(`${label}超出 float32 范围`);
  return data;
}

function outputVector(tensor: Tensor | undefined, width: number, label: string) {
  if (!tensor || tensor.type !== "float32" || tensor.dims.length !== 2 || tensor.dims[0] !== 1
    || tensor.dims[1] !== width || tensor.data.length !== width) throw new Error(`ONNX ${label}输出形状无效`);
  const values = Array.from(tensor.data as Float32Array);
  if (!values.every(Number.isFinite)) throw new Error(`ONNX ${label}输出包含无效数值`);
  return values;
}

export async function loadOnnxModel(run: string): Promise<OnnxRuntimeModel> {
  const manifest = validateOnnxManifest(JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(
    await fetchModelArtifact(run, "manifest", 64 * 1024),
  )));
  const bytes = await fetchModelArtifact(run, "onnx", 1024 * 1024);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const sha256 = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, "0")).join("");
  if (sha256 !== manifest.sha256) throw new Error("ONNX 文件校验失败，请重新导出模型");
  const ort = await import("onnxruntime-web/wasm");
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = { wasm: new URL(wasmUrl, window.location.href).href, mjs: new URL(loaderUrl, window.location.href).href };
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  try { checkSession(session, manifest); } catch (error) { await session.release(); throw error; }
  const recurrent = manifest.inputs.length === 2;
  let disposed = false;
  const pending = new Set<Promise<unknown>>();
  let disposal: Promise<void> | undefined;
  return {
    policy: policyFromManifest(manifest), manifest,
    predict(observation, hidden) {
      if (disposed) return Promise.reject(new Error("ONNX 模型已释放，请重新选择模型"));
      const prediction = (async () => {
      const inputs: Record<string, Tensor> = {
        observations: new ort.Tensor("float32", finiteVector(observation, Number(manifest.inputs[0].shape[1]), "观察"), [1, Number(manifest.inputs[0].shape[1])]),
      };
      let outputs: InferenceSession.ReturnType | undefined;
      try {
        if (recurrent) inputs.hidden = new ort.Tensor("float32", finiteVector(hidden ?? Array(32).fill(0), 32, "记忆"), [1, 32]);
        else if (hidden !== undefined) throw new Error("该课程模型不接受循环记忆");
        outputs = await session.run(inputs);
        const logits = outputVector(outputs.logits, 3, "动作");
        const nextHidden = recurrent ? outputVector(outputs.next_hidden, 32, "记忆") : null;
        const maximum = Math.max(...logits);
        const weights = logits.map(value => Math.exp(value - maximum));
        const total = weights.reduce((sum, value) => sum + value, 0);
        return { values: weights.map(value => value / total), hidden: nextHidden };
      } finally {
        for (const tensor of Object.values(inputs)) tensor.dispose();
        for (const tensor of Object.values(outputs ?? {})) tensor.dispose();
      }
      })();
      pending.add(prediction);
      void prediction.then(() => pending.delete(prediction), () => pending.delete(prediction));
      return prediction;
    },
    dispose() {
      disposed = true;
      disposal ??= Promise.allSettled([...pending]).then(() => session.release());
      return disposal;
    },
  };
}
