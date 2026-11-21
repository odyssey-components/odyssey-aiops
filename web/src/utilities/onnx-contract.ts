import type { LessonId, Maze, Model } from "../../core";
import {generateMaze} from "../lessons/random_map_v1";
import fixedMaze from "../../maze.json";
import { validMaze as randomValidMaze } from "../lessons/random_map_v1";
import { validMaze as localValidMaze } from "../lessons/local_map_v1";

export interface OnnxTensor { name: string; dtype: "float32"; shape: ["batch", number] }
export interface OnnxManifest {
  schema: "odymaze-onnx/v1"; filename: "model.onnx"; sha256: string; opset: 17;
  policy_format: string; lesson: LessonId; encoding: string; generator?: string;
  actions: string[]; maze: Maze; metadata: Model["metadata"];
  outputs_are_logits: true; inputs: OnnxTensor[]; outputs: OnnxTensor[];
  dataset?: {id:string;sha256:string};
  map_ref?: {id:string};
  recurrent?: { initial_hidden: number[]; reset: "per-maze"; update: "once-per-action" };
}
const contracts = [
  ["fixed-map-v1", "odymaze-policy/v1", "one-hot-xy-direction/v1", 18],
  ["random-map-v1", "odymaze-policy/v2", "grid-walls-goal-agent-direction/v1", 151],
  ["local-map-v1", "odymaze-policy/v3", "grid-known-walls-goal-visited-agent-direction/v1", 249],
  ["memory-map-v1", "odymaze-policy/v4", "local-view-xy-direction-previous-action/v1", 40],
] as const;
const object = (value: unknown): value is Record<string, unknown> => Boolean(value && typeof value === "object" && !Array.isArray(value));
export function validateOnnxManifest(input: unknown): OnnxManifest {
  let value:Record<string,unknown>;
  const fail = (): never => { throw new Error("ONNX 模型清单无效或与课程不兼容"); };
  if (!object(input)) return fail();
  value={...input};
  if(value.dataset!==undefined && (!object(value.dataset) || typeof value.dataset.id!=='string' || !value.dataset.id || /[\\/\x00]/.test(value.dataset.id)
    || typeof value.dataset.sha256!=='string' || !/^[a-f0-9]{64}$/.test(value.dataset.sha256))) return fail();
  if(value.maze===undefined && object(value.map_ref)) {
    const id=value.map_ref.id;
    if(value.lesson==='fixed-map-v1') {if(id!==fixedMaze.id)return fail();value.maze=fixedMaze;}
    else {
      if(typeof id!=='string' || !/^odymaze-random-[0-9]+$/.test(id))return fail();
      const seed=Number(id.slice('odymaze-random-'.length));
      if(!Number.isInteger(seed) || seed<0 || seed>0xffffffff)return fail();
      const maze=generateMaze(seed);maze.max_steps=value.lesson==='random-map-v1'?100:200;
      if(maze.id!==id)return fail();value.maze=maze;
    }
  }
  for(const key of ['inputs','outputs']) {
    if(Array.isArray(value[key]))value[key]=value[key].map(item=>object(item) && object(item.shape)
      ? {...item,shape:[item.shape.batch,item.shape.width]} : item);
  }

  const contract = contracts.find(([lesson]) => lesson === value.lesson);
  if (!contract || value.schema !== "odymaze-onnx/v1" || value.filename !== "model.onnx"
      || typeof value.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(value.sha256)
      || value.opset !== 17 || value.policy_format !== contract[1] || value.encoding !== contract[2]
      || value.outputs_are_logits !== true || JSON.stringify(value.actions) !== '["left","right","forward"]') return fail();
  const fixed = contract[0] === "fixed-map-v1", recurrent = contract[0] === "memory-map-v1";
  if (fixed) {
    const maze = value.maze;
    if (!object(maze) || maze.id !== fixedMaze.id || maze.max_steps !== fixedMaze.max_steps
        || JSON.stringify(maze.rows) !== JSON.stringify(fixedMaze.rows) || !object(maze.start)
        || (["x", "y", "direction"] as const).some(key => maze.start && (maze.start as Record<string, unknown>)[key] !== fixedMaze.start[key])
        || value.generator !== undefined) return fail();
  } else if (value.generator !== "lcg-corridor-7/v1"
      || !(contract[0] === "random-map-v1" ? randomValidMaze(value.maze) : localValidMaze(value.maze))) return fail();
  const tensor = (item: unknown, name: string, width: number) => object(item) && item.name === name
    && item.dtype === "float32" && Array.isArray(item.shape) && item.shape.length === 2
    && item.shape[0] === "batch" && item.shape[1] === width;
  if (!Array.isArray(value.inputs) || !Array.isArray(value.outputs)
      || value.inputs.length !== (recurrent ? 2 : 1) || value.outputs.length !== (recurrent ? 2 : 1)
      || !tensor(value.inputs[0], "observations", contract[3]) || !tensor(value.outputs[0], "logits", 3)) return fail();
  if (recurrent) {
    const state = value.recurrent;
    if (!tensor(value.inputs[1], "hidden", 32) || !tensor(value.outputs[1], "next_hidden", 32)
        || !object(state) || state.reset !== "per-maze" || state.update !== "once-per-action"
        || !Array.isArray(state.initial_hidden) || state.initial_hidden.length !== 32
        || state.initial_hidden.some(n => n !== 0)) return fail();
  } else if (value.recurrent !== undefined) return fail();
  const meta = value.metadata;
  const integer = (n: unknown, max: number) => typeof n === "number" && Number.isInteger(n) && n >= 0 && n <= max;
  const rate = (n: unknown) => typeof n === "number" && Number.isFinite(n) && n >= 0 && n <= 1;
  if (!object(meta) || !integer(meta.seed, 2147483647) || !integer(meta.epochs, 5000)
      || !integer(meta.samples, 2147483647) || !rate(meta.accuracy) || typeof meta.success !== "boolean"
      || (meta.test_success_rate !== undefined && !rate(meta.test_success_rate))) return fail();
  return value as unknown as OnnxManifest;
}
export function policyFromManifest(manifest: OnnxManifest): Model {
  const valid = validateOnnxManifest(manifest);
  return { format: valid.policy_format, lesson: valid.lesson, generator: valid.generator,
    maze: valid.maze, encoding: valid.encoding, actions: valid.actions, metadata: valid.metadata, layers: [] };
}
