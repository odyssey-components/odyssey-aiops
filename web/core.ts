import { encode as fixedEncode } from "./src/lessons/fixed_map_v1";
import { encode as randomEncode, validMaze } from "./src/lessons/random_map_v1";
import { encode as localEncode, validMaze as validLocalMaze, type Memory } from "./src/lessons/local_map_v1";
import { encode as memoryEncode } from "./src/lessons/memory_map_v1";
import fixedMaze from "./maze.json";
export type LessonId = "fixed-map-v1" | "random-map-v1" | "local-map-v1" | "memory-map-v1";
export function modelLesson(model: Model): LessonId {
  if (model.lesson === undefined && model.format === "odymaze-policy/v1") return "fixed-map-v1";
  if (model.lesson === "fixed-map-v1" || model.lesson === "random-map-v1" || model.lesson === "local-map-v1" || model.lesson === "memory-map-v1") return model.lesson;
  throw new Error("模型课程不支持");
}
// Lessons 1–3: the same maze rules and model math, independently in JavaScript.
export interface Maze {
  id: string;
  rows: string[];
  start: { x: number; y: number; direction: number };
  max_steps: number;
}

export type GameState = Maze["start"] & {
  steps: number;
  status: "ready" | "running" | "success" | "timeout";
};

interface Layer {
  activation: string;
  weight: number[][];
  bias: number[];
}

export interface Model {
  format: string;
  lesson?: LessonId;
  generator?: string;
  maze: Maze;
  encoding: string;
  actions: string[];
  layers: Layer[];
  gru?: { hidden_size: number; weight_ih: number[][]; weight_hh: number[][]; bias_ih: number[]; bias_hh: number[] };
  metadata: { epochs: number; seed: number; samples: number; accuracy: number; success: boolean; test_success_rate?: number };
}

export const ACTIONS = ["左转", "右转", "前进"];
export const HEADINGS = ["东", "南", "西", "北"];
const DIRECTIONS = [[1, 0], [0, 1], [-1, 0], [0, -1]];
export const MAX_MODEL_BYTES = 1024 * 1024;

export function initialState(maze: Maze): GameState {
  return { ...maze.start, steps: 0, status: "ready" };
}

export function terminal(state: GameState) {
  return state.status === "success" || state.status === "timeout";
}

function walkable(maze: Maze, x: number, y: number) {
  return y >= 0 && y < maze.rows.length && x >= 0 && x < maze.rows[0].length && maze.rows[y][x] !== "#";
}

export function step(maze: Maze, state: GameState, action: number): GameState {
  if (![0, 1, 2].includes(action)) throw new Error("动作必须是左转、右转或前进");
  if (terminal(state)) return state;
  let { x, y, direction } = state;
  if (action !== 2) direction = (direction + (action === 1 ? 1 : 3)) % 4;
  else {
    const [dx, dy] = DIRECTIONS[direction];
    if (walkable(maze, x + dx, y + dy)) { x += dx; y += dy; }
  }
  const steps = state.steps + 1;
  const status = maze.rows[y][x] === "G" ? "success" : steps >= maze.max_steps ? "timeout" : "running";
  return { x, y, direction, steps, status };
}

export const encode = fixedEncode;

export function expertAction(maze: Maze, state: GameState) {
  const distances = new Map<string, number>();
  const queue: { x: number; y: number; distance: number }[] = [];
  maze.rows.forEach((row, y) => [...row].forEach((cell, x) => {
    if (cell === "G") { queue.push({ x, y, distance: 0 }); distances.set(`${x},${y}`, 0); }
  }));
  for (let index = 0; index < queue.length; index++) {
    const { x, y, distance } = queue[index];
    for (const [dx, dy] of DIRECTIONS) {
      const nx = x + dx, ny = y + dy, key = `${nx},${ny}`;
      if (walkable(maze, nx, ny) && !distances.has(key)) {
        distances.set(key, distance + 1);
        queue.push({ x: nx, y: ny, distance: distance + 1 });
      }
    }
  }
  const distance = distances.get(`${state.x},${state.y}`);
  if (!distance) throw new Error("专家只能在可达的非出口位置行动");
  const target = DIRECTIONS.findIndex(([dx, dy]) => distances.get(`${state.x + dx},${state.y + dy}`) === distance - 1);
  const turn = (target - state.direction + 4) % 4;
  return turn === 0 ? 2 : turn === 1 ? 1 : 0;
}

function sameMaze(left: Maze, right: Maze): boolean {
  return Boolean(left && right && left.id === right.id && left.max_steps === right.max_steps
    && JSON.stringify(left.rows) === JSON.stringify(right.rows) && left.start && right.start
    && (["x", "y", "direction"] as const).every(key => left.start[key] === right.start[key]));
}

// Import only bounded numeric data. Metadata never determines game success.
export function validateModel(input: unknown, maze: Maze): Model {
  const model = input as Model;
  function fail(message: string): never { throw new Error(message); }
  if (!model) fail("模型格式不支持，请选择 AIOps 的 model.json");
  const lesson = modelLesson(model);
  const random = lesson === "random-map-v1";
  const memory = lesson === "memory-map-v1";
  const local = lesson === "local-map-v1" || memory;
  if (model.format !== (memory ? "odymaze-policy/v4" : local ? "odymaze-policy/v3" : random ? "odymaze-policy/v2" : "odymaze-policy/v1")) fail("模型格式与课程不兼容");
  if ((random || local) && (model.generator !== "lcg-corridor-7/v1" || !(local ? validLocalMaze(model.maze) : validMaze(model.maze)))) fail("随机地图或生成器无效");
  const expectedMaze = random || local ? maze : fixedMaze;
  if (!random && !local && !sameMaze(model.maze, expectedMaze)) fail("模型对应的地图与当前迷宫不同");
  if (model.encoding !== (memory ? "local-view-xy-direction-previous-action/v1" : local ? "grid-known-walls-goal-visited-agent-direction/v1" : random ? "grid-walls-goal-agent-direction/v1" : "one-hot-xy-direction/v1")
      || JSON.stringify(model.actions) !== JSON.stringify(["left", "right", "forward"])) fail("模型的输入编码或动作顺序不兼容");
  if (!Array.isArray(model.layers) || model.layers.length !== (memory ? 1 : 2)) fail("模型网络层数与课程不兼容");
  const inputSize = local ? 249 : random ? 151 : expectedMaze.rows[0].length + expectedMaze.rows.length + 4;
  const hidden = local ? 64 : random ? 128 : 32;
  const vector = (value: unknown, size: number) => Array.isArray(value) && value.length === size
    && value.every((n) => typeof n === "number" && Number.isFinite(n) && Math.abs(n) <= 100);
  if (memory) {
    const gru = model.gru;
    const matrix = (value: unknown, rows: number, cols: number) => Array.isArray(value) && value.length === rows && value.every(row => vector(row, cols));
    if (!gru || gru.hidden_size !== 32 || !matrix(gru.weight_ih, 96, 40) || !matrix(gru.weight_hh, 96, 32)
        || !vector(gru.bias_ih, 96) || !vector(gru.bias_hh, 96)) fail("GRU 参数尺寸错误，或包含无效数值");
  }
  const shapes: readonly (readonly [number, number, string])[] = memory ? [[32, 3, "linear"]] : [[inputSize, hidden, "tanh"], [hidden, 3, "linear"]];
  for (const [index, [inputs, outputs, activation]] of shapes.entries()) {
    const layer = model.layers[index];
    if (!layer || layer.activation !== activation || !vector(layer.bias, outputs)
        || !Array.isArray(layer.weight) || layer.weight.length !== outputs
        || !layer.weight.every((row: unknown) => vector(row, inputs))) fail("模型参数尺寸错误，或包含无效数值");
  }
  const meta = model.metadata;
  if (!meta || !Number.isInteger(meta.epochs) || meta.epochs < 0 || meta.epochs > 5000
      || !Number.isInteger(meta.seed) || meta.seed < 0 || meta.seed > 2147483647
      || !Number.isInteger(meta.samples) || meta.samples < 0
      || typeof meta.accuracy !== "number" || !Number.isFinite(meta.accuracy) || meta.accuracy < 0 || meta.accuracy > 1
      || (meta.test_success_rate !== undefined && (typeof meta.test_success_rate !== "number"
        || !Number.isFinite(meta.test_success_rate) || meta.test_success_rate < 0 || meta.test_success_rate > 1))
      || typeof meta.success !== "boolean") fail("模型训练摘要无效");
  return model;
}

export function probabilities(model: Model, maze: Maze, state: GameState, memory?: Memory | null) {
  const lesson = modelLesson(model);
  if (lesson === "memory-map-v1") throw new Error("GRU 模型必须使用显式循环状态 predictRecurrent");
  if (lesson === "fixed-map-v1" && !sameMaze(maze, model.maze)) throw new Error("模型与当前课程地图不兼容");
  if (lesson === "random-map-v1" && !validMaze(maze)) throw new Error("随机课程地图无效");
  if (lesson === "local-map-v1" && (!validLocalMaze(maze) || !memory)) throw new Error("局部课程需要有效地图与当前记忆");
  let values = lesson === "local-map-v1" ? localEncode(memory!, state) : lesson === "random-map-v1" ? randomEncode(maze, state) : fixedEncode(maze, state);
  for (const layer of model.layers) {
    values = layer.weight.map((row, i) => {
      const sum = row.reduce((n, weight, j) => n + weight * values[j], layer.bias[i]);
      return layer.activation === "tanh" ? Math.tanh(sum) : sum;
    });
  }
  const max = Math.max(...values);
  const exps = values.map((n) => Math.exp(n - max));
  const total = exps.reduce((sum, n) => sum + n, 0);
  return exps.map((n) => n / total);
}

export function bestAction(values: readonly number[]) {
  return values.indexOf(Math.max(...values));
}


export function predictRecurrent(model: Model, maze: Maze, state: GameState, hidden: number[], previousAction = -1): { hidden: number[]; probabilities: number[] } {
  if (modelLesson(model) !== "memory-map-v1" || !model.gru || !validLocalMaze(maze)) throw new Error("GRU 课程或地图无效");
  if (hidden.length !== 32 || hidden.some(n => !Number.isFinite(n))) throw new Error("GRU 隐藏状态无效");
  const input = memoryEncode(maze, state, previousAction), gru = model.gru;
  const affine = (weights: number[][], bias: number[], values: number[]) => weights.map((row, i) => row.reduce((sum, weight, j) => sum + weight * values[j], bias[i]));
  const i = affine(gru.weight_ih, gru.bias_ih, input), h = affine(gru.weight_hh, gru.bias_hh, hidden);
  const sigmoid = (n: number) => 1 / (1 + Math.exp(-n));
  const next = hidden.map((old, j) => {
    const r = sigmoid(i[j] + h[j]), z = sigmoid(i[32 + j] + h[32 + j]);
    const n = Math.tanh(i[64 + j] + r * h[64 + j]);
    return (1 - z) * n + z * old;
  });
  const logits = affine(model.layers[0].weight, model.layers[0].bias, next);
  const exps = logits.map(n => Math.exp(n - Math.max(...logits))), total = exps.reduce((a, b) => a + b, 0);
  return { hidden: next, probabilities: exps.map(n => n / total) };
}
