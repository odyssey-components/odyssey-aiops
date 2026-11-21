import { describe, expect, test } from "bun:test";
import maze from "../maze.json";
import { encode, expertAction, initialState, step, validateModel, probabilities } from "../core";

describe("迷宫环境", () => {
  test("转向、撞墙、前进均计步；专家真实到达出口", () => {
    let state = initialState(maze);
    state = step(maze, state, 0);
    state = step(maze, state, 2);
    expect([state.x, state.y, state.direction, state.steps]).toEqual([1, 1, 3, 2]);
    state = initialState(maze);
    expect(expertAction(maze, state)).toBe(1);
    while (state.status !== "success" && state.status !== "timeout") {
      state = step(maze, state, expertAction(maze, state));
    }
    expect([state.x, state.y, state.steps, state.status]).toEqual([1, 5, 12, "success"]);
    expect(step(maze, state, 0)).toEqual(state);
  });

  test("上限终止，输入按 x、y、朝向分别编码", () => {
    const state = step(maze, { ...initialState(maze), steps: 99 }, 0);
    expect(state.status).toBe("timeout");
    expect(encode(maze, { x: 2, y: 3, direction: 1 }).flatMap((n, i) => n ? [i] : [])).toEqual([2, 10, 15]);
  });
});

function zeroModel() {
  return {
    format: "odymaze-policy/v1", maze,
    encoding: "one-hot-xy-direction/v1", actions: ["left", "right", "forward"],
    layers: [
      { activation: "tanh", weight: Array.from({ length: 32 }, () => Array(18).fill(0)), bias: Array(32).fill(0) },
      { activation: "linear", weight: Array.from({ length: 3 }, () => Array(32).fill(0)), bias: [0, 0, 0] },
    ],
    metadata: { seed: 7, epochs: 0, samples: 0, accuracy: 0, success: false },
  };
}

describe("模型契约与推理", () => {
  test("执行 tanh 和两层矩阵；稳定 softmax 只接受数字", () => {
    const model = zeroModel();
    model.layers[0].weight[0][1] = 1;
    model.layers[1].weight[1][0] = 2;
    model.layers[1].bias = [0, 0, -1];
    const result = probabilities(validateModel(model, maze), maze, initialState(maze));
    // Independent scalar calculation: softmax([0, 2*tanh(1), -1]).
    expect(result[0]).toBeCloseTo(0.1679344153, 9);
    expect(result[1]).toBeCloseTo(0.7702859658, 9);
    expect(result[2]).toBeCloseTo(0.0617796189, 9);
    expect(result.reduce((sum, n) => sum + n, 0)).toBeCloseTo(1, 12);
  });

  for (const [name, change] of [
    ["版本", (m) => { m.format = "unknown"; }],
    ["地图", (m) => { m.maze = { ...maze, id: "other" }; }],
    ["尺寸", (m) => { m.layers[0].weight[0].pop(); }],
    ["NaN", (m) => { m.layers[1].bias[0] = NaN; }],
    ["无穷", (m) => { m.layers[1].bias[0] = Infinity; }],
    ["字符串", (m) => { m.layers[1].bias[0] = "0"; }],
    ["布尔", (m) => { m.layers[1].bias[0] = true; }],
    ["极大权重", (m) => { m.layers[1].bias[0] = 101; }],
    ["动作顺序", (m) => { m.actions.reverse(); }],
    ["空值", () => {}],
  ]) {
    test(`拒绝${name}异常模型`, () => {
      const model = zeroModel();
      change(model);
      expect(() => validateModel(name === "空值" ? null : model, maze)).toThrow();
    });
  }
});

test('legacy model maze object key order does not change semantic compatibility', () => {
  const source = zeroModel();
  source.maze = { max_steps: maze.max_steps, start: { direction: maze.start.direction, y: maze.start.y, x: maze.start.x }, rows: maze.rows, id: maze.id };
  const model = validateModel(source, maze);
  expect(probabilities(model, maze, initialState(maze))).toEqual([1/3,1/3,1/3]);
});
