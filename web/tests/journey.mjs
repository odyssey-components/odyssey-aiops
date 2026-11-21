import { root, product } from "../../scripts/workspace.mjs";
// Product-owned, bounded headless test. No screenshots or OS automation.
import assert from "node:assert/strict";
import {startGoAPI} from "./go-api.mjs";
import { execFileSync, spawn } from "node:child_process";
import { once } from "node:events";
import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { join } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { verifyCourses } from "./course-journey.mjs";
import { verifyUrlState } from "./url-journey.mjs";


// Reuse the repository's shared test tooling dependency, not another product.
const require = createRequire(join(root, "repositories/odyssey-design/packages/odyssey-web-testkit/package.json"));
const { chromium, expect } = require("@playwright/test");
await mkdir(join(root, "tmp/odymaze-2515"), { recursive: true });
const run = await mkdtemp(join(root, "tmp/odymaze-2515/journey-"));
const uvRun = ["run", "--project", ".", "--no-sync"];
const runs = join(run, ".odyssey/odymaze/runs");
const output = join(runs, "trained");
const web = fileURLToPath(new URL("../", import.meta.url));
const errors = [];
let browser;
let passed = false;
let server;
let api;
let serverClosed;

try {
  execFileSync("uv", [...uvRun, "odymaze-train", "--output", output], { cwd: root, timeout: 60000, stdio: "pipe" });
  const modelPath = join(output, "model.json");
  const model = JSON.parse(await readFile(modelPath, "utf8"));
  // Exercise the exact legacy schema, before course IDs existed.
  delete model.lesson;
  await writeFile(modelPath, JSON.stringify(model));
  const reference = JSON.parse(execFileSync("uv", [...uvRun, "python", "-c", `
import json, sys, torch
from aiops_maze_training.lessons.fixed_map_v1.environment import load_maze, samples, State, step
from aiops_maze_training.lessons.fixed_map_v1.model import make_model
data = json.load(open(sys.argv[1]))
maze = load_maze()
model = make_model(18)
with torch.no_grad():
    for index, layer in zip((0, 2), data['layers']):
        model[index].weight.copy_(torch.tensor(layer['weight']))
        model[index].bias.copy_(torch.tensor(layer['bias']))
    rows = samples(maze)
    for row in rows:
        row['probabilities'] = torch.softmax(model(torch.tensor(row['observation'])), dim=-1).tolist()
        row['transitions'] = [vars(step(maze, State(**row['state']), action)) for action in range(3)]
print(json.dumps(rows))
`, modelPath], { cwd: root, timeout: 30000, encoding: "utf8" }));

  const emptyRun = join(runs, "empty");
  const brokenRun = join(runs, "broken");
  await mkdir(emptyRun);
  await mkdir(brokenRun);
  await writeFile(join(emptyRun, "report.json"), "{}");
  await writeFile(join(brokenRun, "model.json"), "{");
  const privateDir = join(run, "outside");
  await mkdir(privateDir);
  await writeFile(join(privateDir, "model.json"), "{}");
  await symlink(privateDir, join(runs, "linked-directory"));
  const linkedFileRun = join(runs, "linked-file");
  await mkdir(linkedFileRun);
  await symlink(join(privateDir, "model.json"), join(linkedFileRun, "model.json"));
  const preview = process.argv.includes("--preview");
  api = await startGoAPI(root,run);
  server = spawn(join(web, "node_modules/.bin/vite"), [...(preview ? ["preview"] : []), "--port", "0"], {
    cwd: web, env: { ...process.env, HOME: run, ODYMAZE_API_ORIGIN: api.origin, NO_COLOR: "1" },
    stdio: ["ignore", "pipe", "inherit"], timeout: 300000,
  });
  serverClosed = once(server, "close");
  const serverOutput = createInterface({ input: server.stdout });
  let origin;
  for await (const line of serverOutput) {
    const address = line.match(/http:\/\/127\.0\.0\.1:\d+/);
    if (address) { [origin] = address; break; }
  }
  assert.ok(origin, "Vite did not start on loopback");
  const initialPage = await fetch(origin);
  assert.equal(initialPage.status, 200);
  assert.match(await initialPage.text(), /本课有 1 个模型/);
  assert.deepEqual(await (await fetch(`${origin}/api/models`)).json(), ["trained", "broken"]);
  for (const path of ["/api/models/linked-directory", "/api/models/linked-file"]) {
    assert.equal((await fetch(origin + path)).status, 404);
  }
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(origin);
  await expect(page.getByRole("heading", { name: "从第一步，理解模型。" })).toBeVisible();
  await page.waitForFunction(() => localStorage.getItem("odymaze-web-appearance") !== null);
  const bundle = await Bun.build({ entrypoints: [join(web, "core.ts")], target: "browser" });
  assert.ok(bundle.success, String(bundle.logs));
  const source = await bundle.outputs[0].text();
  const coreModule = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
  const maze = JSON.parse(await readFile(join(web, "maze.json"), "utf8"));
  const consistency = await page.evaluate(async ({ coreModule, maze, model, reference }) => {
    const core = await import(coreModule);
    core.validateModel(model, maze);
    let maxError = 0;
    let rulesAgree = true;
    for (const row of reference) {
      const values = core.probabilities(model, maze, row.state);
      values.forEach((value, i) => { maxError = Math.max(maxError, Math.abs(value - row.probabilities[i])); });
      rulesAgree &&= JSON.stringify(core.encode(maze, row.state)) === JSON.stringify(row.observation);
      rulesAgree &&= core.expertAction(maze, row.state) === row.action;
      for (let action = 0; action < 3; action++) {
        const actual = core.step(maze, row.state, action), expected = row.transitions[action];
        rulesAgree &&= ["x", "y", "direction", "steps", "status"].every((key) => actual[key] === expected[key]);
      }
    }
    return { maxError, rulesAgree };
  }, { coreModule, maze, model, reference });
  assert.ok(consistency.maxError < 0.00001, JSON.stringify(consistency));
  assert.equal(consistency.rulesAgree, true);

  await expect(page.locator("#model-directory")).toHaveCount(0);
  await expect(page.locator("#model-file")).toHaveCount(0);
  await expect(page.locator("#load-model-selection")).toHaveCount(0);
  await expect(page.locator("#current-model")).toHaveText("trained/model.json");
  await expect(page.locator("#mode-manual")).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#model-list")).toHaveCount(1);
  await page.getByRole("button", { name: "左转", exact: true }).click();
  await page.getByRole("button", { name: "前进", exact: true }).click();
  await expect(page.locator("#position")).toHaveText("(1, 1)");
  await expect(page.locator("#heading")).toHaveText("北 ↑");
  await expect(page.locator("#step-count")).toHaveText("2 / 100");
  await expect(page.locator("#action-log")).toContainText("撞墙");
  await page.getByRole("button", { name: "重置", exact: true }).click();
  await page.getByRole("button", { name: "专家", exact: true }).click();
  await page.locator("#speed").click();
  await page.getByRole("option", { name: "快速", exact: true }).click();
  await page.getByRole("button", { name: "自动运行", exact: true }).click();
  await expect(page.locator("#game-status")).toHaveText("已到达出口", { timeout: 10000 });
  await expect(page.locator("#step-count")).toHaveText("12 / 100");
  await expect(page.locator("#auto")).toBeDisabled();

  await expect(page.locator("#mode-model")).toBeEnabled();
  const newestRun = join(runs, "zz-newest");
  await mkdir(newestRun);
  await writeFile(join(newestRun, "model.json"), JSON.stringify(model));
  console.log("Checking automatic-model read failure recovery");
  await page.route("**/api/models/zz-newest", route => route.fulfill({status:200,body:"{"}));
  await page.locator("#refresh-models").click();
  await expect(page.locator("#model-directory-status")).toHaveText("本课有 2 个模型。");
  await expect(page.locator("#import-message")).toHaveAttribute("data-kind", "error");
  await expect(page.locator("#current-model")).toHaveText("trained/model.json");
  await expect(page.locator("#mode-model")).toBeEnabled();
  await expect(page.locator("#mode-expert")).toHaveAttribute("aria-pressed", "true");
  await page.unroute("**/api/models/zz-newest");
  await page.locator("#refresh-models").click();
  await expect(page.locator("#current-model")).toHaveText("zz-newest/model.json");
  await page.locator("#model-list").click();
  await expect(page.getByRole("option")).toHaveText(["zz-newest/model.json", "trained/model.json"]);
  await page.getByRole("option", {name:"trained/model.json",exact:true}).click();
  await expect(page.locator("#current-model")).toHaveText("trained/model.json");
  await expect(page).toHaveURL(/model=trained/);

  await expect(page.locator("#mode-expert")).toHaveAttribute("aria-pressed", "true");
  await page.locator("#mode-model").click();
  await expect(page.locator("#model-info")).toContainText("400 轮");
  await expect(page.locator("#step-count")).toHaveText("0 / 100");
  await expect(page.locator("#mode-model")).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#probabilities")).toContainText("%");
  await page.getByRole("button", { name: "执行一步", exact: true }).click();
  await expect(page.locator("#heading")).toHaveText("南 ↓");
  await page.getByRole("button", { name: "自动运行", exact: true }).click();
  await expect(page.locator("#game-status")).toHaveText("已到达出口");
  await expect(page.locator("#position")).toHaveText("(1, 5)");

  await page.goto(`${origin}/?model=trained&mode=model&speed=40`);
  await expect(page.locator("#model-info")).toContainText("trained/model.json");
  await expect(page.locator("#step-count")).toHaveText("0 / 100");
  await page.getByRole("button", { name: "自动运行", exact: true }).click();
  await expect(page.locator("#step-count")).not.toHaveText("0 / 100");
  await page.getByRole("button", { name: "暂停", exact: true }).click();
  const paused = await page.locator("#step-count").textContent();
  // A bounded timer observation checks that cancelled timers cannot fire again.
  await page.waitForTimeout(160);
  await expect(page.locator("#step-count")).toHaveText(paused);
  await page.getByRole("button", { name: "自动运行", exact: true }).click();
  await page.getByRole("button", { name: "手动", exact: true }).click();
  const switched = await page.locator("#step-count").textContent();
  await page.waitForTimeout(160);
  await expect(page.locator("#step-count")).toHaveText(switched);
  await page.getByRole("button", { name: "重置", exact: true }).click();
  await expect(page.locator("#step-count")).toHaveText("0 / 100");

  for (const [name, content] of [
    ["broken.json", "{"],
    ["wrong-map.json", JSON.stringify({ ...model, maze: { ...model.maze, id: "wrong" } })],
    ["large.json", " ".repeat(1024 * 1024 + 1)],
  ]) {
    const invalidRun = join(runs, name);
    await mkdir(invalidRun);
    await writeFile(join(invalidRun, "model.json"), content);
    console.log(`Checking rejected URL model ${name}`);
    await page.goto(`${origin}/?model=${name}&mode=model`);
    await expect(page.locator("#import-message")).toHaveAttribute("data-kind", "error", {timeout:15000});
    await expect(page.locator("#mode-model")).toBeDisabled();
  }
  await page.goto(`${origin}/?model=trained&mode=model&speed=40`);
  await expect(page.locator("#import-message")).toHaveAttribute("data-kind", "success");

  // A valid constant-left policy must really time out, despite success metadata.
  const spinning = structuredClone(model);
  for (const layer of spinning.layers) {
    layer.weight = layer.weight.map((row) => row.map(() => 0));
    layer.bias = layer.bias.map(() => 0);
  }
  spinning.layers[1].bias[0] = 10;
  await mkdir(join(runs, "spinning"));
  await writeFile(join(runs, "spinning/model.json"), JSON.stringify(spinning));
  await page.goto(`${origin}/?model=spinning&mode=model&speed=40`);
  await expect(page.locator("#model-info")).toContainText("spinning/model.json");
  await page.getByRole("button", { name: "自动运行", exact: true }).click();
  await expect(page.locator("#game-status")).toHaveText("达到步数上限", { timeout: 10000 });
  await expect(page.locator("#step-count")).toHaveText("100 / 100");
  await expect(page.locator("#single-step")).toBeDisabled();
  await page.getByRole("button", { name: "重置", exact: true }).click();
  await expect(page.locator("#single-step")).toBeEnabled();

  for (const width of [1440, 820, 390]) {
    await page.setViewportSize({ width, height: 844 });
    for (const [selector, count] of [["#board .tile", 49], ["#encoding .bit", 18]]) {
      const boxes = await page.locator(selector).evaluateAll((cells) => cells.map((cell) => {
        const { width, height } = cell.getBoundingClientRect();
        return { width, height };
      }));
      assert.equal(boxes.length, count);
      assert.ok(boxes.every((box) => box.width > 0 && Math.abs(box.width - box.height) < 0.1
        && Math.abs(box.width - boxes[0].width) < 0.1), `${selector} cells must be equal squares at ${width}px`);
    }
  }
  await expect(page.getByText("选择模型目录", { exact: true })).toHaveCount(0);
  await expect(page.getByText("加载单个模型文件", { exact: true })).toHaveCount(0);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.setViewportSize({ width: 1440, height: 1000 });
  for (let lesson = 1; lesson <= 5; lesson++) {
    await page.locator(`[data-lesson="${lesson}"]`).click();
    await expect(page.locator("#lesson-title")).toHaveText(["认识环境", "跟随专家", "理解模型", "开始训练", "检验结果"][lesson - 1]);
  }
  await verifyCourses({ page, root, web, run, uvRun, coreModule, legacyModelPath: modelPath });
  await verifyUrlState({ page, root, origin, legacyModelPath: modelPath });
  assert.deepEqual(errors, []);
  console.log(`PASS: ${reference.length} states / ${reference.length * 3} transitions, max probability error ${consistency.maxError}; manual/expert/model, lifecycle, refusal/recovery, timeout, narrow layout.`);
  passed = true;
} finally {
  await browser?.close();
  server?.kill();
  await serverClosed;
  await api?.close();
  if (passed) await rm(run, { recursive: true });
  else console.error(`Diagnostic inputs preserved: ${run}`);
}
