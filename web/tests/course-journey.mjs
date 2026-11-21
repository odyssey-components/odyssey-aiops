// Course 2 uses a real Python-trained model and actual browser input events.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { join } from "node:path";

export async function verifyCourses({ page, root, web, run, uvRun, coreModule, legacyModelPath }) {
  const require = createRequire(join(root, "repositories/odyssey-design/packages/odyssey-web-testkit/package.json"));
  const { expect } = require("@playwright/test");
  const course = (id, surface = "sidebar") => page.locator(`[data-od-slot="pro-admin-navigation"][data-od-navigation-surface="${surface}"]`).getByRole("button",{name:({'fixed-map-v1':'第一课：固定地图','random-map-v1':'第二课：随机地图','local-map-v1':'第三课：局部探索','memory-map-v1':'第四课：模型记忆'})[id],exact:true});
  async function chooseCourse(id) {
    if (await course(id).isVisible()) await course(id).click();
    else {
      await page.locator('[data-od-slot="pro-admin-mobile-trigger"]').click();
      await course(id, "mobile").click();
      await expect(page.locator('[data-od-slot="pro-admin-mobile-drawer"]')).toBeHidden();
    }
  }
  const output = join(run, ".odyssey/odymaze/runs/random-trained");
  execFileSync("uv", [...uvRun, "odymaze-train", "--lesson", "random-map-v1", "--output", output],
    { cwd: root, timeout: 180000, stdio: "pipe" });
  const path = join(output, "model.json");
  const model = JSON.parse(await readFile(path, "utf8"));
  const report = JSON.parse(await readFile(join(output, "report.json"), "utf8"));
  assert.equal(model.lesson, "random-map-v1");
  assert.ok(report.final.success_rate > report.initial.success_rate);
  const reference = JSON.parse(execFileSync("uv", [...uvRun, "python", "-c", `
import json, sys, torch
from aiops_maze_training.lessons.fixed_map_v1.environment import State
from aiops_maze_training.lessons.random_map_v1.environment import generate_maze, samples
from aiops_maze_training.lessons.random_map_v1.model import make_model
data = json.load(open(sys.argv[1]))
model = make_model()
result = []
with torch.no_grad():
    for index, layer in zip((0,2), data['layers']):
        model[index].weight.copy_(torch.tensor(layer['weight']))
        model[index].bias.copy_(torch.tensor(layer['bias']))
    for seed in [0,1,2,3,4,5,6,7,42,99,100,123,999,2026,65535,999999,2000009,2000010,2147483647,4294967295]:
        maze = generate_maze(seed)
        rows = samples(maze)
        rows = rows[:4] + rows[-4:]
        for row in rows:
            row['probabilities'] = torch.softmax(model(torch.tensor(row['observation'])), dim=-1).tolist()
        result.append({'seed':seed,'maze':maze,'samples':rows})
print(json.dumps(result))
`, path], { cwd: root, timeout: 30000, encoding: "utf8", maxBuffer: 4 * 1024 * 1024 }));
  const bundle = await Bun.build({ entrypoints: [join(web, "src/lessons/random_map_v1.ts")], target: "browser" });
  assert.ok(bundle.success, String(bundle.logs));
  const randomModule = `data:text/javascript;base64,${Buffer.from(await bundle.outputs[0].text()).toString("base64")}`;
  const consistency = await page.evaluate(async ({ coreModule, randomModule, model, reference }) => {
    const core = await import(coreModule);
    const random = await import(randomModule);
    let maxError = 0;
    for (const item of reference) {
      const maze = random.generateMaze(item.seed);
      if (JSON.stringify(maze) !== JSON.stringify(item.maze)) throw new Error(`map mismatch seed ${item.seed}`);
      core.validateModel(model, maze);
      for (const row of item.samples) {
        if (JSON.stringify(random.encode(maze, row.state)) !== JSON.stringify(row.observation)) throw new Error("encoding mismatch");
        const probabilities = core.probabilities(model, maze, row.state);
        probabilities.forEach((value, i) => { maxError = Math.max(maxError, Math.abs(value - row.probabilities[i])); });
      }
    }
    return { maxError };
  }, { coreModule, randomModule, model, reference });
  assert.ok(consistency.maxError < 0.00001, JSON.stringify(consistency));

  await page.setViewportSize({ width: 1440, height: 1000 });
  await chooseCourse("random-map-v1");
  await expect(page.locator("#mode-manual")).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#maze-seed")).toBeVisible();
  await expect(page.locator("#encoding .bit")).toHaveCount(151);
  await page.getByRole("button", { name: "刷新模型列表" }).click();
  await expect(page.locator("#model-directory-status")).toHaveText("本课有 1 个模型。");
  await expect(page.locator("#current-model")).toHaveText("random-trained/model.json");
  await expect(page.locator("#import-message")).toHaveAttribute("data-kind", "success");
  await expect(page.locator("#mode-manual")).toHaveAttribute("aria-pressed", "true");
  await page.locator("#mode-model").click();

  // Use successful held-out episodes as reproducible demonstrations, not as tuning data.
  const solved = report.final.episodes.filter((episode) => episode.success).slice(0, 2);
  assert.equal(solved.length, 2);
  for (const episode of solved) {
    await page.locator("#maze-seed").fill(String(episode.seed));
    await page.locator("#new-maze").click();
    await expect(page.locator("#step-count")).toHaveText("0 / 100");
    await expect(page.locator("#mode-model")).toHaveAttribute("aria-pressed", "true");
    await page.getByRole("button", { name: "自动运行", exact: true }).click();
    await expect(page.locator("#game-status")).toHaveText("已到达出口", { timeout: 15000 });
    await expect(page.locator("#position")).toHaveText("(5, 5)");
    await expect(page.locator("#step-count")).toHaveText(`${episode.steps} / 100`);
  }
  await page.locator("#new-maze").click();
  await page.getByRole("button", { name: "自动运行", exact: true }).click();
  await chooseCourse("fixed-map-v1");
  await expect(page.locator("#step-count")).toHaveText("0 / 100");
  await page.waitForTimeout(160);
  await expect(page.locator("#step-count")).toHaveText("0 / 100");
  await expect(page.locator("#encoding .bit")).toHaveCount(18);

  // Cross-course models are not offered and URL attempts cannot change the lesson.
  await expect(page.locator("#current-model")).toContainText("zz-newest/model.json");
  await expect(page.locator("#model-list")).toHaveCount(1);
  const origin=new URL(page.url()).origin;
  await page.goto(origin+"/?course=fixed-map-v1&model=random-trained&mode=model");
  await expect(page.locator("#import-message")).toHaveAttribute("data-kind", "error");
  await expect(course("fixed-map-v1")).toHaveAttribute("aria-current", "page");
  await expect(page.locator("#mode-model")).toBeDisabled();
  await page.goto(origin+"/?course=fixed-map-v1&model=trained&mode=model");
  await expect(page.locator("#current-model")).toContainText("trained/model.json");
  await expect(page.locator("#mode-model")).toBeEnabled();

  await page.setViewportSize({ width: 390, height: 844 });
  await chooseCourse("random-map-v1");
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await expect(page.locator("#new-maze")).toBeEnabled();
  console.log(`PASS courses: legacy v1; 20 identical seeded maps; max probability error ${consistency.maxError}; 2 held-out maps solved; course switching, stale timers, rejection and narrow layout.`);
}
