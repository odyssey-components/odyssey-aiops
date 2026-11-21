// Verify real Router navigation rather than assigning React state from a test.
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { join } from "node:path";
import { rename } from "node:fs/promises";

export async function verifyUrlState({ page, root, origin, legacyModelPath }) {
  const require = createRequire(join(root, "repositories/odyssey-design/packages/odyssey-web-testkit/package.json"));
  const { expect } = require("@playwright/test");
  const selectedCourse = (id) => page.locator('[data-od-slot="pro-admin-navigation"][data-od-navigation-surface="sidebar"]').getByRole("button",{name:({'fixed-map-v1':'第一课：固定地图','random-map-v1':'第二课：随机地图','local-map-v1':'第三课：局部探索','memory-map-v1':'第四课：模型记忆'})[id],exact:true});
  await page.setViewportSize({ width: 1440, height: 1000 });

  const direct = `${origin}/?course=random-map-v1&point=3&seed=42&mode=expert&speed=40&keep=learning#notes`;
  await page.goto(direct);
  await page.waitForLoadState("networkidle");
  await expect(selectedCourse("random-map-v1")).toHaveAttribute("aria-current", "page");
  await expect(page.locator("#maze-seed")).toHaveValue("42");
  await expect(page.locator("#lesson-title")).toHaveText("理解模型");
  await expect(page.locator("#mode-expert")).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#speed")).toContainText("快速");
  await expect(page.locator("#running-state")).toHaveText("已暂停");
  await page.reload();
  await page.waitForLoadState("networkidle");
  await expect(page.locator("#lesson-title")).toHaveText("理解模型");
  await expect(page.locator("#maze-seed")).toHaveValue("42");
  await expect(page.locator("#mode-expert")).toHaveAttribute("aria-pressed", "true");

  await page.locator('[data-lesson="4"]').click();
  await expect(page).toHaveURL(/point=4/);
  await expect(page.locator("#lesson-title")).toHaveText("开始训练");
  await page.goBack();
  await expect(page.locator("#lesson-title")).toHaveText("理解模型");
  await page.goForward();
  await expect(page.locator("#lesson-title")).toHaveText("开始训练");
  await page.locator("#maze-seed").fill("99");
  await page.locator("#new-maze").click();
  await expect(page).toHaveURL(/seed=99/);
  await expect(page.locator("#step-count")).toHaveText("0 / 100");
  await page.goBack();
  await expect(page.locator("#maze-seed")).toHaveValue("42");
  await expect(page.locator("#step-count")).toHaveText("0 / 100");
  assert.equal(new URL(page.url()).searchParams.get("keep"), "learning");
  assert.equal(new URL(page.url()).hash, "#notes");

  await page.getByRole("button", { name: "自动运行", exact: true }).click();
  await expect(page.locator("#step-count")).not.toHaveText("0 / 100");
  const runningUrl = page.url();
  assert.equal(new URL(runningUrl).searchParams.has("running"), false);
  await page.reload();
  await page.waitForLoadState("networkidle");
  await expect(page.locator("#running-state")).toHaveText("已暂停");
  await expect(page.locator("#step-count")).toHaveText("0 / 100");

  // The selected model is restored from its fixed-directory name.
  await page.goto(`${origin}/?course=fixed-map-v1&point=5&mode=model&speed=40&model=trained`);
  await page.waitForLoadState("networkidle");
  await expect(page.locator("#model-info")).toContainText("trained/model.json");
  await expect(page.locator("#mode-model")).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: "自动运行", exact: true }).click();
  await expect(page.locator("#game-status")).toHaveText("已到达出口");
  await page.reload();
  await page.waitForLoadState("networkidle");
  await expect(page.locator("#mode-model")).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#step-count")).toHaveText("0 / 100");
  await expect(page.locator("#running-state")).toHaveText("已暂停");
  await page.locator("#refresh-models").click();
  await expect(page.locator("#model-info")).toContainText("zz-newest/model.json");
  await expect.poll(() => new URL(page.url()).searchParams.get("model")).toBe("zz-newest");
  // A real history transition must not keep using a different model if the target disappeared.
  await rename(legacyModelPath, `${legacyModelPath}.saved`);
  try {
    await page.goBack();
    await expect(page.locator("#import-message")).toHaveAttribute("data-kind", "error");
    await expect(page.locator("#mode-model")).toBeDisabled();
  } finally {
    await rename(`${legacyModelPath}.saved`, legacyModelPath);
  }
  await page.reload();
  await page.waitForLoadState("networkidle");
  await expect(page.locator("#mode-model")).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#current-model")).toContainText("trained/model.json");
  await expect(page.locator("#model-file")).toHaveCount(0);

  await page.goto(`${origin}/?course=random-map-v1&point=2&seed=42&mode=model&model=missing-run`);
  await page.waitForLoadState("networkidle");
  await expect(page.locator("#import-message")).toHaveAttribute("data-kind", "error");
  await expect(selectedCourse("random-map-v1")).toHaveAttribute("aria-current", "page");
  await expect(page.locator("#mode-model")).toBeDisabled();
  await page.goto(`${origin}/?course=future&point=-1&seed=NaN&mode=bad&speed=999`);
  await page.waitForLoadState("networkidle");
  await expect(selectedCourse("fixed-map-v1")).toHaveAttribute("aria-current", "page");
  await expect(page.locator("#lesson-title")).toHaveText("认识环境");
  await expect(page.locator("#mode-manual")).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#speed")).toContainText("慢速");
  console.log("PASS URL: direct link, SSR/reload, UI history/back/forward, seed reset, unrelated search/hash, safe pause, named model restore, automatic first-model refresh, invalid values and missing model.");
}
