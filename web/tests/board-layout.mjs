import { root } from "../../scripts/workspace.mjs";
// Product-owned layout regression: real Board markup and maze CSS, no OS capture.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { renderToStaticMarkup } from "react-dom/server";
import { createElement } from "react";
import { Board } from "../src/components/maze/Board";
import maze from "../maze.json";


const require = createRequire(`${root}/repositories/odyssey-design/packages/odyssey-web-testkit/package.json`);
const { chromium } = require("@playwright/test");
const css = await readFile(new URL("../maze.css", import.meta.url), "utf8");
const browser = await chromium.launch({ headless: true });
let count = 0;
try {
  const page = await browser.newPage();
  for (const width of [390, 820, 1440]) {
    await page.setViewportSize({ width, height: 844 });
    for (const [x, y, status] of [[1, 5, "success"], [1, 1, "ready"]]) {
      for (let direction = 0; direction < 4; direction++) {
        const html = renderToStaticMarkup(createElement(Board, {
          maze, state: { x, y, direction, status, steps: 0 }, visited: new Set([`${x},${y}`]),
        }));
        await page.setContent(`<style>*{box-sizing:border-box}body{margin:0;padding:16px}${css}</style>${html}`);
        const geometry = await page.locator(".agent").evaluate((agent) => {
          const tile = agent.parentElement.getBoundingClientRect();
          const body = agent.getBoundingClientRect();
          const svg = agent.querySelector("svg");
          const arrow = svg?.getBoundingClientRect();
          const matrix = svg?.getScreenCTM();
          const tip = matrix && new DOMPoint(21, 12).matrixTransform(matrix);
          const tail = matrix && new DOMPoint(8, 12).matrixTransform(matrix);
          return {
            width: body.width, height: body.height,
            centerX: body.x + body.width / 2 - tile.x - tile.width / 2,
            centerY: body.y + body.height / 2 - tile.y - tile.height / 2,
            contained: body.left >= tile.left && body.top >= tile.top && body.right <= tile.right && body.bottom <= tile.bottom,
            svg: Boolean(svg), arrowWidth: arrow?.width,
            arrowCenterX: arrow && arrow.x + arrow.width / 2 - body.x - body.width / 2,
            arrowCenterY: arrow && arrow.y + arrow.height / 2 - body.y - body.height / 2,
            vector: tip && tail ? [tip.x - tail.x, tip.y - tail.y] : null,
          };
        });
        const context = `${width}px (${x},${y}) heading ${direction}: ${JSON.stringify(geometry)}`;
        assert.ok(Math.abs(geometry.width - geometry.height) < .1, `circle distorted: ${context}`);
        assert.ok(Math.abs(geometry.centerX) < .1 && Math.abs(geometry.centerY) < .1, `off-center: ${context}`);
        assert.ok(geometry.contained, `overflows tile: ${context}`);
        assert.ok(geometry.svg && geometry.arrowWidth >= geometry.width * .6, `missing or undersized SVG: ${context}`);
        assert.ok(Math.abs(geometry.arrowCenterX) < .1 && Math.abs(geometry.arrowCenterY) < .1, `arrow off-center: ${context}`);
        const [dx, dy] = geometry.vector;
        const [wantX, wantY] = [[1, 0], [0, 1], [-1, 0], [0, -1]][direction];
        assert.ok(dx * wantX + dy * wantY > 0 && Math.abs(dx * wantY - dy * wantX) < .1, `wrong arrow heading: ${context}`);
        count++;
      }
    }
  }
  console.log(`PASS: ${count} board layouts; start/goal, four SVG headings, centered circles and tile containment at three viewport widths.`);
} finally {
  await browser.close();
}
