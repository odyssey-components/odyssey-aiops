import { root, product } from "../../scripts/workspace.mjs";
// Controlled product test: inspect static bootstrap before SPA mounting, then restored appearance.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createRequire } from 'node:module';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';

const web = fileURLToPath(new URL('../', import.meta.url));
const { chromium } = createRequire(`${root}/repositories/odyssey-design/packages/odyssey-web-testkit/package.json`)('@playwright/test');
const server = spawn(`${web}/node_modules/.bin/vite`, ['preview', '--port', '0'], {cwd:web, stdio:['ignore','pipe','inherit'], timeout:60000});
const closed = once(server, 'close');
let browser;
try {
  let origin;
  for await (const line of createInterface({input:server.stdout})) {
    const match = line.match(/http:\/\/127\.0\.0\.1:\d+/);
    if (match) { origin=match[0]; break; }
  }
  assert.ok(origin);
  browser = await chromium.launch({headless:true});
  for (const [saved, system] of [['dark','light'], ['system','dark'], ['light','dark']]) {
    const context = await browser.newContext({colorScheme:system});
    await context.addInitScript(({saved}) => localStorage.setItem('odymaze-web-appearance', JSON.stringify({colorScheme:saved})), {saved});
    const page = await context.newPage();
    const errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    let release;
    const gate=new Promise(resolve=>{release=resolve;});
    await page.route('**/*.js', async route=>{
      if (!new URL(route.request().url()).pathname.startsWith('/od-appearance-')) await gate;
      await route.continue();
    });
    await page.route('**/api/models?catalog=1', route=>route.fulfill({json:[]}));
    await page.route('**/api/training?*', route=>route.fulfill({json:{tasks:[],datasets:[]}}));
    const navigation=page.goto(origin, {waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>document.documentElement.dataset.odAppearancePending==='true');
    const initial=await page.evaluate(()=>({
      scheme:document.documentElement.dataset.odAppearanceScheme,
      pending:document.documentElement.dataset.odAppearancePending,
      mounted:!!document.querySelector('[data-od-root]'),
    }));
    const expected=saved==='system'?system:saved;
    assert.equal(initial.scheme,expected);
    assert.equal(initial.pending,'true');
    assert.equal(initial.mounted,false,'SPA must wait for application modules after synchronous theme restoration');
    release();
    await navigation;
    await page.waitForFunction(()=>!document.documentElement.dataset.odAppearancePending);
    assert.equal(await page.locator('[data-od-root]').getAttribute('data-od-resolved-color-scheme'),expected);
    assert.equal(await page.locator('[data-od-root]').evaluate(node=>getComputedStyle(node).visibility),'visible');
    assert.deepEqual(errors,[]);
    await context.close();
  }
  console.log('Appearance: saved dark, system dark, explicit light static bootstrap/SPA restoration passed.');
} finally {
  await browser?.close();
  server.kill();
  await closed;
}
