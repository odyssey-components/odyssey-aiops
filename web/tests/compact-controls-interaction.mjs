import { root, product } from "../../scripts/workspace.mjs";
// Product-owned headless interactions with synthetic API state; no screenshots or real data.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {createRequire} from 'node:module';
import {join} from 'node:path';
const web=join(product,'web');
const {chromium,expect}=createRequire(join(root,'repositories/odyssey-design/packages/odyssey-web-testkit/package.json'))('@playwright/test');
const server=spawn(join(web,'node_modules/.bin/vite'),['preview','--port','0'],{cwd:web,stdio:['ignore','pipe','inherit']});
const closed=once(server,'close');
let resolveOrigin;const ready=new Promise(resolve=>{resolveOrigin=resolve;});
server.stdout.on('data',chunk=>{const found=String(chunk).match(/http:\/\/127\.0\.0\.1:\d+/);if(found) resolveOrigin(found[0]);});
let browser;
try {
 const origin=await Promise.race([ready,new Promise((_,reject)=>setTimeout(()=>reject(new Error('preview timeout')),20000).unref())]);
 browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:390,height:844}});
 const errors=[];page.on('pageerror',error=>errors.push(error.message));
 const course='random-map-v1',id='12345678-1234-1234-1234-123456789abc';
 const dataset={id,course,title:'合成地图组',description:'测试地图',seed:7,maps:16,validation:64,test:100,createdAt:'2026-11-19T02:03:04Z',sha256:'a'.repeat(64)};
 const metrics=[{epoch:0,total:2,successRate:.125,evaluatedMaps:64,split:'validation'},{epoch:2,total:2,successRate:.5,evaluatedMaps:64,split:'validation'}];
 const job={id,course,action:'train',state:'succeeded',startedAt:dataset.createdAt,logs:'epoch 2/2',metrics};
 await page.route('**/api/training**',route=>route.fulfill({json:{datasets:[dataset],job,active:null}}));
 await page.route('**/api/models',route=>route.fulfill({json:[]}));
 await page.goto(`${origin}/?course=${course}`);
 const seed=page.getByRole('spinbutton',{name:'地图种子',exact:true});
 await expect(seed).toBeVisible();
 const compact=await page.locator('.seed-controls').evaluate(element=>({height:element.getBoundingClientRect().height,inputWidth:element.querySelector('input').getBoundingClientRect().width,card:element.closest('[data-od-slot="card"]')!==null}));
 assert.ok(compact.height<=48);assert.ok(compact.inputWidth<=180);assert.equal(compact.card,false);
 await expect(page.locator('#model-info')).toHaveCount(0);
 await seed.fill('42');await page.locator('#new-maze').click();
 await expect(page).toHaveURL(/seed=42/);
 await seed.fill('-1');await page.locator('#new-maze').click();
 await expect(page.getByRole('alert')).toContainText('种子必须是');
 assert.deepEqual(errors,[]);
 console.log('PASS compact seed input, generation, URL update, removed descriptions and retained errors');
} finally {await browser?.close();server.kill();await closed;}
