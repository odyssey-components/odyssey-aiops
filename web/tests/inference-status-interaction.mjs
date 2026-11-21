import { root, product } from "../../scripts/workspace.mjs";
// Product-owned headless interactions with synthetic API state; no screenshots or real data.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {createRequire} from 'node:module';
import {join} from 'node:path';
const web=join(product,'web');
const {chromium,expect}=createRequire(join(root,'repositories/odyssey-design/packages/odyssey-web-testkit/package.json'))('@playwright/test');
const server=spawn(join(web,'node_modules/.bin/vite'),['--port','0'],{cwd:web,stdio:['ignore','pipe','inherit']});
const closed=once(server,'close');
let resolveOrigin;const ready=new Promise(resolve=>{resolveOrigin=resolve;});
server.stdout.on('data',chunk=>{const found=String(chunk).match(/http:\/\/127\.0\.0\.1:\d+/);if(found) resolveOrigin(found[0]);});
let browser;
try {
 const origin=await Promise.race([ready,new Promise((_,reject)=>setTimeout(()=>reject(new Error('preview timeout')),20000).unref())]);
 browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:390,height:844}});
 const errors=[];page.on('pageerror',error=>errors.push(error.message));
 const maze=JSON.parse(await (await import('node:fs/promises')).readFile(join(web,'maze.json'),'utf8'));
 const policy={format:'odymaze-policy/v1',lesson:'fixed-map-v1',encoding:'one-hot-xy-direction/v1',actions:['left','right','forward'],maze,layers:[],metadata:{seed:7,epochs:0,samples:0,accuracy:0,success:false}};
 await page.route(/\/src\/utilities\/onnx-runtime\.ts(?:\?.*)?$/,route=>route.fulfill({contentType:'text/javascript',body:`
  export async function loadOnnxModel() {return {policy:${JSON.stringify(policy)},
   async predict() {await new Promise(resolve=>setTimeout(resolve,120));if(window.failNextPrediction)throw Error('synthetic prediction failure');return {values:[.1,.1,.8],hidden:null};},
   async dispose() {}};}
 `}));
 await page.route('**/api/models*',route=>route.fulfill({json:[{run:'synthetic',lesson:'fixed-map-v1'}]}));
 await page.route('**/api/training**',route=>route.fulfill({json:{datasets:[],job:null,active:null,jobs:[]}}));
 await page.goto(`${origin}/?model=synthetic&mode=model&speed=40`);
 await expect(page.locator('#mode-model')).toHaveAttribute('aria-pressed','true');
 await expect(page.locator('#auto')).toBeEnabled();
 await page.evaluate(()=>{
  const model=document.querySelector('#mode-model');
  window.originalModeButton=model;
  window.flicker={texts:[],disabled:[],busy:[],stale:[],hints:[]};
  const record=()=>{
   window.flicker.texts.push(document.querySelector('#running-state').textContent);
   window.flicker.disabled.push(model.hasAttribute('data-disabled'));
   window.flicker.busy.push(document.querySelector('#running-state').getAttribute('aria-busy'));
   window.flicker.stale.push(document.querySelector('#probabilities').getAttribute('data-stale'));
   window.flicker.hints.push(document.querySelector('.decision-card').textContent.includes('更新中'));
  };
  window.flickerObserver=new MutationObserver(record);window.flickerObserver.observe(document.querySelector('main'),{subtree:true,childList:true,characterData:true,attributes:true});
 });
 await page.locator('#auto').click();
 await expect.poll(()=>page.locator('#step-count').textContent()).toMatch(/^[4-9] \/ 100$/);
 await page.locator('#auto').click();
 await expect(page.locator('#running-state')).toHaveAttribute('aria-busy','false');
 const recorded=await page.evaluate(()=>{window.flickerObserver.disconnect();return {...window.flicker,sameButton:window.originalModeButton===document.querySelector('#mode-model')};});
 assert.ok(recorded.busy.includes('true') && recorded.busy.includes('false'));
 assert.ok(recorded.stale.includes('true') && recorded.stale.includes('false'));
 assert.ok(recorded.texts.every(text=>text==='自动运行中' || text==='已暂停'));
 assert.ok(recorded.disabled.every(value=>!value));assert.ok(recorded.hints.every(value=>!value));assert.ok(recorded.sameButton);
 await page.evaluate(()=>{window.failNextPrediction=true;});await page.locator('#single-step').click();
 await expect(page.getByRole('alert')).toContainText('synthetic prediction failure');
 await expect(page.locator('#mode-model')).toBeDisabled();
 assert.deepEqual(errors,[]);
 console.log('PASS repeated delayed steps keep status text, probability heading and selected model stable; failures still disable model');
} finally {await browser?.close();server.kill('SIGTERM');await closed;}
