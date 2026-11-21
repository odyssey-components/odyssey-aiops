import { root, product } from "../../scripts/workspace.mjs";
// Narrow inference regression using synthetic models; no screenshots or user writes.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {createRequire} from 'node:module';
import {readFile,readdir} from 'node:fs/promises';
import {join} from 'node:path';
import TOML from '@iarna/toml';
const web=join(product,'web');
const fixtures=process.argv[2];assert.ok(fixtures,'pass isolated synthetic runs directory');
const models=new Map();
for(const name of await readdir(fixtures)) {
 const manifest=TOML.parse(await readFile(join(fixtures,name,'model.manifest.toml'),'utf8'));
 models.set(manifest.lesson,{manifest,binary:await readFile(join(fixtures,name,'model.onnx'))});
}
assert.equal(models.size,4);
const {chromium}=createRequire(join(root,'repositories/odyssey-design/packages/odyssey-web-testkit/package.json'))('@playwright/test');
const server=spawn(join(web,'node_modules/.bin/vite'),['--port','0'],{cwd:web,stdio:['ignore','pipe','inherit']});
const closed=once(server,'close');let resolveOrigin;
const ready=new Promise(resolve=>{resolveOrigin=resolve;});
server.stdout.on('data',chunk=>{const match=String(chunk).match(/http:\/\/127\.0\.0\.1:\d+/);if(match)resolveOrigin(match[0]);});
let browser;
try {
 const origin=await Promise.race([ready,new Promise((_,reject)=>setTimeout(()=>reject(Error('dev timeout')),20000).unref())]);
 browser=await chromium.launch({headless:true});const page=await browser.newPage();
 await page.route('**/api/models*',route=>route.fulfill({json:[]}));
 await page.route('**/api/models/**',route=>{
  const url=new URL(route.request().url()),model=models.get(decodeURIComponent(url.pathname.split('/').at(-1)));
  return model?route.fulfill({body:url.searchParams.get('artifact')==='manifest'?JSON.stringify(model.manifest):model.binary}):route.fulfill({status:404});
 });
 await page.goto(origin);
 for(const [lesson,width] of [['fixed-map-v1',18],['random-map-v1',151],['local-map-v1',249],['memory-map-v1',40]]) {
  const result=await page.evaluate(async({lesson,width})=>{
   const {loadOnnxModel}=await import('/src/utilities/onnx-runtime.ts');
   const model=await loadOnnxModel(lesson);
   try {return await model.predict(Array(width).fill(0));} finally {await model.dispose();}
  },{lesson,width});
  assert.equal(result.values.length,3);assert.ok(result.values.every(Number.isFinite));
  assert.ok(Math.abs(result.values.reduce((a,b)=>a+b,0)-1)<1e-6);
  if(lesson==='memory-map-v1') {assert.equal(result.hidden.length,32);assert.ok(result.hidden.every(Number.isFinite));}
  else assert.equal(result.hidden,null);
  console.log('PASS TOML references → real ONNX WASM inference',lesson);
 }
} finally {await browser?.close();server.kill('SIGTERM');await closed;}
