import { root, product } from "../../scripts/workspace.mjs";
import TOML from "@iarna/toml";
// Product-controlled dev-mode resource and real WASM inference regression, without capture.
import assert from 'node:assert/strict';
import {startGoAPI} from './go-api.mjs';
import {spawn,execFileSync} from 'node:child_process';
import {once} from 'node:events';
import {createRequire} from 'node:module';
import {readFile,mkdir,mkdtemp} from 'node:fs/promises';
import {join,resolve} from 'node:path';
const web=join(product,'web');
const {chromium,expect}=createRequire(join(root,'repositories/odyssey-design/packages/odyssey-web-testkit/package.json'))('@playwright/test');
const fixture=resolve(process.argv[2] ?? join(root,'tmp/onnx-loader-fix/model'));
const manifest=await readFile(join(fixture,'model.manifest.toml'));
const binary=await readFile(join(fixture,'model.onnx'));
const reference=JSON.parse(execFileSync('uv',['run','--project','.', '--no-sync','python','-c',
 'import onnx,json,math,sys; m=onnx.load(sys.argv[1]); print(json.dumps(dict(producer=m.producer_name,nodes=len(m.graph.node),weights=len(m.graph.initializer),parameters=sum(math.prod(t.dims) for t in m.graph.initializer))))',join(fixture,'model.onnx')],{cwd:root,encoding:'utf8'}));
await mkdir(join(root,'tmp/onnx-loader-fix'),{recursive:true});
const syntheticHome=await mkdtemp(join(root,'tmp/onnx-loader-fix/dev-home-'));
const api=await startGoAPI(root,syntheticHome,process.env.HOME+'/.odyssey/odymaze/python/bin/python');
const server=spawn(join(web,'node_modules/.bin/vite'),['--port','0'],{cwd:web,env:{...process.env,HOME:syntheticHome,ODYMAZE_API_ORIGIN:api.origin},stdio:['ignore','pipe','inherit']});
const closed=once(server,'close');let resolveOrigin;
const ready=new Promise(resolve=>{resolveOrigin=resolve;});
server.stdout.on('data',chunk=>{const found=String(chunk).match(/http:\/\/127\.0\.0\.1:\d+/);if(found) resolveOrigin(found[0]);});
let browser;
try {
 const origin=await Promise.race([ready,new Promise((_,reject)=>setTimeout(()=>reject(new Error('dev server timeout')),20000).unref())]);
 const source=await (await fetch(origin+'/src/utilities/onnx-runtime.ts')).text();
 assert.ok(!source.includes('/deps/odymaze-ort-loader'), 'loader?url must be an asset URL, not an optimized JS dependency');
 browser=await chromium.launch({headless:true});const page=await browser.newPage();page.setDefaultTimeout(15000);
 let failManifest=false;
 await page.route('**/api/models/**',route=>route.fulfill(failManifest?{status:500,body:'synthetic failure'}:{body:route.request().url().includes('artifact=manifest')?JSON.stringify(TOML.parse(manifest.toString())):binary}));
 await page.goto(origin);await page.waitForLoadState('networkidle');
 const result=await page.evaluate(async()=>{
  const {loadOnnxModel}=await import('/src/utilities/onnx-runtime.ts');
  const model=await loadOnnxModel('synthetic');
  try {return await model.predict(Array(18).fill(0));} finally {await model.dispose();}
 });
 assert.equal(result.values.length,3);assert.ok(result.values.every(Number.isFinite));
 assert.ok(Math.abs(result.values.reduce((a,b)=>a+b,0)-1)<1e-6);assert.equal(result.hidden,null);
 // Detail browsing fetches metadata only and does not choose an inference model.
 await page.goto(origin+'/?tab=models&modelDetail=synthetic');
 await expect(page.getByRole('region',{name:'模型详情',exact:true})).toContainText('训练轮数');
 await expect(page.getByRole('region',{name:'模型详情',exact:true})).toContainText('observations · float32 · [batch, 18]');
 await expect(page.locator('#onnx-structure')).toContainText(reference.producer);
 const counts=await page.locator('#onnx-structure dl').first().locator('dd').allTextContents();
 assert.equal(counts.at(-3),String(reference.nodes));assert.equal(counts.at(-2),String(reference.weights));
 assert.equal(counts.at(-1),reference.parameters.toLocaleString('zh-CN'));
 await page.getByText(`权重张量（${reference.weights}）`,{exact:true}).click();
 await expect(page.getByRole('table',{name:'ONNX 权重张量',exact:true}).getByRole('row')).toHaveCount(reference.weights+1);
 await page.getByText(`主图节点与连接（${reference.nodes}）`,{exact:true}).click();
 await expect(page.getByRole('table',{name:'ONNX 节点连接',exact:true}).getByRole('row')).toHaveCount(reference.nodes+1);
 assert.equal(new URL(page.url()).searchParams.has('model'),false);
 await page.reload();await expect(page.getByRole('heading',{name:'模型详情',exact:true})).toBeVisible();
 await page.getByRole('button',{name:'返回模型列表',exact:true}).click();
 await expect(page.getByRole('table',{name:'本课模型列表'})).toBeVisible();
 await page.goto(origin+'/?course=random-map-v1&tab=models&modelDetail=synthetic');
 await expect(page.getByRole('alert')).toContainText('模型不属于当前课时');
 failManifest=true;
 await page.goto(origin+'/?tab=models&modelDetail=synthetic');
 await expect(page.getByRole('alert')).toContainText('无法读取 ONNX manifest');
 failManifest=false;
 await page.getByRole('button',{name:'重试',exact:true}).click();
 await expect(page.getByRole('region',{name:'模型详情',exact:true})).toContainText('训练轮数');
 // Render a pending frame between real probability frames; DOM bars must survive.
 async function render(frame) {
  await page.evaluate(async frame=>{const fixture=await import('/tests/decisions-fixture.tsx');fixture.renderPrediction(frame);},frame);
 }
 await render({values:[.2,.1,.7],pending:false,steps:0});
 await page.evaluate(()=>{window.previousBar=document.querySelector('#prediction-fixture .probability-row');});
 await render({values:null,pending:true,steps:1});
 await expect(page.locator('#prediction-fixture .probability-row')).toHaveCount(3);
 assert.equal(await page.evaluate(()=>window.previousBar===document.querySelector('#prediction-fixture .probability-row')),true);
 await expect(page.locator('#probabilities')).toHaveAttribute('data-stale','true');
 await expect(page.locator('#probabilities')).toContainText('70.0%');
 await render({values:[.1,.8,.1],pending:false,steps:1});
 await expect(page.locator('#probabilities')).toHaveAttribute('data-stale','false');
 await expect(page.locator('#probabilities')).toContainText('80.0%');
 await render({values:null,pending:true,steps:0});
 await expect(page.locator('#prediction-fixture .probability-row')).toHaveCount(0);
 await render({values:[.2,.1,.7],pending:false,steps:1});
 await render({values:null,pending:true,steps:2,identity:1});
 await expect(page.locator('#prediction-fixture .probability-row')).toHaveCount(0);
 console.log('PASS ONNX binary fields/counts match Python onnx; weight and node tables expand');
 console.log('PASS model metadata details/return/reload/course guard and probability bar DOM continuity/reset/model isolation');
 console.log('PASS dev loader stays a URL; actual ONNX WASM loads, predicts finite probabilities and disposes');
} finally {await browser?.close();server.kill();await closed;await api.close();}
