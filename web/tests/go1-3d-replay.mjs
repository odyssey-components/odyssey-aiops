import { root, product } from "../../scripts/workspace.mjs";
// Controlled real WebGL with synthetic saved poses; no OS capture or training.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {mkdir,mkdtemp,readdir,readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {join} from 'node:path';
import {createInterface} from 'node:readline';
const web=join(product,'web');
const {chromium,expect}=createRequire(join(root,'repositories/odyssey-design/packages/odyssey-web-testkit/package.json'))('@playwright/test');
await mkdir(join(root,'tmp/aiops-replay3d'),{recursive:true});const run=await mkdtemp(join(root,'tmp/aiops-replay3d/run-'));
const engines=[];for(const name of await readdir(join(web,'dist/assets'))){if(name.endsWith('.js')&&(await readFile(join(web,'dist/assets',name),'utf8')).includes('THREE.WebGLRenderer'))engines.push(name);}assert.ok(engines.length);
const server=spawn(join(web,'node_modules/.bin/vite'),['preview','--port','0'],{cwd:web,stdio:['ignore','pipe','inherit']});const closed=once(server,'close');let browser;
const ids=['aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee','bbbbbbbb-cccc-dddd-eeee-ffffffffffff'];
const frames=Array.from({length:101},(_,i)=>({time:i*.02,position_m:[i*.01,0,.29+Math.sin(i*.1)*.015],orientation_wxyz:[Math.cos(i*.005),0,0,Math.sin(i*.005)],joint_position_rad:Array.from({length:12},(_,j)=>j%3===0?.05*Math.sin(i*.1):j%3===1?.8+.1*Math.sin(i*.1+j):-1.6-.15*Math.sin(i*.1+j)),episode_end:i===100}));
try {
 let origin;for await(const line of createInterface({input:server.stdout})){const match=line.match(/http:\/\/127\.0\.0\.1:\d+/);if(match){origin=match[0];break;}}assert.ok(origin);
 browser=await chromium.launch({headless:true,args:['--enable-unsafe-swiftshader']});const page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[],requests=[];page.on('pageerror',error=>errors.push(error.message));page.on('request',request=>requests.push(request.url()));
 let bad=false,empty=false,hold=false,release;const pending=[],gate=new Promise(resolve=>{release=resolve;});
 await page.route('**/api/go1**',route=>{
  const url=new URL(route.request().url());
  if(url.searchParams.has('replay')||url.searchParams.has('simulation-replay')){
   const deliver=()=>route.fulfill({json:{frames:empty?[]:bad?[{...frames[0],orientation_wxyz:[0,0,0,0]}]:url.searchParams.get('replay')===ids[1]?[frames[0]]:frames}});
   if(hold&&url.searchParams.get('replay')===ids[0]){const done=gate.then(()=>deliver()).catch(()=>{});pending.push(done);return done;}return deliver();
  }
  return route.fulfill({json:{tasks:[],scenes:[],simulations:[],environment:null,activity:{active:null,queued:[]},models:ids.map((id,index)=>({run:id,task:id,title:'三维模型 '+(index+1),createdAt:'2026-11-19T00:00:00Z',manifest:{}}))}});
 });
 await page.route('**/api/models**',route=>route.fulfill({json:[]}));
 await page.goto(origin+'/?course=fixed-map-v1');await page.waitForLoadState('networkidle');assert.ok(!requests.some(url=>engines.some(name=>url.endsWith('/'+name))),'Three.js remains unloaded on Maze');
 const base=origin+'/?module=go1&go1Tab=replay&go1Model=';
 await page.goto(base+ids[0]);const viewport=page.getByRole('img',{name:'Go1 机器人三维回放',exact:true}),canvas=viewport.locator('canvas');
 await expect(canvas).toHaveCount(1);await expect(canvas).toHaveAttribute('data-render-calls',/\d+/);assert.ok(Number(await canvas.getAttribute('data-triangles'))>100000);assert.ok(requests.some(url=>engines.some(name=>url.endsWith('/'+name))),'replay loads the actual Three.js engine');assert.deepEqual(JSON.parse(await canvas.getAttribute('data-body-position')),frames[0].position_m);
 const camera=await canvas.getAttribute('data-camera-position'),box=await canvas.boundingBox();assert.ok(box);await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.down();await page.mouse.move(box.x+box.width/2+90,box.y+box.height/2+40,{steps:8});await page.mouse.up();assert.notEqual(await canvas.getAttribute('data-camera-position'),camera);
 await page.getByRole('button',{name:'重置视角',exact:true}).click();await page.getByRole('combobox',{name:'回放倍速',exact:true}).click();await page.getByRole('option',{name:'2×',exact:true}).click();
 await page.getByRole('button',{name:'播放',exact:true}).click();await page.waitForFunction(()=>Number(document.querySelector('.go1-viewport canvas')?.dataset.frameTime)>.35);await page.getByRole('button',{name:'暂停',exact:true}).click();
 const paused=await canvas.getAttribute('data-frame-time');await page.waitForTimeout(120);assert.equal(await canvas.getAttribute('data-frame-time'),paused);
 await page.getByRole('slider',{name:'评估回放位置',exact:true}).focus();await page.keyboard.press('End');await expect(canvas).toHaveAttribute('data-frame-time','2');assert.deepEqual(JSON.parse(await canvas.getAttribute('data-joints')),frames.at(-1).joint_position_rad);await expect(page.getByText('2.00 / 2.00 秒 · 回合结束',{exact:true})).toBeVisible();
 await page.getByRole('button',{name:'适配轨迹',exact:true}).click();await expect(page.getByRole('button',{name:'跟随机器人',exact:true})).toHaveAttribute('aria-pressed','false');await page.getByRole('button',{name:'重置',exact:true}).click();await expect(canvas).toHaveAttribute('data-frame-time','0');await page.getByRole('button',{name:'跟随机器人',exact:true}).click();
 await page.getByRole('button',{name:'重置视角',exact:true}).click();await page.evaluate(()=>window.scrollTo(0,0));
 await page.screenshot({path:join(run,'replay-desktop.png'),fullPage:true});await page.setViewportSize({width:390,height:844});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:join(run,'replay-mobile.png'),fullPage:true});
 await page.getByRole('combobox',{name:'评估模型',exact:true}).click();await page.getByRole('option',{name:'三维模型 2',exact:true}).click();await expect(canvas).toHaveAttribute('data-path-segments','0');await expect(page.getByRole('button',{name:'播放',exact:true})).toBeDisabled();
 bad=true;await page.goto(base+ids[0]);await expect(page.getByRole('alert')).toContainText('朝向无效');await expect(page.locator('.go1-viewport canvas')).toHaveCount(0);bad=false;empty=true;await page.reload();await expect(page.getByRole('alert')).toContainText('没有有效');empty=false;
 hold=true;const requested=page.waitForRequest(request=>request.url().includes('replay='+ids[0]));await page.reload({waitUntil:'domcontentloaded'});await requested;await page.getByRole('combobox',{name:'评估模型',exact:true}).click();await page.getByRole('option',{name:'三维模型 2',exact:true}).click();await expect(canvas).toHaveAttribute('data-path-segments','0');release();await Promise.all(pending);await expect(page.getByRole('button',{name:'播放',exact:true})).toBeDisabled();hold=false;
 await page.goto(`${origin}/?module=go1&go1Tab=replay&go1Simulation=${ids[0]}`);await expect(page.getByRole('heading',{name:/仿真回放/})).toBeVisible();await expect(canvas).toHaveAttribute('data-path-segments','100');
 await page.getByRole('slider',{name:'评估回放位置',exact:true}).focus();await page.keyboard.press('End');await expect(canvas).toHaveAttribute('data-frame-time','2');
 await canvas.evaluate(node=>node.dispatchEvent(new Event('webglcontextlost',{cancelable:true})));await expect(page.getByRole('alert')).toContainText('图形上下文');await page.getByRole('button',{name:'重新创建视图',exact:true}).click();await expect(page.getByRole('alert')).toHaveCount(0);await expect(canvas).toHaveAttribute('data-render-calls',/\d+/);await expect(canvas).toHaveAttribute('data-frame-time','2');
 await page.addInitScript(()=>{const getContext=HTMLCanvasElement.prototype.getContext;window.__restoreGo1Context=()=>{HTMLCanvasElement.prototype.getContext=getContext;};HTMLCanvasElement.prototype.getContext=function(kind,...args){return kind==='webgl2'?null:getContext.call(this,kind,...args);};});
 await page.reload();await expect(page.getByRole('alert')).toContainText('无法创建三维视图');await expect(canvas).toHaveCount(0);await page.evaluate(()=>window.__restoreGo1Context());await page.getByRole('button',{name:'重新创建视图',exact:true}).click();await expect(canvas).toHaveAttribute('data-render-calls',/\d+/);
 await page.goto(origin+'/?module=go1');await expect(page.locator('.go1-viewport canvas')).toHaveCount(0);assert.deepEqual(errors,[]);
 console.log('PASS actual WebGL/joints, orbit, clock/speed/pause/seek, fit/follow, single/invalid/empty/stale loads, simulation, context recovery, unmount, mobile. Screens:',run);
} finally {await browser?.close();server.kill();await closed;}
