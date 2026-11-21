import { root, product } from "../../scripts/workspace.mjs";
// Real shipped Go1 visual assets and WebGL, synthetic policy records/poses; no screenshots.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {createRequire} from 'node:module';
import {join} from 'node:path';
const web=join(product,'web');
const {chromium,expect}=createRequire(join(root,'repositories/odyssey-design/packages/odyssey-web-testkit/package.json'))('@playwright/test');
const server=spawn(join(web,'node_modules/.bin/vite'),['preview','--port','0'],{cwd:web,stdio:['ignore','pipe','inherit']});
const closed=once(server,'close');let resolveReady;const ready=new Promise(resolve=>{resolveReady=resolve;});server.stdout.on('data',chunk=>{const found=String(chunk).match(/http:\/\/127\.0\.0\.1:\d+/);if(found)resolveReady(found[0]);});
let browser;
const id='aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',sceneID='bbbbbbbb-cccc-dddd-eeee-ffffffffffff';
const scene={id:sceneID,title:'原始网格场景',kind:'turn',format:'go1-scene/v1',parameters:{angle_deg:90,road_width_m:.8,distance_m:1.2},createdAt:'2026-11-19T00:00:00Z'};
const frames=Array.from({length:3},(_,i)=>({time:i*.02,position_m:[i*.02,0,.3],orientation_wxyz:[1,0,0,0],joint_position_rad:i===0?Array(4).fill([0,.9,-1.8]).flat():Array(12).fill(i*.1),episode_end:i===2}));
const snapshot={tasks:[],models:[],scenes:[scene],environment:null,activity:{active:null,queued:[]},simulations:[{id,title:'网格回放',model:id,modelManifest:{},modelFiles:{'policy.npz':'a'.repeat(64)},scene,seed:7,steps:50,progress:3,state:'succeeded',startedAt:scene.createdAt,logs:'',result:{outcome:'timeout'}}]};
async function mockAPI(page){
 await page.route('**/api/models**',route=>route.fulfill({json:[]}));
 await page.route('**/api/go1**',route=>route.fulfill({json:new URL(route.request().url()).searchParams.has('simulation-replay')?{frames}:snapshot}));
}
try{
 const origin=await Promise.race([ready,new Promise((_,reject)=>setTimeout(()=>reject(Error('preview timeout')),20000).unref())]);
 browser=await chromium.launch({headless:true,args:['--enable-unsafe-swiftshader']});
 const page=await browser.newPage({viewport:{width:1400,height:1000}});page.setDefaultTimeout(15000);const errors=[];page.on('pageerror',error=>errors.push(error.message));await mockAPI(page);
 let downloads=0,fail=true;
 await page.route('**/models/go1/assets/*.stl',route=>{downloads++;if(fail&&route.request().url().endsWith('/trunk.stl')){fail=false;return route.fulfill({status:404,body:'synthetic failure'});}return route.continue();});
 await page.goto(origin);assert.equal(downloads,0,'Maze must not load robot meshes');
 await page.goto(`${origin}/?module=go1&go1Tab=scenes&go1Scene=${sceneID}`);
 await expect(page.getByRole('alert')).toContainText('网格 trunk 加载失败');
 await expect(page.getByRole('img',{name:'Go1 场景三维结构'})).toHaveCount(0);
 await page.getByRole('button',{name:'重试加载模型',exact:true}).click();
 const preview=page.getByRole('img',{name:'Go1 场景三维结构',exact:true});
 await expect(page.locator('[data-testid="go1-scene-viewport"]')).toHaveAttribute('data-ready','true');
 assert.ok(Number(await preview.getAttribute('data-triangles'))>100000,'original detailed meshes actually rendered');
 const loaded=downloads;assert.equal(loaded,10,'five assets retried after failure');
 await page.getByRole('button',{name:'返回场景列表'}).click();await expect(preview).toHaveCount(0);
 await page.getByRole('button',{name:'仿真任务',exact:true}).click();await page.getByRole('button',{name:'详情',exact:true}).click();await page.getByRole('button',{name:'查看仿真回放'}).click();
 const canvas=page.getByRole('img',{name:'Go1 机器人三维回放'}).locator('canvas');
 await expect(canvas).toHaveAttribute('data-frame-time','0');assert.ok(Number(await canvas.getAttribute('data-triangles'))>100000);
 assert.deepEqual(JSON.parse(await canvas.getAttribute('data-joints')),frames[0].joint_position_rad);assert.equal(downloads,loaded,'both pages reuse the same successful asset load');
 await page.getByRole('slider',{name:'评估回放位置'}).focus();await page.keyboard.press('End');await expect(canvas).toHaveAttribute('data-frame-time','0.04');assert.deepEqual(JSON.parse(await canvas.getAttribute('data-joints')),frames[2].joint_position_rad);
 await canvas.evaluate(element=>element.dispatchEvent(new Event('webglcontextlost',{cancelable:true})));await expect(page.getByRole('alert')).toContainText('图形上下文');await page.getByRole('button',{name:'重新创建视图'}).click();await expect(canvas).toHaveAttribute('data-frame-time','0.04');assert.equal(downloads,loaded);
 await page.setViewportSize({width:390,height:844});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
 const delayed=await browser.newPage();delayed.setDefaultTimeout(15000);delayed.on('pageerror',error=>errors.push(error.message));await mockAPI(delayed);
 let finished=0;delayed.on('requestfinished',request=>{if(request.url().includes('/models/go1/assets/'))finished++;});
 let release;const gate=new Promise(resolve=>{release=resolve;});const pending=[];
 await delayed.route('**/models/go1/assets/*.stl',route=>{const wait=gate.then(()=>route.continue()).catch(()=>{});pending.push(wait);return wait;});
 await delayed.goto(`${origin}/?module=go1&go1Tab=scenes&go1Scene=${sceneID}`);await expect(delayed.getByText('正在加载 Go1 三维模型…',{exact:true})).toBeVisible();
 await expect.poll(()=>pending.length).toBe(5);
 await delayed.getByRole('button',{name:'仿真任务',exact:true}).click();release();await Promise.all(pending);await expect.poll(()=>finished).toBe(5);await delayed.evaluate(()=>new Promise(requestAnimationFrame));
 await expect(delayed.getByRole('table',{name:'Go1 仿真任务列表'})).toBeVisible();await expect(delayed.locator('canvas')).toHaveCount(0);
 assert.deepEqual(errors,[]);console.log('PASS real Go1 STL rendering in both views, no Maze download, 404/retry, shared cache, recorded joints, context recovery, delayed-unmount isolation and mobile');
}finally{await browser?.close();server.kill();await closed;}
