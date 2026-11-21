import { root, product } from "../../scripts/workspace.mjs";
// Product-owned targeted interactions: synthetic API, no screenshots or real writes.
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
server.stdout.on('data',chunk=>{const match=String(chunk).match(/http:\/\/127\.0\.0\.1:\d+/);if(match)resolveOrigin(match[0]);});
let browser;
try {
 const origin=await Promise.race([ready,new Promise((_,reject)=>setTimeout(()=>reject(Error('preview timeout')),20000).unref())]);
 browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:1440,height:900}});
 page.setDefaultTimeout(10000);
 const errors=[];page.on('pageerror',error=>errors.push(error.message));
 const scenes=[
  ['flat','空旷场地',{forward_m_s:.3,yaw_rate_rad_s:0},0],
  ['turn','拐弯',{distance_m:1.2,road_width_m:.8,angle_deg:90},4],
  ['uturn','窄路掉头',{turn_at_m:1.2,road_width_m:.8},3],
  ['slope','越过小坡',{length_m:1.2,angle_deg:8},2],
  ['duck','俯身穿过横梁',{distance_m:1.2,height_m:.28},3],
  ['jump','跳过低横梁',{distance_m:1.2,height_m:.2},1],
 ].map(([kind,title,parameters,boxes],i)=>({id:`a10f5001-0001-4000-8000-${String(i+1).padStart(12,'0')}`,format:'go1-scene/v1',title,kind,parameters,boxes,createdAt:'2026-11-19T00:00:00Z'}));
 let reads=0,writes=0;
 const snapshot={scenes,tasks:[],models:[],simulations:[],environment:null,activity:{active:null,queued:[]}};
 await page.route('**/api/**',async route=>{
  if(route.request().method()==='POST')writes++;else reads++;
  const url=new URL(route.request().url());
  if(url.searchParams.has('simulation-replay'))return route.fulfill({json:{frames:[{time:.02,position_m:[0,0,.3],orientation_wxyz:[1,0,0,0],joint_position_rad:Array(12).fill(0),episode_end:true}]}});
  return route.fulfill({json:url.pathname.includes('/go1')?snapshot:[]});
 });
 await page.goto(`${origin}/?module=go1&go1Tab=scenes`);
 const table=page.getByRole('table',{name:'Go1 场景列表',exact:true});
 for(const scene of scenes){
  const row=table.getByRole('row').filter({has:page.getByRole('rowheader',{name:scene.title,exact:true})});
  const view=row.getByRole('button',{name:'查看',exact:true});await view.click();
  const dialog=page.getByRole('dialog',{name:`查看场景 · ${scene.title}`,exact:true});await expect(dialog).toBeVisible();
  const viewport=dialog.locator('[data-testid="go1-scene-viewport"]');await expect(viewport).toHaveAttribute('data-ready','true');
  await expect(viewport).toHaveAttribute('data-box-count',String(scene.boxes));
  await expect(viewport).toHaveAttribute('data-robot-position','0,0,0.3');
  const canvas=dialog.getByRole('img',{name:'Go1 场景三维结构'});await canvas.scrollIntoViewIfNeeded();
  const bounds=await canvas.boundingBox();await page.mouse.move(bounds.x+bounds.width/2,bounds.y+bounds.height/2);
  await page.mouse.down();await page.mouse.move(bounds.x+bounds.width/2+40,bounds.y+bounds.height/2+20,{steps:5});await page.mouse.up();
  await dialog.getByRole('button',{name:'重置视角',exact:true}).click();
  const before=reads;await page.waitForTimeout(1700);assert.equal(reads,before,'scene view must not poll');
  await dialog.getByRole('button',{name:'关闭查看',exact:true}).click();await expect(page.locator('.go1-scene-canvas')).toHaveCount(0);await expect(view).toBeFocused();
  await row.getByRole('button',{name:'详情',exact:true}).click();
  await expect(page.getByRole('region',{name:'Go1 场景详情'})).toBeVisible();
  await expect(page.locator('.go1-scene-canvas')).toHaveCount(0);
  await expect(page.getByText('正在加载三维场景…')).toHaveCount(0);
  await page.getByRole('button',{name:'关闭场景详情',exact:true}).click();
 }
 for(const width of [1920,390]){
  await page.setViewportSize({width,height:900});await table.getByRole('button',{name:'查看',exact:true}).first().click();
  const dialog=page.getByRole('dialog');await expect(dialog.locator('[data-testid="go1-scene-viewport"]')).toHaveAttribute('data-ready','true');
  const box=await dialog.boundingBox();assert.ok(box.x>=0&&box.x+box.width<=width+1,'dialog fits viewport');
  await page.keyboard.press('Escape');await expect(page.getByRole('dialog')).toHaveCount(0);await expect(page.locator('.go1-scene-canvas')).toHaveCount(0);
 }
 assert.equal(writes,0);assert.deepEqual(errors,[]);
 await page.goto(`${origin}/go1/scenes?go1Scene=${scenes[3].id}`);
 await expect(page.getByRole('dialog',{name:'Go1 场景详情'})).toContainText(scenes[3].title);
 await page.reload();
 await expect(page.getByRole('dialog',{name:'Go1 场景详情'})).toBeVisible();
 await page.keyboard.press('Escape');
 await expect(table).toBeVisible();
 assert.equal(new URL(page.url()).searchParams.has('go1Scene'),false);
 const fixture={id:scenes[0].id,title:'合成测试用例',model:scenes[1].id,scene:scenes[0],state:'succeeded',startedAt:'2026-11-19T00:00:00Z',progress:1,steps:50,seed:7,modelFiles:{},modelManifest:{},logs:''};
 snapshot.simulations.push(fixture);
 await page.goto(`${origin}/go1/test-cases`);
 const recordingUrl=page.url();
 await page.getByRole('button',{name:'仿真详情',exact:true}).click();
 await expect(page.getByRole('dialog',{name:'仿真任务详情'})).toContainText(fixture.title);
 await expect(page).toHaveURL(recordingUrl);
 await page.getByRole('button',{name:'查看仿真回放',exact:true}).click();
 await expect(page.getByRole('dialog',{name:'仿真回放 · 合成测试用例'})).toBeVisible();
 await expect(page).toHaveURL(recordingUrl);
 await page.getByRole('button',{name:'关闭回放',exact:true}).click();
 await page.getByRole('button',{name:'关闭仿真详情',exact:true}).click();
 await expect(page.getByRole('table',{name:'Go1 测试用例列表'})).toBeVisible();
 assert.deepEqual(errors,[]);
 console.log('PASS six independent scene views, geometry/orbit/reset, close/reopen/Escape/focus, no viewport in details, no polling/writes, desktop/mobile');
} finally {await browser?.close();server.kill('SIGTERM');await closed;}
