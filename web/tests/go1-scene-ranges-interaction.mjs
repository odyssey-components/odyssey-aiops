import { root, product } from "../../scripts/workspace.mjs";
// Fixed scene choices only: no training, rendering or real scene writes.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {createRequire} from 'node:module';
import {join} from 'node:path';
const web=join(product,'web');
const {chromium,expect}=createRequire(join(root,'repositories/odyssey-design/packages/odyssey-web-testkit/package.json'))('@playwright/test');
const server=spawn(join(web,'node_modules/.bin/vite'),['preview','--port','0'],{cwd:web,stdio:['ignore','pipe','inherit']});const closed=once(server,'close');
let ready;const started=new Promise(resolve=>{ready=resolve;});server.stdout.on('data',chunk=>{const match=String(chunk).match(/http:\/\/127\.0\.0\.1:\d+/);if(match)ready(match[0]);});
let browser;
try {
 const origin=await started;browser=await chromium.launch();const page=await browser.newPage();const errors=[];page.on('pageerror',error=>errors.push(error.message));
 const scenes=[
  ['flat','空旷场地',{forward_m_s:.3,yaw_rate_rad_s:0}],
  ['turn','拐弯',{angle_deg:90,road_width_m:.8,distance_m:1.2}],
  ['uturn','窄路掉头',{road_width_m:.8,turn_at_m:1.2}],
  ['slope','越过小坡',{angle_deg:8,length_m:1.2}],
  ['duck','俯身穿过横梁',{distance_m:1.2,height_m:.28}],
  ['jump','跳过低横梁',{distance_m:1.2,height_m:.2}],
 ].map(([kind,title,parameters],i)=>({id:`a10f5001-0001-4000-8000-${String(i+1).padStart(12,'0')}`,format:'go1-scene/v1',createdAt:'2026-11-19T00:00:00Z',kind,title,parameters}));
 const snapshot={scenes,tasks:[],models:[],environment:null,activity:{active:null,queued:[]}};let writes=0;
 await page.route('**/api/**',async route=>{
  if(route.request().method()==='POST')writes++;
  return route.fulfill({json:route.request().url().includes('/go1')?snapshot:[]});
 });
 await page.goto(origin+'/?module=go1&go1Tab=scenes');
 const table=page.getByRole('table',{name:'Go1 场景列表',exact:true});
 await expect(table.getByRole('row')).toHaveCount(7);
 for(const scene of scenes)await expect(table.getByRole('rowheader',{name:scene.title,exact:true})).toBeVisible();
 await expect(page.getByRole('button',{name:'添加场景',exact:true})).toHaveCount(0);
 await expect(page.getByRole('spinbutton')).toHaveCount(0);
 await expect(table.getByRole('button',{name:'创建训练任务',exact:true})).toHaveCount(5);
 await page.reload();await expect(table.getByRole('row')).toHaveCount(7);
 assert.equal(writes,0);assert.deepEqual(errors,[]);
 console.log('PASS six fixed scene choices, five training entries, no parameter inputs or scene creation writes');
} finally {await browser?.close();server.kill();await closed;}
