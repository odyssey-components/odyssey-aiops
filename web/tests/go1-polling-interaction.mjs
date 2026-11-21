import { root, product } from "../../scripts/workspace.mjs";
// Product-owned headless regression: no screenshots, real user data, or training.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {createRequire} from 'node:module';
import {join} from 'node:path';
const web=join(product,'web');
const {chromium,expect}=createRequire(join(root,'repositories/odyssey-design/packages/odyssey-web-testkit/package.json'))('@playwright/test');
const server=spawn(join(web,'node_modules/.bin/vite'),['preview','--port','0'],{cwd:web,stdio:['ignore','pipe','inherit']});
const closed=once(server,'close');let resolveOrigin;
const ready=new Promise(resolve=>{resolveOrigin=resolve;});
server.stdout.on('data',chunk=>{const found=String(chunk).match(/http:\/\/127\.0\.0\.1:\d+/);if(found)resolveOrigin(found[0]);});
let browser;
try {
 const origin=await Promise.race([ready,new Promise((_,reject)=>setTimeout(()=>reject(Error('preview timeout')),20000).unref())]);
 browser=await chromium.launch({headless:true});const page=await browser.newPage();
 let reads=0,documents=0,live=false;
 page.on('request',request=>{if(request.resourceType()==='document')documents++;});
 const id='12345678-1234-1234-1234-123456789abc';
 const snapshot={tasks:[],models:[],scenes:[{id,format:'go1-scene/v1',title:'保持输入',kind:'turn',createdAt:'2026-11-19T00:00:00Z',parameters:{angle_deg:90,road_width_m:.8,distance_m:1.2}}],environment:{ready:true,device:'cpu',devices:['fixture'],errors:[]},activity:{active:null,queued:[]}};
 await page.route('**/api/go1**',async route=>{
  reads++;
  const tasks=live?[{id,title:'活动任务',format:'go1-local-job/v1',state:'running',startedAt:'2026-11-19T00:00:00Z',config:{},logs:'step '+reads,metrics:[]}]:[];
  return route.fulfill({json:{...snapshot,tasks}});
 });
 await page.goto(`${origin}/?module=go1&go1Tab=scenes`);
 await expect(page.getByRole('rowheader',{name:'保持输入',exact:true})).toBeVisible();
 const detail=page.getByRole('button',{name:'详情',exact:true});await detail.focus();
 const initialReads=reads;
 await page.waitForTimeout(3300);
 assert.equal(reads,initialReads,'idle scene page must stop automatic snapshot requests');
 await expect(detail).toBeFocused();
 assert.equal(documents,1,'polling must never reload the document');
 live=true;
 await page.getByRole('button',{name:'训练任务',exact:true}).click();
 await expect(page.getByRole('rowheader',{name:'活动任务',exact:true})).toBeVisible();
 const activeReads=reads;
 await expect.poll(()=>reads,{timeout:5000}).toBeGreaterThan(activeReads);
 await page.getByRole('button',{name:'模型',exact:true}).click();await expect(page.getByRole('table',{name:'Go1 模型列表',exact:true})).toBeVisible();
 const staticReads=reads;await page.waitForTimeout(3300);
 assert.equal(reads,staticReads,'static model page must stop polling even while a task runs');
 assert.equal(documents,1);
 console.log('PASS idle/static polling stopped, scene action focus preserved, active task polling retained, no document reload');
} finally {await browser?.close();server.kill('SIGTERM');await closed;}
