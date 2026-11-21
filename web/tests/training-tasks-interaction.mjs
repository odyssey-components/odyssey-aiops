import { root, product } from "../../scripts/workspace.mjs";
// Focused task list/detail actions with synthetic API data; no screenshot capture.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {mkdir,mkdtemp} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {join} from 'node:path';
const web=join(product,'web');
const {chromium,expect}=createRequire(join(root,'repositories/odyssey-design/packages/odyssey-web-testkit/package.json'))('@playwright/test');
await mkdir(join(root,'tmp/aiops-maze-training-tasks'),{recursive:true});const home=await mkdtemp(join(root,'tmp/aiops-maze-training-tasks/browser-'));
const server=spawn(join(web,'node_modules/.bin/vite'),['preview','--port','0'],{cwd:web,stdio:['ignore','pipe','inherit']});
const closed=once(server,'close');let ready;const originPromise=new Promise(resolve=>{ready=resolve;});server.stdout.on('data',chunk=>{const match=String(chunk).match(/http:\/\/127\.0\.0\.1:\d+/);if(match)ready(match[0]);});
let browser;
try {
 const origin=await originPromise;browser=await chromium.launch();const page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[];page.on('pageerror',error=>errors.push(error.message));
 const course='random-map-v1',first='12345678-1234-1234-1234-123456789abc',second='12345678-1234-1234-1234-123456789abd',created='12345678-1234-1234-1234-123456789abe';
 const dataset={id:'12345678-1234-1234-1234-123456789abf',course,title:'课堂地图',description:'固定地图组',maps:16,validation:64,test:100,seed:7,createdAt:'2026-11-19T02:00:00Z',sha256:'a'.repeat(64)};
 const job=(id,state,startedAt)=>({id,course,action:'train',state,startedAt,finishedAt:state==='succeeded'?'2026-11-19T03:04:00Z':undefined,logs:'课堂训练日志 '+id,metrics:[{epoch:0,total:3,successRate:0,evaluatedMaps:64,split:'validation'}],epochs:3,seed:7,dataset:dataset.id,datasetTitle:dataset.title});
 let tasks=[job(second,'succeeded','2026-11-19T03:00:00Z'),job(first,'failed','2026-11-19T02:00:00Z')],active=null,fail=true,command;
 await page.route('**/api/models**',route=>route.fulfill({json:[]}));
 await page.route('**/api/training**',async route=>{
  if(route.request().method()==='POST') {
   command=route.request().postDataJSON();
   if(command.action==='train') {
    if(fail)return route.fulfill({status:400,json:{error:'合成启动失败'}});
    active=job(created,'running','2026-11-19T04:00:00Z');tasks=[active,...tasks];return route.fulfill({status:202,json:active});
   }
   active={...active,state:'cancelled',finishedAt:'2026-11-19T04:01:00Z'};tasks=tasks.map(item=>item.id===active.id?active:item);const result=active;active=null;return route.fulfill({status:202,json:result});
  }
  const url=new URL(route.request().url()),same=url.searchParams.get('course')===course,id=url.searchParams.get('task');
  await route.fulfill({json:{datasets:same?[dataset]:[],tasks:same?tasks.map(({logs,metrics,...summary})=>({...summary,logs:''})):[],job:same?tasks[0]:null,active,
   ...(id?{detail:same?tasks.find(item=>item.id===id)??null:null}:{})}});
 });
 await page.goto(`${origin}/?course=${course}&tab=training`);await page.waitForLoadState('networkidle');
 await expect(page.getByRole('tab',{name:'训练任务',exact:true})).toHaveAttribute('aria-selected','true');
 await expect(page.getByRole('table',{name:'本课训练任务列表'})).toBeVisible();await expect(page.locator('[data-task-id]')).toHaveCount(2);
 await expect(page.locator('#training-epochs')).toHaveCount(0);
 await page.getByRole('button',{name:`查看训练任务 ${first}`,exact:true}).click();await expect(page).toHaveURL(new RegExp('task='+first));
 await expect(page.getByRole('region',{name:'训练任务详情'})).toBeVisible();await expect(page.getByText('课堂训练日志 '+first,{exact:true})).toBeVisible();
 await page.reload();await expect(page.getByText('课堂训练日志 '+first,{exact:true})).toBeVisible();
 await page.locator('#back-training-tasks').click();await expect(page.getByRole('table',{name:'本课训练任务列表'})).toBeVisible();assert.equal(new URL(page.url()).searchParams.has('task'),false);
 await page.goBack();await expect(page.getByText('课堂训练日志 '+first,{exact:true})).toBeVisible();await page.goForward();await expect(page.locator('[data-task-id]')).toHaveCount(2);
 await page.locator('#add-training-task').click();await page.locator('#training-epochs').fill('3');await page.locator('#start-training').click();
 await expect(page.getByRole('alert')).toContainText('合成启动失败');await expect(page.locator('#training-epochs')).toHaveValue('3');
 fail=false;await page.locator('#start-training').click();await expect(page).toHaveURL(new RegExp('task='+created));
 assert.deepEqual(command,{action:'train',course,dataset:dataset.id,seed:7,epochs:3});
 await expect(page.locator('#training-status')).toHaveText('模型训练：运行中');await page.locator('#cancel-training').click();await expect(page.locator('#training-status')).toHaveText('模型训练：已取消');
 await page.locator('#back-training-tasks').click();await expect(page.locator('[data-task-id]')).toHaveCount(3);
 await page.goto(`${origin}/?course=fixed-map-v1&tab=training&task=${first}`);await expect(page.getByText('本课找不到此训练任务。',{exact:true})).toBeVisible();await page.locator('#back-training-tasks').click();
 await expect(page.locator('[data-task-id]')).toHaveCount(0);
 await page.goto(`${origin}/?course=${course}&tab=training`);await page.setViewportSize({width:390,height:844});await expect(page.locator('[data-task-id]')).toHaveCount(3);
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);assert.deepEqual(errors,[]);
 console.log('PASS task list/detail, exact task logs, URL/reload/back/forward, create failure/retry, automatic detail, cancel, course isolation and narrow layout');
} finally {await browser?.close();server.kill();await closed;}
