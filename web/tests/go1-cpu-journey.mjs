import { root, product } from "../../scripts/workspace.mjs";
// Actual CPU PPO through the static page and Go API, using synthetic task data.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {mkdir,mkdtemp,readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {join} from 'node:path';
import {createInterface} from 'node:readline';
import {startGoAPI} from './go-api.mjs';
const web=join(product,'web');
const {chromium,expect}=createRequire(join(root,'repositories/odyssey-design/packages/odyssey-web-testkit/package.json'))('@playwright/test');
await mkdir(join(root,'tmp/aiops-cpu'),{recursive:true});
const home=await mkdtemp(join(root,'tmp/aiops-cpu/browser-'));
const source=process.env.AIOPS_GO1_SOURCE ?? join(process.env.HOME,'.odyssey/go1/source');
const api=await startGoAPI(root,home,join(root,'.venv/bin/python'),source);
const server=spawn(join(web,'node_modules/.bin/vite'),['preview','--port','0'],{cwd:web,env:{...process.env,AIOPS_API_ORIGIN:api.origin},stdio:['ignore','pipe','inherit']});
const closed=once(server,'close');let browser;
try {
 let origin;
 for await(const line of createInterface({input:server.stdout})) {const match=line.match(/http:\/\/127\.0\.0\.1:\d+/);if(match){origin=match[0];break;}}
 assert.ok(origin);browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const errors=[];page.on('pageerror',error=>errors.push(error.message));
 await page.goto(origin+'/?module=go1');
 await expect(page.getByRole('combobox',{name:'训练设备',exact:true})).toContainText('CPU');
 await expect(page.getByRole('button',{name:'添加训练任务',exact:true})).toBeEnabled({timeout:60000});
 await page.getByRole('button',{name:'添加训练任务',exact:true}).click();
 await page.getByLabel('任务名称',{exact:true}).fill('CPU 真实 PPO 验收');
 for(const [label,value] of [['PPO 迭代次数','1'],['每次采样步数','8'],['评估步数','2'],['检查点间隔（迭代）','1']])await page.getByLabel(label,{exact:true}).fill(value);
 await page.getByRole('button',{name:'提交训练',exact:true}).click();
 await expect(page.getByRole('heading',{name:'Go1 训练任务详情',exact:true})).toBeVisible();
 await expect(page.getByRole('region',{name:'Go1 训练任务',exact:true}).getByText('已完成',{exact:true})).toBeVisible({timeout:180000});
 const snapshot=await(await fetch(origin+'/api/go1')).json();const job=snapshot.tasks[0];
 assert.equal(job.state,'succeeded');assert.equal(job.device,'cpu');assert.equal(job.metrics.length,1);assert.equal(job.metrics[0].timesteps,8);assert.equal(snapshot.models.length,1);
 const manifest=JSON.parse(await readFile(join(home,'.odyssey/go1/runs',job.run,'model-manifest.json'),'utf8'));
 assert.match(manifest.runtime.device,/CPU/i);
 const frames=await(await fetch(origin+'/api/go1?replay='+job.run)).json();assert.equal(frames.frames.length,2);
 await page.reload();await expect(page.getByRole('region',{name:'Go1 训练任务',exact:true}).getByText('已完成',{exact:true})).toBeVisible();
 const post=async body=>{const response=await fetch(origin+'/api/go1',{method:'POST',headers:{origin,'content-type':'application/json'},body:JSON.stringify(body)});const value=await response.json();assert.ok(response.ok,JSON.stringify(value));return value;};
 const scene=await post({action:'create-scene',scene:{title:'CPU视觉转弯',kind:'turn',parameters:{angle_deg:90,road_width_m:.8,distance_m:1.2}}});
 const visual=await post({action:'train',device:'cpu',title:'CPU视觉真实训练',scene:scene.id,config:{iterations:1,rollout_steps:8,eval_steps:2,checkpoint_interval:1}});
 let completed;
 for(let attempt=0;attempt<240;attempt++){
  const current=await(await fetch(origin+'/api/go1')).json();completed=current.tasks.find(task=>task.id===visual.id);
  if(completed&&!['queued','running','finalizing'].includes(completed.state))break;
  await new Promise(resolve=>setTimeout(resolve,500));
 }
 assert.equal(completed?.state,'succeeded',JSON.stringify(completed));assert.equal(completed.device,'cpu');assert.equal(completed.metrics.length,1);
 const final=await(await fetch(origin+'/api/go1')).json();const visualModel=final.models.find(model=>model.task===visual.id);
 assert.equal(visualModel.manifest.physics,'mujoco-cpu');
 assert.deepEqual(errors,[]);
 console.log('PASS CPU visual scene → Go task → native MuJoCo/JAX PPO → verified model publication; data:',home);
 console.log('PASS actual CPU page → environment → Go queued task → PPO update → metrics → validated model → evaluation replay → reload; data:',home);
} finally {await browser?.close();server.kill();await closed;await api.close();}
