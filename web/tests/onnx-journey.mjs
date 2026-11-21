import { root } from "../../scripts/workspace.mjs";
// Real offline WASM execution of all courses. No JSON network endpoint or OS automation.
import assert from 'node:assert/strict';
import {startGoAPI} from './go-api.mjs';
import {spawn,execFileSync} from 'node:child_process';
import {once} from 'node:events';
import {mkdir,mkdtemp,readFile,writeFile,copyFile,rm} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {join,resolve} from 'node:path';
import {createInterface} from 'node:readline';
import {fileURLToPath} from 'node:url';

const web=fileURLToPath(new URL('../',import.meta.url));
const {chromium,expect}=createRequire(join(root,'repositories/odyssey-design/packages/odyssey-web-testkit/package.json'))('@playwright/test');
await mkdir(join(root,'tmp/odymaze-onnx-journey'),{recursive:true});
const run=await mkdtemp(join(root,'tmp/odymaze-onnx-journey/run-'));
let server,browser,closed,api;
try{
  const lessons=['fixed-map-v1','random-map-v1','local-map-v1','memory-map-v1'];
  const sources=process.argv[2]?JSON.parse(await readFile(resolve(process.argv[2]),'utf8')):{};
  for(const lesson of lessons)if(!sources[lesson]){
    sources[lesson]=join(run,'trained-'+lesson);
    execFileSync('uv',['run','--project','.','--no-sync','odymaze-train','--lesson',lesson,'--epochs','0',...(lesson==='fixed-map-v1'?[]:['--maps','16']),'--output',sources[lesson]],{cwd:root,stdio:'pipe',timeout:60000});
  }
  const indexPath=join(run,'sources.json');await writeFile(indexPath,JSON.stringify(sources));
  const reference=JSON.parse(execFileSync('uv',['run','--project','.','--no-sync','python','-c',`
import json, sys, torch
from dataclasses import asdict
from aiops_maze_training.lessons.fixed_map_v1.environment import initial_state, step, encode as fixed_encode
from aiops_maze_training.lessons.random_map_v1.environment import generate_maze, encode as random_encode
from aiops_maze_training.lessons.local_map_v1.environment import observe, encode as local_encode
from aiops_maze_training.lessons.memory_map_v1.environment import encode as memory_encode
from aiops_maze_training.common.onnx_export import _restore, _validate
from aiops_maze_training.common.manifest import resolve_policy
sources=json.load(open(sys.argv[1]))
result=[]
with torch.no_grad():
    for lesson, source in sources.items():
        policy=resolve_policy(json.load(open(source+'/model.json')))
        report=json.load(open(source+'/report.json'))
        _,width,hidden_size=_validate(policy)
        recurrent=lesson=='memory-map-v1'
        network=_restore(policy,width,hidden_size,recurrent)
        episode=next((item for item in report['final'].get('episodes',[]) if item['success']),None)
        episodes=report['final'].get('episodes',[])
        seed=(episode or (episodes[0] if episodes else {})).get('seed',7)
        maze=policy['maze'] if lesson=='fixed-map-v1' else generate_maze(seed)
        if lesson in ('local-map-v1','memory-map-v1'):
            maze['max_steps']=200
        state=initial_state(maze)
        memory=observe(maze,state)
        previous=-1
        hidden=torch.zeros(1,32)
        trace=[]
        while True:
            observation=memory_encode(maze,state,previous) if recurrent else local_encode(memory,state) if lesson=='local-map-v1' else random_encode(maze,state) if lesson=='random-map-v1' else fixed_encode(maze,state)
            tensor=torch.tensor([observation],dtype=torch.float32)
            if recurrent:
                logits,hidden=network(tensor,hidden)
            else:
                logits=network(tensor)
            trace.append({'state':asdict(state),'values':torch.softmax(logits[0],dim=-1).tolist(),'hidden':hidden[0].tolist() if recurrent else None})
            if state.status in ('success','timeout'):
                break
            previous=int(logits[0].argmax())
            state=step(maze,state,previous)
            memory=observe(maze,state,memory)
        result.append({'lesson':lesson,'seed':seed,'trace':trace,'max_steps':maze['max_steps']})
print(json.dumps(result))
`,indexPath],{cwd:root,encoding:'utf8',timeout:30000,maxBuffer:8*1024*1024}));
  for(const lesson of lessons){
    const output=join(run,'.odyssey/odymaze/runs',lesson);await mkdir(output,{recursive:true});
    for(const name of ['model.onnx','model.manifest.toml'])await copyFile(join(sources[lesson],name),join(output,name));
  }
  api=await startGoAPI(root,run);
  assert.equal((await fetch(api.origin+'/')).status,404);
  server=spawn(join(web,'node_modules/.bin/vite'),['preview','--port','0'],{cwd:web,env:{...process.env,HOME:run,ODYMAZE_API_ORIGIN:api.origin},stdio:['ignore','pipe','inherit'],timeout:180000});
  closed=once(server,'close');let origin;
  for await(const line of createInterface({input:server.stdout})){
    const address=line.match(/http:\/\/127\.0\.0\.1:\d+/);if(address){origin=address[0];break;}
  }
  assert.ok(origin);
  const catalog=await(await fetch(origin+'/api/models?catalog=1')).json();assert.equal(catalog.length,4);
  browser=await chromium.launch({headless:true});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[],requests=[];
  page.on('pageerror',error=>errors.push(error.message));page.on('request',request=>requests.push(request.url()));
  let maxProbabilityError=0,maxHiddenError=0;
  for(const item of reference){
    await page.goto(`${origin}/?course=${item.lesson}&seed=${item.seed}&mode=model&speed=40`);
    await expect(page.locator('#current-model')).toHaveText(item.lesson+'/model.onnx');
    const modelBox=await page.locator('.model-card').boundingBox(),decisionBox=await page.locator('.decision-card').boundingBox();
    assert.ok(modelBox && decisionBox && modelBox.y+modelBox.height<=decisionBox.y,'model selector is first in right sidebar');
    await expect(page.locator('.inspector > .model-card:first-child')).toHaveCount(1);
    assert.ok(await page.locator('.workspace').evaluate(element=>element.parentElement.firstElementChild===element),'no settings row above game workspace');
    await expect(page.locator('#probabilities')).toHaveAttribute('data-values',/\d/,{timeout:30000});
    await expect(page.locator('.learning-path')).toHaveCount(0);await expect(page.locator('#lesson-body')).toHaveCount(0);
    await expect(page.locator('.intro-copy')).toHaveCount(0);await expect(page.locator('.scope-note')).toHaveCount(0);
    const count=Math.min(5,item.trace.length);
    for(let index=0;index<count;index++){
      const expected=item.trace[index];
      await expect(page.locator('#step-count')).toHaveText(`${expected.state.steps} / ${item.max_steps}`);
      await expect(page.locator('#position')).toHaveText(`(${expected.state.x}, ${expected.state.y})`);
      const terminal=['success','timeout'].includes(expected.state.status);
      if(!terminal){
        await expect(page.locator('#probabilities')).toHaveAttribute('data-values',/\d/);
        const actual=(await page.locator('#probabilities').getAttribute('data-values')).split(',').map(Number);
        actual.forEach((value,i)=>{maxProbabilityError=Math.max(maxProbabilityError,Math.abs(value-expected.values[i]));});
      }
      if(expected.hidden){
        await expect(page.locator('#gru-observations')).toHaveText(`已处理 ${index+1} 次观察`);
        const actual=await page.locator('#gru-hidden .hidden-unit').evaluateAll(nodes=>nodes.map(node=>Number(node.dataset.value)));
        actual.forEach((value,i)=>{maxHiddenError=Math.max(maxHiddenError,Math.abs(value-expected.hidden[i]));});
      }
      if(index<count-1)await page.locator('#single-step').click();
    }
    const final=item.trace.at(-1).state;
    if(!['success','timeout'].includes(item.trace[count-1].state.status))await page.locator('#auto').click();
    await expect(page.locator('#game-status')).toHaveText(final.status==='success'?'已到达出口':'达到步数上限',{timeout:30000});
    await expect(page.locator('#step-count')).toHaveText(`${final.steps} / ${item.max_steps}`);
    console.log('PASS ONNX WASM course',item.lesson,final.steps,final.status);
  }
  assert.ok(maxProbabilityError<1e-5,`probability error ${maxProbabilityError}`);assert.ok(maxHiddenError<1e-5,`hidden error ${maxHiddenError}`);
  await page.locator('#reset').click();await expect(page.locator('#gru-observations')).toHaveText('已处理 1 次观察');
  await page.locator('#mode-manual').click();await page.locator('[data-action="0"]').click();
  await expect(page.locator('#gru-observations')).toHaveText('已处理 2 次观察');
  const pose=await page.locator('#position').textContent(),steps=await page.locator('#step-count').textContent();
  await page.locator('#forget-memory').click();await expect(page.locator('#gru-observations')).toHaveText('已处理 1 次观察');
  await expect(page.locator('#position')).toHaveText(pose);await expect(page.locator('#step-count')).toHaveText(steps);
  await page.reload();await expect(page.locator('#gru-observations')).toHaveText('已处理 1 次观察');
  await expect(page.locator('#running-state')).toHaveText('已暂停');
  // A corrupt response must fail without falling back to JSON weights.
  await page.route('**/api/models/fixed-map-v1?artifact=onnx',route=>route.fulfill({status:200,body:Buffer.from([1,2,3])}));
  await page.goto(`${origin}/?course=fixed-map-v1&model=fixed-map-v1&mode=model`);
  await expect(page.locator('#import-message')).toHaveAttribute('data-kind','error');
  await expect(page.locator('#mode-model')).toBeDisabled();
  await page.unroute('**/api/models/fixed-map-v1?artifact=onnx');
  await page.reload();await expect(page.locator('#probabilities')).toHaveAttribute('data-values',/\d/);
  // Start from another course so navigation must create a new fixed-course load.
  await page.goto(`${origin}/?course=memory-map-v1&model=memory-map-v1&mode=model`);
  await expect(page.locator('#current-model')).toHaveText('memory-map-v1/model.onnx');
  await expect(page.locator('#probabilities')).toHaveAttribute('data-values',/\d/);
  // Finish a superseded load after a course switch; it cannot replace the new session.
  let release;const gate=new Promise(resolve=>{release=resolve;});
  const delayedRoutes=[];
  await page.route('**/api/models/fixed-map-v1?artifact=onnx',route=>{
    const delayed=(async()=>{await gate;try{await route.continue();}catch(error){if(!String(error).includes('already handled'))throw error;}})();
    delayedRoutes.push(delayed);return delayed;
  });
  const requested=page.waitForRequest(request=>request.url().endsWith('/api/models/fixed-map-v1?artifact=onnx'));
  await page.goto(`${origin}/?course=fixed-map-v1&model=fixed-map-v1&mode=model`,{waitUntil:'domcontentloaded'});
  await requested;
  await page.locator('[data-od-slot="pro-admin-navigation"][data-od-navigation-surface="sidebar"]').getByRole('button',{name:'第四课：模型记忆',exact:true}).click();
  await expect(page.locator('#current-model')).toHaveText('memory-map-v1/model.onnx');
  release();
  await Promise.all(delayedRoutes);
  await page.unroute('**/api/models/fixed-map-v1?artifact=onnx');
  await page.waitForLoadState('networkidle');
  await expect(page.locator('#probabilities')).toHaveAttribute('data-values',/\d/);
  await expect(page.locator('#current-model')).toHaveText('memory-map-v1/model.onnx');
  await expect(page.locator('#step-count')).toHaveText('0 / 200');
  await page.setViewportSize({width:390,height:844});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  const jsonWeights=requests.filter(url=>new URL(url).pathname.startsWith('/api/models/')&&!['manifest','onnx'].includes(new URL(url).searchParams.get('artifact')));
  assert.deepEqual(jsonWeights,[]);assert.deepEqual(errors,[]);
  assert.ok(requests.some(url=>new URL(url).pathname.endsWith('.wasm')),'real WASM binary was requested');
  assert.ok(requests.every(url=>url.startsWith(origin)||url.startsWith('blob:'+origin)||url.startsWith('data:')),'no external CDN');
  console.log(`PASS browser ONNX: four actual WASM sessions, max probability error ${maxProbabilityError}, hidden error ${maxHiddenError}; no JSON weight requests/CDN; lifecycle, simplified page, narrow UI.`);
}finally{
  await browser?.close();server?.kill();if(closed)await closed;await api?.close();await rm(run,{recursive:true,force:true});
}
