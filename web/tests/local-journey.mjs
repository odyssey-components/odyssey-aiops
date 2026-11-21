import { root, product } from "../../scripts/workspace.mjs";
// Real export, cross-language memory traces, and product-owned headless UI.
import assert from 'node:assert/strict';
import {startGoAPI} from './go-api.mjs';
import {spawn,execFileSync} from 'node:child_process';
import {once} from 'node:events';
import {mkdir,mkdtemp,readFile,writeFile,rm} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {join,resolve} from 'node:path';
import {createInterface} from 'node:readline';
import {fileURLToPath} from 'node:url';

const web=fileURLToPath(new URL('../',import.meta.url));
const {chromium,expect}=createRequire(join(root,'repositories/odyssey-design/packages/odyssey-web-testkit/package.json'))('@playwright/test');
await mkdir(join(root,'tmp/odymaze-local-journey'),{recursive:true});
const run=await mkdtemp(join(root,'tmp/odymaze-local-journey/run-'));
let server,browser,closed,api;
try {
  const trained=process.argv[2]?resolve(process.argv[2]):join(run,'trained');
  if(!process.argv[2])execFileSync('uv',['run','--project','.','--no-sync','odymaze-train','--lesson','local-map-v1','--epochs','5','--maps','16','--output',trained],{cwd:root,timeout:60000,stdio:'pipe'});
  const model=JSON.parse(await readFile(join(trained,'model.json'),'utf8'));
  const report=JSON.parse(await readFile(join(trained,'report.json'),'utf8'));
  const reference=JSON.parse(execFileSync('uv',['run','--project','.','--no-sync','python','-c',`
import json, sys, torch
from dataclasses import asdict
from aiops_maze_training.lessons.fixed_map_v1.environment import initial_state, step
from aiops_maze_training.lessons.local_map_v1.environment import generate_maze, initial_memory, observe, encode, expert_action
from aiops_maze_training.lessons.local_map_v1.model import make_model
model=make_model()
data=json.load(open(sys.argv[1]))
with torch.no_grad():
    for index, layer in zip((0,2),data['layers']):
        model[index].weight.copy_(torch.tensor(layer['weight']))
        model[index].bias.copy_(torch.tensor(layer['bias']))
    result=[]
    for seed in [0,1,2,3,4,5,6,7,42,99,100,123,999,2026,65535,999999,2000009,2000010,2147483647,4294967295]:
        maze=generate_maze(seed)
        state=initial_state(maze)
        memory=initial_memory(maze,state)
        trace=[]
        while state.status not in ('success','timeout'):
            observation=encode(memory,state)
            action=expert_action(memory,state)
            trace.append({'state':asdict(state),'memory':memory,'observation':observation,'action':action,'probabilities':torch.softmax(model(torch.tensor(observation)),dim=-1).tolist()})
            state=step(maze,state,action)
            memory=observe(maze,state,memory)
        result.append({'seed':seed,'maze':maze,'trace':trace,'final':asdict(state)})
print(json.dumps(result))
`,join(trained,'model.json')],{cwd:root,encoding:'utf8',timeout:30000,maxBuffer:8*1024*1024}));
  const runs=join(run,'.odyssey/odymaze/runs/local-trained');
  await mkdir(runs,{recursive:true});await writeFile(join(runs,'model.json'),JSON.stringify(model));
  api=await startGoAPI(root,run);
  server=spawn(join(web,'node_modules/.bin/vite'),['preview','--port','0'],{cwd:web,env:{...process.env,HOME:run,ODYMAZE_API_ORIGIN:api.origin},stdio:['ignore','pipe','inherit'],timeout:120000});
  closed=once(server,'close');let origin;
  for await(const line of createInterface({input:server.stdout})){
    const address=line.match(/http:\/\/127\.0\.0\.1:\d+/);if(address){origin=address[0];break;}
  }
  assert.ok(origin);
  assert.deepEqual(await(await fetch(origin+'/api/models?catalog=1')).json(),[{run:'local-trained',lesson:'local-map-v1'}]);
  browser=await chromium.launch({headless:true});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto(origin+'/?course=local-map-v1&seed=7&mode=expert&speed=40');await page.waitForLoadState('networkidle');
  const bundles=await Promise.all(['core.ts','src/lessons/local_map_v1.ts'].map(entry=>Bun.build({entrypoints:[join(web,entry)],target:'browser'})));
  const urls=await Promise.all(bundles.map(async bundle=>{assert.ok(bundle.success,String(bundle.logs));return 'data:text/javascript;base64,'+Buffer.from(await bundle.outputs[0].text()).toString('base64');}));
  const result=await page.evaluate(async({urls,model,reference})=>{
    const core=await import(urls[0]),local=await import(urls[1]);let count=0,maxError=0;
    for(const expected of reference){
      const maze=local.generateMaze(expected.seed);let state=core.initialState(maze),memory=local.initialMemory(maze,state);
      if(JSON.stringify(maze)!==JSON.stringify(expected.maze))throw Error('map mismatch');
      core.validateModel(model,maze);
      for(const row of expected.trace){
        if(JSON.stringify(memory)!==JSON.stringify(row.memory))throw Error('memory mismatch');
        if(JSON.stringify(local.encode(memory,state))!==JSON.stringify(row.observation))throw Error('encoding mismatch');
        const action=local.expertAction(memory,state);if(action!==row.action)throw Error('expert mismatch');
        core.probabilities(model,maze,state,memory).forEach((p,i)=>{maxError=Math.max(maxError,Math.abs(p-row.probabilities[i]));});
        state=core.step(maze,state,action);memory=local.observe(maze,state,memory);count++;
      }
      if(JSON.stringify(state)!==JSON.stringify(expected.final))throw Error('final state mismatch');
    }
    return {count,maxError};
  },{urls,model,reference});
  assert.ok(result.maxError<0.00001,JSON.stringify(result));
  await expect(page.locator('#current-model')).toHaveText('local-trained/model.json');
  await expect(page.locator('#encoding .bit')).toHaveCount(249);
  await expect(page.locator('#board .unknown')).toHaveCount(40);
  await expect(page.locator('#board .goal-label')).toHaveCount(0);
  await expect(page.locator('#board')).toHaveAttribute('aria-label',/出口尚未发现/);
  await expect(page.locator('#local-view .sensor-cell')).toHaveCount(9);
  await expect(page.locator('#mode-expert')).toHaveAttribute('aria-pressed','true');
  await page.locator('#auto').click();
  await expect(page.locator('#game-status')).toHaveText('已到达出口',{timeout:15000});
  await expect(page.locator('#step-count')).toHaveText(`${reference.find(item=>item.seed===7).final.steps} / 200`);
  await page.locator('#reset').click();
  await expect(page.locator('#board .unknown')).toHaveCount(40);
  await expect(page.locator('#visited-count')).toHaveText('已访问 1 格');
  const episode=report.final.episodes.find(item=>item.success)??report.final.episodes[0];
  await page.locator('#maze-seed').fill(String(episode.seed));await page.locator('#new-maze').click();
  await page.locator('#mode-model').click();await page.locator('#auto').click();
  await expect(page.locator('#game-status')).toHaveText(episode.success?'已到达出口':'达到步数上限',{timeout:20000});
  await expect(page.locator('#step-count')).toHaveText(`${episode.steps} / 200`);
  await page.reload();await page.waitForLoadState('networkidle');
  await expect(page.locator('#board .unknown')).toHaveCount(40);
  await expect(page.locator('#running-state')).toHaveText('已暂停');
  await page.setViewportSize({width:390,height:844});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  assert.deepEqual(errors,[]);
  console.log(`PASS local course: ${result.count} cross-language transitions / 20 maps; max probability error ${result.maxError}; expert/model actual rollouts, fog/reset/reload/catalog/URL/narrow UI.`);
}finally{
  await browser?.close();server?.kill();if(closed)await closed;await api?.close();await rm(run,{recursive:true,force:true});
}
