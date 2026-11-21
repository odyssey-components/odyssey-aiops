import { root } from "../../scripts/workspace.mjs";
// Product-owned headless test: exported GRU sequences, state lifecycle and real UI.
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
await mkdir(join(root,'tmp/odymaze-memory-journey'),{recursive:true});
const run=await mkdtemp(join(root,'tmp/odymaze-memory-journey/run-'));
let server,browser,closed,api;
try {
  const trained=process.argv[2]?resolve(process.argv[2]):join(run,'trained');
  if(!process.argv[2])execFileSync('uv',['run','--project','.','--no-sync','odymaze-train','--lesson','memory-map-v1','--epochs','2','--maps','16','--output',trained],{cwd:root,timeout:60000,stdio:'pipe'});
  const model=JSON.parse(await readFile(join(trained,'model.json'),'utf8'));
  const report=JSON.parse(await readFile(join(trained,'report.json'),'utf8'));
  const reference=JSON.parse(execFileSync('uv',['run','--project','.','--no-sync','python','-c',`
import json, sys, torch
from dataclasses import asdict
from aiops_maze_training.lessons.fixed_map_v1.environment import initial_state, step
from aiops_maze_training.lessons.memory_map_v1.environment import generate_maze, encode
from aiops_maze_training.lessons.memory_map_v1.model import make_model
model=make_model()
data=json.load(open(sys.argv[1]))
with torch.no_grad():
    for key in ('weight_ih','weight_hh','bias_ih','bias_hh'):
        getattr(model.gru,key+'_l0').copy_(torch.tensor(data['gru'][key]))
    model.output.weight.copy_(torch.tensor(data['layers'][0]['weight']))
    model.output.bias.copy_(torch.tensor(data['layers'][0]['bias']))
    result=[]
    for seed in [0,1,2,3,4,5,6,7,42,99,100,123,999,2026,65535,999999,2000009,2000010,2147483647,4294967295]:
        maze=generate_maze(seed)
        state=initial_state(maze)
        hidden=None
        previous=-1
        trace=[]
        while True:
            observation=encode(maze,state,previous)
            logits,hidden=model.step(torch.tensor(observation),hidden)
            trace.append({'state':asdict(state),'previous':previous,'observation':observation,'hidden':hidden.flatten().tolist(),'probabilities':torch.softmax(logits,dim=-1).tolist()})
            if state.status in ('success','timeout'):
                break
            action=int(logits.argmax())
            state=step(maze,state,action)
            previous=action
        result.append({'seed':seed,'trace':trace})
print(json.dumps(result))
`,join(trained,'model.json')],{cwd:root,encoding:'utf8',timeout:30000,maxBuffer:16*1024*1024}));
  for(const name of ['memory-trained','memory-copy']){
    const output=join(run,'.odyssey/odymaze/runs',name);await mkdir(output,{recursive:true});
    await writeFile(join(output,'model.json'),JSON.stringify(model));
  }
  api=await startGoAPI(root,run);
  server=spawn(join(web,'node_modules/.bin/vite'),['preview','--port','0'],{cwd:web,env:{...process.env,HOME:run,ODYMAZE_API_ORIGIN:api.origin},stdio:['ignore','pipe','inherit'],timeout:120000});
  closed=once(server,'close');let origin;
  for await(const line of createInterface({input:server.stdout})){
    const address=line.match(/http:\/\/127\.0\.0\.1:\d+/);if(address){origin=address[0];break;}
  }
  assert.ok(origin);
  assert.deepEqual(await(await fetch(origin+'/api/models?catalog=1')).json(),[
    {run:'memory-trained',lesson:'memory-map-v1'},{run:'memory-copy',lesson:'memory-map-v1'},
  ]);
  browser=await chromium.launch({headless:true});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto(origin+'/?course=memory-map-v1&seed=7&mode=expert&speed=40');await page.waitForLoadState('networkidle');
  const bundles=await Promise.all(['src/utilities/game.ts','src/lessons/memory_map_v1.ts'].map(entry=>Bun.build({entrypoints:[join(web,entry)],target:'browser'})));
  const urls=await Promise.all(bundles.map(async bundle=>{assert.ok(bundle.success,String(bundle.logs));return 'data:text/javascript;base64,'+Buffer.from(await bundle.outputs[0].text()).toString('base64');}));
  const result=await page.evaluate(async({urls,model,reference})=>{
    const gameModule=await import(urls[0]),environment=await import(urls[1]);let count=0,maxHiddenError=0,maxProbabilityError=0;
    for(const expected of reference){
      let game=gameModule.updateGame(gameModule.startGame('memory-map-v1',expected.seed),{type:'loaded',model});
      for(const [index,row] of expected.trace.entries()){
        if(JSON.stringify(game.state)!==JSON.stringify(row.state))throw Error('rollout state mismatch');
        if(game.previousAction!==row.previous)throw Error('previous action mismatch');
        if(JSON.stringify(environment.encode(game.maze,game.state,game.previousAction))!==JSON.stringify(row.observation))throw Error('observation mismatch');
        if(game.recurrent.observations!==index+1)throw Error('observation count mismatch');
        game.recurrent.hidden.forEach((value,i)=>{maxHiddenError=Math.max(maxHiddenError,Math.abs(value-row.hidden[i]));});
        game.recurrent.values.forEach((value,i)=>{maxProbabilityError=Math.max(maxProbabilityError,Math.abs(value-row.probabilities[i]));});
        if(index<expected.trace.length-1)game=gameModule.updateGame(game,{type:'step'});
        count++;
      }
    }
    return {count,maxHiddenError,maxProbabilityError};
  },{urls,model,reference});
  assert.ok(result.maxHiddenError<0.00001,JSON.stringify(result));
  assert.ok(result.maxProbabilityError<0.00001,JSON.stringify(result));
  await expect(page.locator('#current-model')).toHaveText('memory-trained/model.json');
  await expect(page.locator('#encoding .bit')).toHaveCount(40);
  await expect(page.locator('#gru-hidden .hidden-unit')).toHaveCount(32);
  await expect(page.locator('#gru-observations')).toHaveText('已处理 1 次观察');
  await expect(page.locator('#board .unknown')).toHaveCount(40);
  await page.locator('#single-step').click();
  await expect(page.locator('#gru-observations')).toHaveText('已处理 2 次观察');
  const hiddenBefore=await page.locator('#gru-hidden').innerHTML();
  await page.locator('[data-lesson="3"]').click();
  await page.locator('#mode-manual').click();
  await expect(page.locator('#gru-observations')).toHaveText('已处理 2 次观察');
  assert.equal(await page.locator('#gru-hidden').innerHTML(),hiddenBefore);
  await page.locator('[data-action="0"]').click();
  await expect(page.locator('#gru-observations')).toHaveText('已处理 3 次观察');
  const pose=await page.locator('#position').textContent(),steps=await page.locator('#step-count').textContent(),visited=await page.locator('#visited-count').textContent();
  await page.locator('#forget-memory').click();
  await expect(page.locator('#gru-observations')).toHaveText('已处理 1 次观察');
  await expect(page.locator('#position')).toHaveText(pose);await expect(page.locator('#step-count')).toHaveText(steps);
  await expect(page.locator('#visited-count')).toHaveText(visited);
  await page.locator('#model-list').click();
  await expect(page.getByRole('option')).toHaveText(['memory-trained/model.json','memory-copy/model.json']);
  await page.getByRole('option',{name:'memory-copy/model.json',exact:true}).click();
  await expect(page.locator('#step-count')).toHaveText('0 / 200');
  await expect(page.locator('#gru-observations')).toHaveText('已处理 1 次观察');
  await expect(page).toHaveURL(/model=memory-copy/);
  const episode=report.final.episodes.find(item=>item.success)??report.final.episodes[0];
  await page.locator('#maze-seed').fill(String(episode.seed));await page.locator('#new-maze').click();
  await page.locator('#mode-model').click();await page.locator('#auto').click();
  await expect(page.locator('#game-status')).toHaveText(episode.success?'已到达出口':'达到步数上限',{timeout:20000});
  await expect(page.locator('#step-count')).toHaveText(`${episode.steps} / 200`);
  await expect(page.locator('#gru-observations')).toHaveText(`已处理 ${episode.steps+1} 次观察`);
  await page.reload();await page.waitForLoadState('networkidle');
  await expect(page.locator('#board .unknown')).toHaveCount(40);
  await expect(page.locator('#gru-observations')).toHaveText('已处理 1 次观察');
  await expect(page.locator('#running-state')).toHaveText('已暂停');
  await page.setViewportSize({width:390,height:844});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));assert.deepEqual(errors,[]);
  console.log(`PASS GRU: ${result.count} observations / 20 actual model rollouts; max hidden error ${result.maxHiddenError}, probability error ${result.maxProbabilityError}; current-only inputs, once-per-observation, forget/mode/load/seed/reload, catalog/model select and narrow UI.`);
}finally{
  await browser?.close();server?.kill();if(closed)await closed;await api?.close();await rm(run,{recursive:true,force:true});
}
