import { root } from "../../scripts/workspace.mjs";
// Real Go API and numeric NPZ fixtures in an isolated HOME; no GPU training.
import assert from 'node:assert/strict';
import {spawn,execFileSync} from 'node:child_process';
import {once} from 'node:events';
import {mkdir,mkdtemp,writeFile,appendFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';

const binary=resolve(process.argv[2] ?? join(root,'tmp/local-training-validation/aiops-server'));
await mkdir(join(root,'tmp/local-training-validation'),{recursive:true});const run=await mkdtemp(join(root,'tmp/local-training-validation/api-'));
const id='12345678-1234-1234-1234-123456789abc',base=join(run,'.odyssey/go1'),model=join(base,'runs',id),job=join(base,'jobs',id);
await mkdir(model,{recursive:true});await mkdir(job,{recursive:true});
const config={iterations:1,rollout_steps:8,seed:42,target_speed_m_s:0,learning_rate:.0003,eval_seed:10001,eval_steps:2,checkpoint_interval:1};
execFileSync(join(root,'.venv/bin/python'),['-c',`
import json,numpy as np,sys
from pathlib import Path
from aiops_go1_training.locomotion.model_manifest import create_manifest
from aiops_go1_training.locomotion.replay import write_replay
p=Path(sys.argv[1]);config=json.loads(sys.argv[2])
shapes={'l1/weight':(45,64),'l1/bias':(64,),'l2/weight':(64,64),'l2/bias':(64,),'actor/weight':(64,12),'actor/bias':(12,),'critic/weight':(64,1),'critic/bias':(1,),'log_std':(12,)}
np.savez_compressed(p/'policy.npz',**{k:np.zeros(s,dtype=np.float32) for k,s in shapes.items()})
e=dict(source_commit='f4fb0d12ac68842c1567bf1de6f771ce8e6671b5',model_sha256='b'*64,config=config,jax='synthetic',mujoco='synthetic',device='synthetic')
(p/'model-manifest.json').write_text(json.dumps(create_manifest(p/'policy.npz',e,{'synthetic':True})))
write_replay(p/'evaluation-replay.json',[[0,0,.3,1,0,0,0]+[0]*12],[True],.02)
`,model,JSON.stringify(config)],{cwd:root});
await writeFile(join(job,'job.json'),JSON.stringify({format:'go1-local-job/v1',id,title:'合成NPZ模型',state:'succeeded',startedAt:'2026-11-19T00:00:00Z',finishedAt:'2026-11-19T00:01:00Z',config,logs:'synthetic fixture',metrics:[],run:id}));
const server=spawn(binary,['--listen','127.0.0.1:0','--data-dir',join(run,'maze'),'--python',join(root,'.venv/bin/python'),'--go1-python',join(root,'.venv/bin/python'),'--go1-source',join(run,'missing-source')],{env:{...process.env,HOME:run},stdio:['ignore','pipe','pipe']});
const closed=once(server,'close');let readyResolve,readyReject;const ready=new Promise((resolve,reject)=>{readyResolve=resolve;readyReject=reject;});
server.once('error',readyReject);server.stderr.on('data',data=>{const found=String(data).match(/http:\/\/127\.0\.0\.1:\d+/);if(found)readyResolve(found[0]);});
try {
 const origin=await Promise.race([ready,new Promise((_,reject)=>setTimeout(()=>reject(Error('Go API timeout')),15000).unref())]);
 const get=async path=>{const r=await fetch(origin+path);return {status:r.status,data:await r.json()};};
 const post=(value,otherOrigin=origin)=>fetch(origin+'/api/go1',{method:'POST',headers:{origin:otherOrigin,'content-type':'application/json'},body:JSON.stringify(value)});
 const initial=await get('/api/go1');assert.equal(initial.status,200);assert.equal(initial.data.models[0].manifest.observation_dimension,45);assert.equal(initial.data.models[0].manifest.action_dimension,12);
 assert.equal((await get('/api/go1?replay='+id)).data.frames[0].position_m[2],.3);
 const environment=await (await post({action:'check'})).json();assert.equal(environment.ready,false);assert.ok(environment.errors.length);
 assert.equal((await post({action:'train',title:'must refuse',config})).status,503);
 assert.equal((await post({action:'train',title:'bad',config,command:'arbitrary'})).status,400);
 assert.equal((await post({action:'check'},'http://cross-origin.test')).status,403);
 assert.equal((await fetch(origin+'/api/go1?replay=../escape')).status,400);
 await appendFile(join(model,'policy.npz'),'corrupt');assert.equal((await get('/api/go1')).data.models.length,0);
 console.log('PASS real Go1 API: numeric NPZ metadata/replay, checksum rejection, unavailable environment, train refusal and same-origin/parameter guards');
} finally {server.kill('SIGTERM');await closed;}
