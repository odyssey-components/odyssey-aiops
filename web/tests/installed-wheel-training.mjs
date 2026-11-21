import { root } from "../../scripts/workspace.mjs";
import assert from 'node:assert/strict';
import {mkdir,mkdtemp} from 'node:fs/promises';
import {startGoAPI} from './go-api.mjs';

await mkdir(root+'/tmp/odymaze-go-real',{recursive:true});
const home=await mkdtemp(root+'/tmp/odymaze-go-real/run-');
const api=await startGoAPI(root,home,process.env.HOME+'/.odyssey/odymaze/python/bin/python');
try {
 assert.equal((await fetch(api.origin+'/')).status,404);
 const command=async(body)=>{const r=await fetch(api.origin+'/api/training',{method:'POST',headers:{origin:api.origin,'content-type':'application/json'},body:JSON.stringify(body)});const value=await r.json();assert.ok(r.ok,JSON.stringify(value));return value;};
 const snapshot=async(course)=>await(await fetch(api.origin+'/api/training?course='+course)).json();
 const wait=async(course)=>{for(let i=0;i<240;i++){const s=await snapshot(course);if(s.job&&!['queued','running'].includes(s.job.state)){assert.equal(s.job.state,'succeeded',JSON.stringify(s.job));return s;}await new Promise(r=>setTimeout(r,500));}throw Error('timeout');};
 for(const course of ['fixed-map-v1','random-map-v1','local-map-v1','memory-map-v1']){
  await command({action:'prepare',course,seed:7,maps:16,title:'Smoke'});
  const prepared=await wait(course);const dataset=prepared.datasets[0].id;
  await command({action:'train',course,seed:7,dataset,epochs:2});
  const done=await wait(course);assert.deepEqual(done.job.metrics.map(m=>m.epoch),[0,1,2]);
  console.log('PASS installed wheel training',course,done.job.run);
 }
 const catalog=await(await fetch(api.origin+'/api/models?catalog=1')).json();assert.equal(catalog.length,4);
 console.log('PASS Go API-only, installed wheels, four datasets/training/epoch metrics/model catalog; data:',home);
} finally {await api.close();}
