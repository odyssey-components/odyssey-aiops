import {apiPath} from '../../apis/path';
import type {Go1Device,Go1Command,Go1Snapshot,Go1Environment,Go1SimulationCommand,Go1RegressionStep,Go1RegressionDepth} from './contract';
import type {Go1Frame} from './contract';
async function request(path:string,command?:unknown,signal?:AbortSignal) {
 const response=await fetch(apiPath(path),command?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(command),signal}:{signal});
 const result=await response.json();if(!response.ok)throw new Error(result.error || '无法读取Go1状态');return result;
}
export async function go1Snapshot(signal?:AbortSignal,query?:URLSearchParams):Promise<Go1Snapshot> {
 const snapshot:Go1Snapshot=await request(query?'/api/go1?'+query:'/api/go1',undefined,signal);
 if (!snapshot.baseModels?.length) return snapshot;
 // Published runs have their own title even when the policy manifest inherits the imported name.
 const titles=new Map(snapshot.models.map(model=>[model.run,model.title]));
 return {...snapshot,baseModels:snapshot.baseModels.map(base=>{
  const title=titles.get(base.id);
  return title ? {...base,title} : base;
 })};
}
export const go1Command=(command:Go1Command|{action:'check';device?:Go1Device}):Promise<Go1Environment|unknown>=>request('/api/go1',command);
export const go1Replay=(id:string,signal?:AbortSignal):Promise<{frames:Go1Frame[]}>=>request('/api/go1?replay='+encodeURIComponent(id),undefined,signal);


export const createGo1Simulation=(command:Go1SimulationCommand):Promise<{id:string}>=>request("/api/go1",command);
export const cancelGo1Simulation=(id:string)=>request("/api/go1",{action:"cancel-simulation",id});
export const go1SimulationReplay=(id:string,signal?:AbortSignal,caseId?:string):Promise<{frames:Go1Frame[];result?:Record<string,unknown>}>=>request("/api/go1?simulation-replay="+encodeURIComponent(id)+(caseId?"&case="+encodeURIComponent(caseId):""),undefined,signal);

export const go1RegressionReport=(id:string,signal?:AbortSignal):Promise<{steps:Go1RegressionStep[]}>=>request("/api/go1?regression-report="+encodeURIComponent(id),undefined,signal);

export const createGo1TestCaseSet=(testCaseSet:{title:string;cases:string[]})=>request("/api/go1",{action:"create-test-case-set",testCaseSet});

export const go1RegressionDepth=(id:string,step:number,signal:AbortSignal):Promise<Go1RegressionDepth>=>request("/api/go1?"+new URLSearchParams({"regression-report":id,step:String(step)}),undefined,signal);

export interface Go1RecordingFrame {
 step:number;
 time:number;
 camera?:{width:number;height:number};
 frame:{depth_f32le_base64?:string|null;body:number[];network_state:number[];action:number[];applied:number[]};
}
export function go1RecordingFrame(id:string,step:number,signal:AbortSignal,caseId?:string):Promise<Go1RecordingFrame> {
 const query=new URLSearchParams({'recording-frame':id,step:String(step)});
 if(caseId)query.set('case',caseId);
 return request('/api/go1?'+query,undefined,signal);
}

export interface Go1RecordingMetrics {
 frames:number;expectedFrames:number;depthFrames:number;observedTopics:number;requiredTopics:number;maxGapMs:number|null;
}
export function go1RecordingMetrics(id:string,signal:AbortSignal,caseId?:string):Promise<Go1RecordingMetrics> {
 const query=new URLSearchParams({'recording-frame':id,step:'metrics'});
 if(caseId)query.set('case',caseId);
 return request('/api/go1?'+query,undefined,signal);
}
