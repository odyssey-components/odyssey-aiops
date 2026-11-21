import {useEffect,useState} from 'react';
import {Alert} from '@odyssey/ui/primitives/alert';
import {Button} from '@odyssey/ui/primitives/button';
import {Drawer,DrawerBody,DrawerContent} from '@odyssey/ui/primitives/drawer';
import {Choice} from '../../../components/common/Choice';
import {go1RecordingFrame,go1RecordingMetrics,type Go1RecordingFrame,type Go1RecordingMetrics} from '../api';
import type {Go1Simulation} from '../contract';
import {RecordedDepth} from './RecordedDepth';

export function RecordingFrames({task,onClose}:{task:Go1Simulation;onClose:()=>void}) {
 const [caseId,setCase]=useState(task.cases?.[0]?.id);
 const [step,setStep]=useState(1);
 const [response,setResponse]=useState<{key:string;data?:Go1RecordingFrame;error?:string}>();
 const [metrics,setMetrics]=useState<{key:string;data?:Go1RecordingMetrics}>();
 const metricsKey=JSON.stringify([task.id,caseId]);
 useEffect(()=>{
  const abort=new AbortController();
  go1RecordingMetrics(task.id,abort.signal,caseId).then(data=>{
   if(!abort.signal.aborted)setMetrics({key:metricsKey,data});
  }).catch(()=>{if(!abort.signal.aborted)setMetrics({key:metricsKey});});
  return ()=>abort.abort();
 },[task.id,caseId,metricsKey]);
 const quality=metrics?.key===metricsKey?metrics.data:undefined;
 const results=task.result?.cases as {scene:string;steps:number}[]|undefined;
 const count=caseId ? (results?.find(item=>item.scene===caseId)?.steps ?? 0) : Number(task.result?.steps ?? 0);
 const key=JSON.stringify([task.id,caseId,step]);
 useEffect(()=>{
  const abort=new AbortController();
  go1RecordingFrame(task.id,step,abort.signal,caseId).then(data=>{
   if(!abort.signal.aborted)setResponse({key,data});
  }).catch(error=>{
   if(!abort.signal.aborted)setResponse({key,error:String(error)});
  });
  return ()=>abort.abort();
 },[task.id,caseId,step,key]);
 const data=response?.key===key?response.data:undefined;
 const error=response?.key===key?response.error:undefined;
 return <Drawer open onOpenChange={open=>{if(!open)onClose();}}>
  <DrawerContent title="录制帧" closeLabel="关闭录制帧" className="go1-simulation-drawer">
   <DrawerBody className="dataset-detail">
    <p>{task.title} · policy-input.mcap</p>
    {task.cases&&<Choice id="go1-recording-case" label="录制场景" value={caseId ?? ''} options={task.cases.map(scene=>({value:scene.id,label:scene.title}))} onChange={id=>{setCase(id);setStep(1);}}/>}
    {quality?<dl aria-label="MCAP 输入质量" className="go1-scene-facts">
     <div><dt>必需 Topic 覆盖率</dt><dd>{100*quality.observedTopics/quality.requiredTopics}%（{quality.observedTopics}/{quality.requiredTopics}）</dd></div>
     <div><dt>最大帧间隔</dt><dd>{quality.maxGapMs===null?'暂无数据':quality.maxGapMs.toFixed(1)+' ms'}</dd></div>
     <div><dt>策略帧覆盖</dt><dd>{quality.frames}/{quality.expectedFrames} 帧</dd></div>
     <div><dt>含深度数据</dt><dd>{quality.depthFrames}/{quality.frames} 帧</dd></div>
    </dl>:<p role="status">{metrics?.key===metricsKey?'录制质量指标不可用':'正在计算录制质量…'}</p>}
    <div className="training-actions">
     <Button disabled={step<=1} onClick={()=>setStep(step-1)}>上一帧</Button>
     <label>录制帧 <input type="number" min={1} max={count} value={step} onChange={event=>{
      const value=event.currentTarget.valueAsNumber;if(Number.isInteger(value)&&value>=1&&value<=count)setStep(value);
     }}/></label>
     <Button disabled={step>=count} onClick={()=>setStep(step+1)}>下一帧</Button>
    </div>
    {error?<Alert tone="danger" role="alert">{error}</Alert>:!data?<p role="status">正在读取录制帧…</p>:<>
     <p>第 {data.step} / {count} 帧 · {data.time.toFixed(2)} 秒</p>
     {data.camera&&data.frame.depth_f32le_base64?<RecordedDepth {...data.camera} f32le={data.frame.depth_f32le_base64}/>:<p>该帧没有深度数据</p>}
     {([['机身观测',data.frame.body],['策略机身输入',data.frame.network_state],['策略动作',data.frame.action],['施加动作',data.frame.applied]] as const).map(([label,values])=>
      <details key={label}><summary>{label}</summary><pre className="training-logs">{JSON.stringify(values)}</pre></details>)}
    </>}
   </DrawerBody>
  </DrawerContent>
 </Drawer>;
}
