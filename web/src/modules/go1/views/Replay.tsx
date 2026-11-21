import {lazy,Suspense,useEffect,useMemo,useRef,useState} from 'react';
import {Button} from '@odyssey/ui/primitives/button';
import {Card} from '@odyssey/ui/primitives/card';
import {Alert} from '@odyssey/ui/primitives/alert';
import {Slider,SliderControl,SliderTrack,SliderIndicator,SliderThumb,SliderLabel} from '@odyssey/ui/primitives/slider';
import {Choice} from '../../../components/common/Choice';
import {go1Replay,go1SimulationReplay} from '../api';
import type {Go1Scene} from '../scenes';
import type {Go1Frame,Go1Model} from '../contract';
import {frameAtTime,parseFrames,samplePose} from '../replay/trajectory';
import '../replay/replay.css';
import {ReplayMetrics} from './ReplayMetrics';
import {ReplayDepth} from '../replay/ReplayDepth';

const Viewport=lazy(()=>import('../replay/Viewport'));
export function Go1Replay({models,selected,onSelect,simulation,onBack,embedded=false}:{simulation?:{id:string;title:string;scene?:Go1Scene;caseId?:string;visionInjected?:boolean;originalId?:string};onBack?:()=>void;embedded?:boolean;models:Go1Model[];selected?:string;onSelect:(id:string)=>void}) {
 const id=simulation?.id ?? selected ?? models[0]?.run,source=simulation?'simulation':'evaluation',caseId=simulation?.caseId,key=source+':'+(id ?? '')+':'+(caseId ?? '');
 const [loaded,setLoaded]=useState(''),[frames,setFrames]=useState<Go1Frame[]>([]),[error,setError]=useState(''),[loading,setLoading]=useState(false);
 const [candidateResult,setCandidateResult]=useState<Record<string,unknown>>();
 const [showEyes,setShowEyes]=useState(true);
 const [showDepth,setShowDepth]=useState(true);
 const [readyDepth,setReadyDepth]=useState<string>();
 const [time,setTime]=useState(0),[playing,setPlaying]=useState(false),[speed,setSpeed]=useState(1),[follow,setFollow]=useState(true);
 const originalId=simulation?.originalId;
 const [original,setOriginal]=useState<{id:string;frames?:Go1Frame[];result?:Record<string,unknown>;error?:string}>();
 const [showOriginal,setShowOriginal]=useState(false);
 const originalFrames=original?.id===originalId?original?.frames:undefined;
 const activeFrames=showOriginal&&originalFrames?originalFrames:frames;
 useEffect(()=>{
  if(!originalId)return;
  const abort=new AbortController();
  void go1SimulationReplay(originalId,abort.signal).then(({frames,result})=>{
   if(!abort.signal.aborted)setOriginal({id:originalId,frames:parseFrames(frames),result});
  }).catch(error=>{if(!abort.signal.aborted)setOriginal({id:originalId,error:String(error)});});
  return ()=>abort.abort();
 },[originalId]);
 const cursor=useRef(time);cursor.current=time;
 useEffect(()=>{
  const controller=new AbortController();setLoaded('');setFrames([]);setError('');setTime(0);setPlaying(false);setLoading(Boolean(id));
  if(id)void (source==='simulation'?go1SimulationReplay(id,controller.signal,caseId):go1Replay(id,controller.signal)).then((value:{frames:Go1Frame[];result?:Record<string,unknown>})=>{
   if(controller.signal.aborted)return;const poses=parseFrames(value.frames);setFrames(poses);setCandidateResult(value.result);setTime(poses[0].time);setLoaded(key);
  }).catch(error=>{if(!controller.signal.aborted)setError(error instanceof Error?error.message:'无法读取回放');}).finally(()=>{if(!controller.signal.aborted)setLoading(false);});
  return ()=>controller.abort();
 },[id,source,key,caseId]);
 const ready=loaded===key&&frames.length>0,start=ready?activeFrames[0].time:0,end=ready?activeFrames.at(-1)!.time:0;
 useEffect(()=>{
  if(!playing||!ready||activeFrames.length<2)return;
  let request=0,previous=performance.now(),position=cursor.current;
  const tick=(now:number)=>{position=Math.min(end,position+(now-previous)/1000*speed);previous=now;cursor.current=position;setTime(position);if(position>=end){setPlaying(false);return;}request=requestAnimationFrame(tick);};
  const pauseHidden=()=>{if(document.hidden)setPlaying(false);};document.addEventListener('visibilitychange',pauseHidden);request=requestAnimationFrame(tick);
  return ()=>{cancelAnimationFrame(request);document.removeEventListener('visibilitychange',pauseHidden);};
 },[playing,ready,activeFrames,end,speed]);
 const frame=useMemo(()=>ready?samplePose(activeFrames,time):undefined,[activeFrames,time,ready]);
 const index=ready?frameAtTime(activeFrames,time):0;
 const injected=Boolean(simulation?.visionInjected&&!showOriginal);
 const depthId=showOriginal ? originalId : simulation?.id;
 const depthCase=showOriginal ? undefined : caseId;
 const depthKey=depthId+':'+(depthCase ?? '')+':'+injected;
 const depthAvailable=Boolean(depthId&&readyDepth===depthKey);
 const depthIndex=Math.min(activeFrames.length-1,Math.max(0,Math.floor(time*50+1e-7)));
 function seek(value:number){setPlaying(false);cursor.current=value;setTime(value);}
 function switchTrack(toOriginal:boolean) {
  if(!originalFrames)return;
  const target=toOriginal?originalFrames:frames;
  seek(Math.max(target[0].time,Math.min(time,target.at(-1)!.time)));
  setShowOriginal(toOriginal);
 }
 return <section className="training-workspace go1-replay-workspace" aria-label={simulation?"Go1 仿真回放":"Go1 训练评估记录"}>
  {!embedded&&<div className="dataset-heading go1-replay-heading"><h1>{simulation?`仿真回放 · ${simulation.title}`:'训练评估记录'}</h1>{onBack&&<Button onClick={onBack}>{simulation?'返回仿真任务':'返回模型详情'}</Button>}
  {!simulation&&<Choice id="go1-replay-model" label="评估模型" value={id ?? ''} disabled={!models.length} placeholder="暂无评估模型" onChange={onSelect} options={models.map(model=>({value:model.run,label:model.title}))}/>}
  </div>}
  {!embedded&&simulation?.scene&&<p>场景：{simulation.scene.title}</p>}
  {error&&<Alert role="alert" tone="danger">{error}</Alert>}{loading&&<p role="status">正在读取回放…</p>}{!id&&<p>暂无已保存训练评估轨迹</p>}
  {originalId&&original?.id===originalId&&original.error&&<Alert role="alert" tone="danger">无法读取原录制轨迹：{original.error}</Alert>}
  {frame&&<Card className="card dataset-detail">
   {originalId&&<div className="go1-track-controls">
    <div className="training-actions">
     <Button aria-pressed={showOriginal} disabled={!originalFrames} onClick={()=>switchTrack(true)}>原录制回放</Button>
     <Button aria-pressed={!showOriginal} onClick={()=>switchTrack(false)}>本次回灌回放</Button>
    </div>
    <div className="go1-track-legend">
     <span><i className="go1-original-track"/>原录制路径</span><span><i className="go1-injected-track"/>回灌测试路径</span>
    </div>
   </div>}
   <div className={originalFrames?'go1-replay-comparison':undefined}>
   <Suspense fallback={<div className="go1-viewport go1-viewport-loading" role="status">正在创建三维视图…</div>}><Viewport scene={simulation?.scene} frame={frame} frames={frames} originalFrames={originalFrames} follow={follow} showEyes={showEyes} injectedVision={injected} hasDepth={depthAvailable} depthView={depthId ? <ReplayDepth key={depthKey} depthKey={depthKey} onReady={setReadyDepth} id={depthId} caseId={depthCase} index={depthIndex} count={activeFrames.length} recorded={!injected} hidden={!showDepth}/> : undefined} onFit={()=>setFollow(false)}/></Suspense>
   {originalFrames&&<ReplayMetrics original={originalFrames} candidate={frames} originalResult={original?.result} candidateResult={candidateResult}/>}
   </div>
   <div className="go1-playback-controls">
    <Button aria-pressed={showEyes} onClick={()=>setShowEyes(value=>!value)}>{(simulation?.visionInjected&&!showOriginal)?'机器人参考视角':'机器人眼睛'}</Button>
    {depthAvailable&&<Button aria-pressed={showDepth} onClick={()=>setShowDepth(value=>!value)}>深度帧</Button>}
    <Button disabled={activeFrames.length<2} onClick={()=>{if(time>=end)seek(start);setPlaying(value=>!value);}}>{playing?'暂停':'播放'}</Button>
    <Button onClick={()=>seek(start)}>重置</Button>
    <Choice id="go1-replay-speed" label="回放倍速" value={String(speed)} onChange={value=>setSpeed(Number(value))} options={[.25,.5,1,2].map(value=>({value:String(value),label:`${value}×`}))}/>
    <Button aria-pressed={follow} onClick={()=>setFollow(value=>!value)}>跟随机器人</Button>
   </div>
   <div className="go1-replay-timeline"><Slider min={start} max={Math.max(end,start+.001)} step={.001} value={time} disabled={activeFrames.length<2} onValueChange={value=>seek(Number(value))}>
    <SliderLabel>播放位置</SliderLabel><SliderControl><SliderTrack><SliderIndicator/></SliderTrack><SliderThumb aria-label="评估回放位置"/></SliderControl>
   </Slider></div>
   <div className="go1-replay-facts" role="status"><span>{time.toFixed(2)} / {end.toFixed(2)} 秒 · {activeFrames[index].episode_end?'回合结束':'回合中'}</span><span>采样 {index+1} / {activeFrames.length}</span><span>机身高度 {frame.position_m[2].toFixed(3)} m</span></div>
   <details><summary>姿态数据</summary><dl><dt>机身位置（m）</dt><dd>{frame.position_m.map(n=>n.toFixed(3)).join(' / ')}</dd><dt>朝向（w/x/y/z）</dt><dd>{frame.orientation_wxyz.map(n=>n.toFixed(3)).join(' / ')}</dd><dt>关节位置（rad）</dt><dd>{frame.joint_position_rad.map(n=>n.toFixed(3)).join(' / ')}</dd></dl></details>
  </Card>}
 </section>;
}
