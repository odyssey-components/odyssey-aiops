import {apiPath} from '../../../apis/path';
import {useEffect,useRef,useState} from 'react';
import {paintDepth} from '../views/RecordedDepth';

export function ReplayDepth({id,caseId,index,count,recorded=false,hidden=false,depthKey,onReady}:{id:string;caseId?:string;index:number;count:number;recorded?:boolean;hidden?:boolean;depthKey:string;onReady:(key?:string)=>void}) {
 const canvas=useRef<HTMLCanvasElement>(null);
 const [data,setData]=useState<{width:number;height:number;frames:DataView}>();
 useEffect(()=>{
  onReady();
  const controller=new AbortController();
  const query=new URLSearchParams({[recorded?'recorded-depth':'injected-vision']:id});
  if(caseId)query.set('case',caseId);
  void fetch(apiPath('/api/go1?'+query),{signal:controller.signal})
   .then(async response=>{
    if(!response.ok)throw Error(recorded?'无录制深度帧':'无法读取回灌视觉输入');
    const bytes=new DataView(await response.arrayBuffer());
    const width=bytes.getUint32(0,true),height=bytes.getUint32(4,true);
    if(width<16||width>128||height<16||height>128||bytes.byteLength!==8+width*height*4*count)throw Error('深度帧尺寸不匹配');
    if(!controller.signal.aborted){setData({width,height,frames:bytes});onReady(depthKey);}
   }).catch(()=>{});
  return ()=>controller.abort();
 },[id,caseId,count,recorded,depthKey,onReady]);
 useEffect(()=>{
  if(!data||!canvas.current)return;
  const length=data.width*data.height*4;
  paintDepth(canvas.current,data.width,data.height,new DataView(data.frames.buffer,8+length*index,length));
 },[data,index]);
 if(!data)return null;
 return <div className="go1-eyes-inset go1-injected-inset" hidden={hidden}>
  <canvas ref={canvas} role="img" aria-label={recorded?"录制深度帧":"回灌视觉输入"} data-depth-time={(index/50).toFixed(2)}/>
  <span>{recorded?`录制深度 · ${(index/50).toFixed(2)} 秒`:'回灌视觉 · 策略输入'}</span>
 </div>;
}
