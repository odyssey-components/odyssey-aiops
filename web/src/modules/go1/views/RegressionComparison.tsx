import {useLocalTablePagination} from '../../../utilities/list-pagination';
import {useEffect,useRef,useState} from 'react';
import {Alert} from '@odyssey/ui/primitives/alert';
import {Button} from '@odyssey/ui/primitives/button';
import {Table,TableHeader,TableBody,TableRow,TableHead,TableCell} from '@odyssey/ui/primitives/table';
import {go1RegressionDepth,go1RegressionReport} from '../api';
import type {Go1RegressionStep} from '../contract';

function RegressionDepth({id,step}:{id:string;step:number}) {
 const canvas=useRef<HTMLCanvasElement>(null);
 const [status,setStatus]=useState('正在读取深度观测…');
 useEffect(()=>{
  const abort=new AbortController();
  go1RegressionDepth(id,step,abort.signal).then(({width,height,f32le})=>{
   if(abort.signal.aborted)return;
   if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||width>128||height>128)throw Error('深度帧尺寸无效');
   const bytes=Uint8Array.from(atob(f32le),character=>character.charCodeAt(0));
   if(bytes.length!==width*height*4)throw Error('深度帧长度无效');
   const context=canvas.current?.getContext('2d');
   if(!context)return;
   context.canvas.width=width;context.canvas.height=height;
   const data=new DataView(bytes.buffer),image=context.createImageData(width,height);
   for(let i=0;i<width*height;i++){
    const value=255*Math.max(0,Math.min(1,data.getFloat32(i*4,true)));
    image.data.set([value,value,value,255],i*4);
   }
   context.putImageData(image,0,0);setStatus('');
  }).catch(error=>{if(!abort.signal.aborted)setStatus(String(error));});
  return ()=>abort.abort();
 },[id,step]);
 return <><canvas ref={canvas} role="img" aria-label="待测模型深度输入" hidden={!!status} style={{width:256,maxWidth:'100%',height:'auto',imageRendering:'pixelated'}}/>
  {status&&<p role="status">{status}</p>}</>;
}

export function RegressionComparison({id}:{id:string}) {
 const [steps,setSteps]=useState<Go1RegressionStep[]>([]);
 const [index,setIndex]=useState(0);
 const [error,setError]=useState('');
 useEffect(()=>{
  const abort=new AbortController();
  setSteps([]);setIndex(0);setError('');
  go1RegressionReport(id,abort.signal).then(report=>{
   if(!abort.signal.aborted)setSteps(report.steps);
  }).catch(error=>{if(!abort.signal.aborted)setError(String(error));});
  return ()=>abort.abort();
 },[id]);
 const row=steps[index];
 const pagination=useLocalTablePagination(row?.baseline_action.length??0);
 if(error)return <Alert role="alert" tone="danger">{error}</Alert>;
 if(!row)return <p role="status">正在读取动作对比…</p>;
 const actionDeltas=steps.map(step=>step.max_abs_delta).sort((a,b)=>a-b);
 const p95=actionDeltas[Math.ceil(actionDeltas.length*.95)-1];
 return <>
  <dl aria-label="动作回灌指标" className="go1-scene-facts">
   <div><dt>比较步数</dt><dd>{steps.length}</dd></div>
   <div><dt>逐步最大动作差 P95</dt><dd>{p95.toPrecision(4)}</dd></div>
   <div><dt>全程最大动作差</dt><dd>{actionDeltas.at(-1)?.toPrecision(4)}</dd></div>
  </dl>
  <h4>动作对比</h4>
  <div className="training-actions">
   <Button disabled={index===0} onClick={()=>setIndex(index-1)}>上一步</Button>
   <label>步数 <input type="number" min={1} max={steps.length} value={index+1}
    onChange={event=>{const step=event.currentTarget.valueAsNumber;if(Number.isInteger(step)&&step>=1&&step<=steps.length)setIndex(step-1);}}/></label>
   <Button disabled={index===steps.length-1} onClick={()=>setIndex(index+1)}>下一步</Button>
  </div>
  <p>第 {row.step} / {steps.length} 步 · {row.time_seconds.toFixed(2)} 秒 · 最大动作差 {row.max_abs_delta.toPrecision(4)}</p>
  <h4>待测模型输入观测</h4>
  {row.candidate_state?<details open><summary>机身状态（{row.candidate_state.length} 维）</summary>
   <pre className="training-logs">{JSON.stringify(row.candidate_state)}</pre>
  </details>:<p>这份历史报告未保存输入观测。重新回灌后可查看。</p>}
  {row.has_depth&&<details><summary>深度帧（灰度显示归一化输入）</summary>
   <RegressionDepth key={`${id}:${row.step}`} id={id} step={row.step}/>
  </details>}
  <Table pagination={pagination.options} aria-label="逐关节动作对比"><TableHeader><TableRow>{['关节索引','录制模型','待测模型','动作差'].map(label=><TableHead key={label}>{label}</TableHead>)}</TableRow></TableHeader>
   <TableBody>{pagination.slice(row.baseline_action.map((baseline,joint)=>({baseline,joint}))).map(({baseline,joint})=><TableRow key={joint}><TableHead scope="row">{joint+1}</TableHead><TableCell>{baseline.toPrecision(5)}</TableCell><TableCell>{row.candidate_action[joint].toPrecision(5)}</TableCell><TableCell>{row.action_delta[joint].toPrecision(5)}</TableCell></TableRow>)}</TableBody>
  </Table>
 </>;
}
