import {useLocalTablePagination} from '../../../utilities/list-pagination';
import {useMemo} from 'react';
import {Table,TableHeader,TableBody,TableRow,TableHead,TableCell} from '@odyssey/ui/primitives/table';
import type {Go1Frame} from '../contract';
import {go1Evaluation} from '../outcomes';
import {compareTrajectories,summarizeTrajectory} from '../replay/comparison';

const format=(value:number|undefined,unit:string)=>value===undefined?'未记录':`${value.toFixed(3)} ${unit}`;
const number=(value:unknown)=>typeof value==='number'&&Number.isFinite(value)?value:undefined;

const outcome=(result?:Record<string,unknown>)=>result?.vision_input_exhausted===true?'录制视觉输入已用完':go1Evaluation(result);

export function ReplayMetrics({original,candidate,originalResult,candidateResult}:{
 original:readonly Go1Frame[];candidate:readonly Go1Frame[];
 originalResult?:Record<string,unknown>;candidateResult?:Record<string,unknown>;
}) {
 const baseline=useMemo(()=>summarizeTrajectory(original),[original]);
 const current=useMemo(()=>summarizeTrajectory(candidate),[candidate]);
 const comparison=useMemo(()=>compareTrajectories(original,candidate),[original,candidate]);
 const baselineOutcome=outcome(originalResult);
 const candidateOutcome=outcome(candidateResult);
 const rows:[string,string,string,string][]=[
  ['终止原因',baselineOutcome,candidateOutcome,originalResult&&candidateResult?(baselineOutcome===candidateOutcome?'一致':'不同'):'—'],
 ];
 function metric(label:string,a:number|undefined,b:number|undefined,unit:string){
  rows.push([label,format(a,unit),format(b,unit),a===undefined||b===undefined?'—':`${b>a?'+':''}${format(b-a,unit)}`]);
 }
 metric('终止时刻',original.at(-1)?.time,candidate.at(-1)?.time,'s');
 metric('最终进度',number(originalResult?.final_progress_m),number(candidateResult?.final_progress_m),'m');
 const endpoint=(frames:readonly Go1Frame[])=>frames.at(-1)!.position_m.slice(0,2).map(value=>value.toFixed(3)).join(', ');
 rows.push(['终点 (x, y) / 平面偏差',endpoint(original),endpoint(candidate),format(comparison.endpoint,'m')]);
 metric('平面路径长度',baseline.length,current.length,'m');
 metric('横滚峰值',baseline.roll,current.roll,'°');
 metric('俯仰峰值',baseline.pitch,current.pitch,'°');
 if(comparison.overlap){
  rows.push(['逐时刻平面偏差 · 均值','—','—',format(comparison.overlap.mean,'m')]);
  rows.push(['逐时刻平面偏差 · 峰值','—','—',format(comparison.overlap.peak,'m')]);
 }
 const pagination=useLocalTablePagination(rows.length);
 return <section className="go1-replay-metrics" aria-label="回灌轨迹指标">
  <h3>轨迹指标对比</h3>
  <Table pagination={pagination.options} aria-label="原录制与回灌指标" containerClassName="dataset-list">
   <TableHeader><TableRow>{['指标','原录制','本次回灌','差值'].map(label=><TableHead key={label} scope="col">{label}</TableHead>)}</TableRow></TableHeader>
   <TableBody>{pagination.slice(rows).map(([label,a,b,delta])=><TableRow key={label}><TableHead scope="row">{label}</TableHead><TableCell>{a}</TableCell><TableCell>{b}</TableCell><TableCell>{delta}</TableCell></TableRow>)}</TableBody>
  </Table>
  <p>{comparison.overlap?`逐时刻偏差按共同时间段采样 ${comparison.overlap.start.toFixed(2)}–${comparison.overlap.end.toFixed(2)} 秒。`:'两条轨迹没有共同时间段。'}</p>
 </section>;
}
