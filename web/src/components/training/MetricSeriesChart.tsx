import {useState} from 'react';
import {Card} from '@odyssey/ui/primitives/card';
import {Button} from '@odyssey/ui/primitives/button';
export function MetricSeriesChart({title,unit='',points}:{title:string;unit?:string;points:{step:number;value:number}[]}) {
 const [selected,setSelected]=useState(0);const values=points.filter(point=>Number.isFinite(point.value) && Number.isFinite(point.step));
 const min=Math.min(...values.map(p=>p.value)),max=Math.max(...values.map(p=>p.value)),span=max-min || 1;
 const first=values[0]?.step ?? 0,last=values.at(-1)?.step ?? 1;
 const coordinates=values.map(p=>`${40+(p.step-first)/Math.max(1,last-first)*560},${160-(p.value-min)/span*130}`).join(' ');
 const point=values[Math.min(selected,values.length-1)];
 function csv() {const blob=new Blob(['\uFEFFstep,value\n'+values.map(p=>`${p.step},${p.value}`).join('\n')],{type:'text/csv;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='metrics.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
 return <Card className="card training-chart"><div className="dataset-heading"><h3>{title}{unit?` (${unit})`:''}</h3><Button size="sm" disabled={!values.length} onClick={csv}>导出指标 CSV</Button></div>
 {point?<><svg viewBox="0 0 640 190" role="img" aria-label={`${title}，${values.length}个实测点`} style={{width:'100%'}}><text x="0" y="22" fill="currentColor" fontSize="10">{max.toPrecision(4)}</text><text x="0" y="165" fill="currentColor" fontSize="10">{min.toPrecision(4)}</text><polyline fill="none" stroke="currentColor" strokeWidth="2" points={coordinates}/>{values.length===1 && <circle cx="40" cy="160" r="4" fill="currentColor"/>}</svg>
 <label>查看迭代<InputRange max={values.length-1} value={Math.min(selected,values.length-1)} onChange={setSelected}/></label><p>第 {point.step} 次：{point.value} {unit}</p></>:<p>暂无实测指标</p>}</Card>;
}
function InputRange({max,value,onChange}:{max:number;value:number;onChange:(value:number)=>void}) {return <input aria-label="指标采样点" type="range" min="0" max={max} step="1" value={value} onChange={e=>onChange(Number(e.currentTarget.value))} style={{width:'100%'}}/>;}
