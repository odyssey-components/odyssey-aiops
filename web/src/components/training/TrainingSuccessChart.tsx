import {Stat,StatLabel,StatValue} from '@odyssey/ui/primitives/stat';
import {EmptyState,EmptyStateTitle} from '@odyssey/ui/primitives/empty-state';
import {Slider,SliderLabel,SliderControl,SliderTrack,SliderIndicator,SliderThumb} from '@odyssey/ui/primitives/slider';
import {useEffect,useRef,useState} from 'react';
import {Card} from '@odyssey/ui/primitives/card';
import {Button} from '@odyssey/ui/primitives/button';
import type {LessonId} from '../../../core';
import type {TrainingMetric} from '../../utilities/training-contract';
import {chartMetrics,metricSummary,metricsCsv,nearestMetric} from '../../utilities/training-chart-data';

const percent=(rate:number)=>(rate*100).toFixed(2)+'%';
export function TrainingSuccessChart({metrics,course}:{metrics:readonly TrainingMetric[];course:LessonId}) {
  const points=chartMetrics(metrics),summary=metricSummary(points);
  const hasPoints=points.length>0;
  const [activeEpoch,setActiveEpoch]=useState<number|null>(null),[width,setWidth]=useState(720);
  const plot=useRef<HTMLDivElement>(null);
  useEffect(()=>{
    const element=plot.current;if(!element)return;
    const measure=()=>{const size=element.getBoundingClientRect().width;if(size>0)setWidth(Math.max(280,Math.round(size)));};
    measure();const observer=new ResizeObserver(measure);observer.observe(element);return ()=>observer.disconnect();
  },[hasPoints]);
  const title=course==='fixed-map-v1'?'固定地图通关率':'验证集通关率';
  const active=activeEpoch===null?summary?.latest:nearestMetric(points,activeEpoch);
  const maximum=Math.max(1,summary?.latest.epoch??0),left=48,right=16,top=18,bottom=218;
  const x=(epoch:number)=>left+epoch/maximum*(width-left-right);
  const y=(rate:number)=>bottom-rate*(bottom-top);
  const ticks=[...new Set([0,...[.25,.5,.75,1].map(fraction=>Math.round((summary?.latest.epoch??0)*fraction))])];
  function download() {
    const url=URL.createObjectURL(new Blob(['\uFEFF'+metricsCsv(points)],{type:'text/csv;charset=utf-8'}));
    const anchor=document.createElement('a');anchor.href=url;anchor.download=`${course}-success.csv`;anchor.click();setTimeout(()=>URL.revokeObjectURL(url),0);
  }
  return <Card className="card training-chart" aria-label={title}>
    <div className="card-heading"><h2>{title}</h2><Button size="sm" disabled={!points.length} onClick={download}>导出 CSV</Button></div>
    {summary?<>
      <div className="chart-summary">
        <Stat><StatLabel>最新</StatLabel><StatValue id="success-latest">{percent(summary.latest.successRate)}</StatValue></Stat>
        <Stat><StatLabel>最佳</StatLabel><StatValue id="success-best">{percent(summary.best.successRate)}</StatValue></Stat>
        <Stat><StatLabel>最佳轮次</StatLabel><StatValue id="success-best-epoch">{summary.best.epoch}</StatValue></Stat>
        <Stat><StatLabel>较初始（百分点）</StatLabel><StatValue id="success-change">{summary.change===null?'—':`${summary.change>0?'+':''}${summary.change.toFixed(2)}`}</StatValue></Stat>
      </div>
      <div ref={plot} className="success-plot">
        <svg id="success-chart" viewBox={`0 0 ${width} 256`} role="img" aria-label={`${title}，共${points.length}个实测点`}
          onPointerMove={event=>{const box=event.currentTarget.getBoundingClientRect();const epoch=((event.clientX-box.left)/box.width*width-left)/(width-left-right)*maximum;setActiveEpoch(nearestMetric(points,epoch)?.epoch??null);}}
          onPointerLeave={()=>setActiveEpoch(null)}>
          <title>{title}</title>
          {[0,.25,.5,.75,1].map(rate=><g key={rate}><line x1={left} x2={width-right} y1={y(rate)} y2={y(rate)} className="chart-grid"/><text x={left-8} y={y(rate)+4} textAnchor="end">{rate*100}%</text></g>)}
          {ticks.map(epoch=><text key={epoch} x={x(epoch)} y={bottom+22} textAnchor="middle">{epoch}</text>)}
          <text x={width-right} y={252} textAnchor="end">轮次</text>
          <polyline id="success-line" points={points.map(point=>`${x(point.epoch)},${y(point.successRate)}`).join(' ')} fill="none" className="success-line"/>
          <circle className="best-point" cx={x(summary.best.epoch)} cy={y(summary.best.successRate)} r="4"/>
          {active&&<g><line className="chart-cursor" x1={x(active.epoch)} x2={x(active.epoch)} y1={top} y2={bottom}/><circle className="active-point" cx={x(active.epoch)} cy={y(active.successRate)} r="5"/></g>}
        </svg>
      </div>
      <div id="success-detail" className="success-detail" role="status">{active&&<>第 {active.epoch} / {active.total} 轮 · {percent(active.successRate)} · {active.evaluatedMaps} 张地图</>}</div>
      <Slider id="success-epoch" min={0} max={Math.max(1,points.length-1)} step={1} disabled={points.length<2}
        value={Math.max(0,points.findIndex(point=>point.epoch===active?.epoch))}
        onValueChange={value=>{const point=points[Number(value)];if(point)setActiveEpoch(point.epoch);}}>
        <SliderLabel>查看通关率轮次</SliderLabel>
        <SliderControl><SliderTrack><SliderIndicator /></SliderTrack><SliderThumb /></SliderControl>
      </Slider>
    </>:<EmptyState role="status"><EmptyStateTitle>暂无通关率数据</EmptyStateTitle></EmptyState>}
  </Card>;
}
