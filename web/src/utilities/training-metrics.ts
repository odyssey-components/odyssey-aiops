import type { LessonId } from '../../core';
import type { TrainingMetric } from '../utilities/training-contract';

const PREFIX='ODYMAZE_METRIC ';
const MAX_LINE=16384;
interface Expected {course:LessonId;total:number;}
const integer=(value:unknown,low:number,high:number):value is number=>typeof value==='number' && Number.isInteger(value) && value>=low && value<=high;
const unit=(value:unknown):value is number=>typeof value==='number' && Number.isFinite(value) && value>=0 && value<=1;

export function parseTrainingMetric(line:string,expected?:Expected):TrainingMetric|null {
  if(!line.startsWith(PREFIX) || line.length>MAX_LINE) return null;
  try {
    const value=JSON.parse(line.slice(PREFIX.length));
    if(!value || typeof value!=='object' || Array.isArray(value)
      || !integer(value.total,0,5000) || !integer(value.epoch,0,value.total) || !unit(value.successRate)
      || !integer(value.evaluatedMaps,1,64) || !['fixed','validation'].includes(value.split)
      || value.evaluatedMaps!==(value.split==='fixed'?1:64)
      || (value.loss!==undefined && (typeof value.loss!=='number' || !Number.isFinite(value.loss) || value.loss<0))
      || (value.accuracy!==undefined && !unit(value.accuracy))) return null;
    if(expected && (value.total!==expected.total || value.split!==(expected.course==='fixed-map-v1'?'fixed':'validation'))) return null;
    return {epoch:value.epoch,total:value.total,successRate:value.successRate,evaluatedMaps:value.evaluatedMaps,split:value.split,
      ...(value.loss===undefined?{}:{loss:value.loss}),...(value.accuracy===undefined?{}:{accuracy:value.accuracy})};
  } catch {return null;}
}

export function upsertTrainingMetric(metrics:TrainingMetric[],metric:TrainingMetric) {
  const index=metrics.findIndex(point=>point.epoch===metric.epoch);
  if(index<0) {metrics.push(metric);metrics.sort((a,b)=>a.epoch-b.epoch);} else metrics[index]=metric;
}

/** Old human logs contain sampled measurements only; missing epochs remain missing. */
export function extractLegacyMetrics(logs:string,course:LessonId,total?:number):TrainingMetric[] {
  const points:TrainingMetric[]=[];
  for(const match of logs.matchAll(/(?:^|\n)epoch (\d+)\/(\d+);[^\r\n]*?validation success ([\d.]+)%/g)) {
    const epoch=Number(match[1]),count=Number(match[2]),success=Number(match[3])/100;
    if(!integer(count,0,1000) || !integer(epoch,0,count) || !unit(success) || (total!==undefined && count!==total)) continue;
    upsertTrainingMetric(points,{epoch,total:count,successRate:success,evaluatedMaps:course==='fixed-map-v1'?1:64,split:course==='fixed-map-v1'?'fixed':'validation'});
  }
  return points;
}

/** Add newly observed legacy samples without replacing exact structured measurements. */
export function mergeLegacyMetrics(metrics:TrainingMetric[],logs:string,course:LessonId,total?:number) {
  const epochs=new Set(metrics.map(point=>point.epoch));
  for(const point of extractLegacyMetrics(logs,course,total)) {
    if(!epochs.has(point.epoch)) {upsertTrainingMetric(metrics,point);epochs.add(point.epoch);}
  }
  return metrics;
}

/** stdout only: bounded partial lines cannot consume unbounded memory or mix with stderr. */
export function createTrainingMetricCollector(options:Expected & {onLog:(text:string)=>void;onMetric:(metric:TrainingMetric)=>void}) {
  let pending='',discarding=false;
  const line=(text:string,newline:boolean)=>{
    if(text.startsWith(PREFIX)) {const metric=parseTrainingMetric(text,options);if(metric) options.onMetric(metric);}
    else options.onLog(text+(newline?'\n':''));
  };
  return {
    push(chunk:string) {
      for(const part of chunk.split(/(?<=\n)/)) {
        const complete=part.endsWith('\n');
        if(discarding) {if(complete) discarding=false;continue;}
        pending+=part;
        if(pending.length>MAX_LINE) {
          if(!pending.startsWith(PREFIX)) options.onLog(pending.slice(-MAX_LINE));
          pending='';discarding=!complete;continue;
        }
        if(complete) {line(pending.replace(/\r?\n$/,''),true);pending='';}
      }
    },
    end() {if(pending && !discarding) line(pending,false);pending='';discarding=false;},
  };
}
