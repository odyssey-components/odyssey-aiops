import type {PageInfo} from '../utilities/list-pagination';
import type { LessonId } from '../../core';

export const TRAINING_LESSONS: readonly LessonId[] = ['fixed-map-v1','random-map-v1','local-map-v1','memory-map-v1'];
export const validId = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/.test(value);
export const isLesson = (value: unknown): value is LessonId => TRAINING_LESSONS.includes(value as LessonId);
const integer = (value: unknown, low: number, high: number): value is number => typeof value === 'number' && Number.isInteger(value) && value >= low && value <= high;
export type TrainingCommand =
  | {action:'prepare';course:LessonId;seed:number;maps:number;title?:string;description?:string}
  | {action:'train';course:LessonId;dataset:string;epochs:number;seed:number}
  | {action:'cancel';id:string};
export function parseTrainingCommand(value: unknown): TrainingCommand {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('训练参数无效');
  const data=value as Record<string,unknown>;
  if(data.action==='cancel' && Object.keys(data).every(key=>['action','id'].includes(key)) && validId(data.id)) return {action:'cancel',id:data.id};
  if(!isLesson(data.course) || !integer(data.seed,0,2147483647)) throw new Error('课时或种子无效');
  if(data.action==='prepare' && Object.keys(data).every(key=>['action','course','seed','maps','title','description'].includes(key)) && integer(data.maps,16,512)
    && (data.title===undefined || typeof data.title==='string' && data.title.length<=80)
    && (data.description===undefined || typeof data.description==='string' && data.description.length<=500)) {
    return {action:'prepare',course:data.course,seed:data.seed,maps:data.maps,
      ...(data.title===undefined?{}:{title:data.title as string}),...(data.description===undefined?{}:{description:data.description as string})};
  }
  if(data.action==='train' && Object.keys(data).every(key=>['action','course','seed','dataset','epochs'].includes(key)) && validId(data.dataset) && integer(data.epochs,0,1000)) {
    return {action:'train',course:data.course,seed:data.seed,dataset:data.dataset,epochs:data.epochs};
  }
  throw new Error('参数超出范围：地图 16–512 张，轮数 0–1000；必须先准备本课地图集');
}
export interface DatasetInfo {id:string;course:LessonId;seed:number;maps:number;validation:number;test:number;sha256:string;createdAt:string;title:string;description:string;}
export interface TrainingMetric {
  epoch:number;total:number;successRate:number;evaluatedMaps:number;split:'fixed'|'validation';loss?:number;accuracy?:number;
}
export interface TrainingJob {

  id:string;course:LessonId;action:'prepare'|'train';state:'queued'|'running'|'cancelling'|'finalizing'|'succeeded'|'failed'|'cancelled';
  startedAt:string;finishedAt?:string;logs:string;metrics?:TrainingMetric[];error?:string;dataset?:string;datasetSha256?:string;run?:string;
  epochs?:number;seed?:number;datasetTitle?:string;
}
export interface TrainingSnapshot {pagination?:PageInfo;datasetDetail?:DatasetInfo;datasets:DatasetInfo[];job:TrainingJob|null;active:TrainingJob|null;tasks?:TrainingJob[];detail?:TrainingJob|null;}
export const busyJob = (job: TrainingJob | null | undefined) => job?.state === 'queued' || job?.state === 'running' || job?.state === 'cancelling' || job?.state === 'finalizing';
