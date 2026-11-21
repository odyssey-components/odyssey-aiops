import type {LessonId} from '../../core';
import {validId} from './training-contract';
import type {Mode} from './game';
export type MazeSearch = Record<string,unknown> & {course:LessonId;point:number;seed:number;mode:Mode;speed:number;model?:string;modelError?:string;tab?:'training'|'datasets'|'models'|'recordings'|'playback'|'reinjection'|'simulation';dataset?:string;task?:string;modelDetail?:string;module?:'go1';go1Tab?:'models'|'replay'|'scenes'|'test-case-sets'|'simulation'|'recording'|'vision-injection';go1Task?:string;go1Model?:string;go1Scene?:string;go1Simulation?:string;go1Replay?:string;go1ReplayCase?:string};
const safeRun=(value:unknown):value is string=>typeof value==='string' && value.length>0 && value.length<=200 && value!=='.' && value!=='..' && !/[\\/\x00]/.test(value);
const integer=(value:unknown):number|undefined => {
  if(typeof value!=='number' && (typeof value!=='string' || !/^\d+$/.test(value))) return undefined;
  const number=Number(value);return Number.isSafeInteger(number) ? number : undefined;
};
export function validateMazeSearch(raw:Record<string,unknown>):MazeSearch {
  const point=integer(raw.point),seed=integer(raw.seed),speed=integer(raw.speed);
  const model=safeRun(raw.model)?raw.model:undefined;
  const {modelError:_,tab:__,dataset:___,task:____,modelDetail:_____,module:______,go1Tab:_______,go1Task:________,go1Model:_________,go1Simulation:__________,go1Device:___________,go1Scene:____________,go1Replay:_____________,go1Regression:______________,go1ReplayCase:_______________,...rest}=raw;
  return {...rest,...(raw.module==='go1'?{module:'go1' as const,
    ...(raw.go1Tab==='models' || raw.go1Tab==='replay' || raw.go1Tab==='scenes' || raw.go1Tab==='test-case-sets' || raw.go1Tab==='simulation' || raw.go1Tab==='recording' || raw.go1Tab==='vision-injection'?{go1Tab:raw.go1Tab}:{}),
    ...((raw.go1Tab==='scenes'||raw.go1Tab===undefined) && validId(raw.go1Scene)?{go1Scene:raw.go1Scene}:{}),
    ...((raw.go1Tab==='simulation' || raw.go1Tab==='vision-injection' || raw.go1Tab==='models') && validId(raw.go1Replay)?{go1Replay:raw.go1Replay}:{}),
    ...((raw.go1Tab==='simulation'||raw.go1Tab==='vision-injection') && validId(raw.go1Replay) && validId(raw.go1ReplayCase)?{go1ReplayCase:raw.go1ReplayCase}:{}),
    ...((raw.go1Tab==='simulation' || raw.go1Tab==='vision-injection' || raw.go1Tab==='replay') && validId(raw.go1Simulation)?{go1Simulation:raw.go1Simulation}:{}),
    ...(validId(raw.go1Task)?{go1Task:raw.go1Task}:{}),...(validId(raw.go1Model)?{go1Model:raw.go1Model}:{})}:{}),course:raw.course==='random-map-v1' || raw.course==='local-map-v1' || raw.course==='memory-map-v1' ? raw.course : 'fixed-map-v1',
    point:point!==undefined && point>=1 && point<=5 ? point : 1,
    seed:seed!==undefined && seed>=0 && seed<=0xffffffff ? seed : 7,
    mode:raw.mode==='expert' || raw.mode==='model' ? raw.mode : 'manual',
    speed:speed===40 || speed===250 ? speed : 650,model,...(raw.tab==='training' && validId(raw.task)?{task:raw.task}:{}),...(raw.tab==='training' || raw.tab==='datasets' || raw.tab==='models' || raw.tab==='recordings' || raw.tab==='playback' || raw.tab==='reinjection' || raw.tab==='simulation'?{tab:raw.tab}:{}),
    ...(raw.tab==='datasets' && validId(raw.dataset)?{dataset:raw.dataset}:{}),
    ...(raw.tab==='models' && safeRun(raw.modelDetail)?{modelDetail:raw.modelDetail}:{}),
    ...(raw.model!==undefined && model===undefined ? {modelError:'URL 模型名称无效，请从默认目录重新选择模型。'} : {}),
  };
}
export const DEFAULT_SEARCH=validateMazeSearch({});
export const searchKey=(search:MazeSearch)=>JSON.stringify([search.course,search.point,search.seed,search.mode,search.speed,search.model,search.modelError,search.tab,search.dataset,search.task,search.modelDetail,search.module,search.go1Tab,search.go1Task,search.go1Model,search.go1Simulation,search.go1Replay,search.go1ReplayCase,search.go1Scene]);
