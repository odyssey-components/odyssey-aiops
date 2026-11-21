import {apiPath} from './path';
import type {LessonId} from '../../core';
import type {TrainingCommand,TrainingJob,TrainingSnapshot} from '../utilities/training-contract';

export async function getTrainingSnapshot(course:LessonId,signal?:AbortSignal,task?:string,list?:{kind:"tasks"|"datasets";page:number;pageSize?:number;dataset?:string}):Promise<TrainingSnapshot> {
  const query=new URLSearchParams({course});
  if(task)query.set('task',task);
  if(list){query.set('list',list.kind);query.set('page',String(list.page));if(list.pageSize)query.set('pageSize',String(list.pageSize));if(list.dataset)query.set('dataset',list.dataset);}
  const response=await fetch(apiPath('/api/training?'+query),{signal});
  if(!response.ok) throw new Error('无法读取本地训练状态，请确认服务与 Python 环境');
  return response.json();
}

export async function submitTrainingCommand(command:TrainingCommand):Promise<TrainingJob> {
  const response=await fetch(apiPath('/api/training'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(command)});
  const result=await response.json() as TrainingJob & {error?:string};
  if(!response.ok) throw new Error(result.error ?? '无法启动任务');
  return result;
}
