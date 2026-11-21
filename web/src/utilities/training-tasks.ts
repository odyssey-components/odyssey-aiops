import type {LessonId} from '../../core';
import type {TrainingJob,TrainingSnapshot} from './training-contract';
export const trainingStates:Record<TrainingJob['state'],string>={queued:'排队中',running:'运行中',cancelling:'正在取消',finalizing:'正在保存产物',succeeded:'已完成',failed:'失败',cancelled:'已取消'};
export const trainingTime=(value?:string)=>value?new Date(value).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23'}):'—';
export function courseTasks(snapshot:TrainingSnapshot|null,course:LessonId):TrainingJob[] {
 const tasks=new Map<string,TrainingJob>();
 for(const job of [...(snapshot?.tasks ?? []),snapshot?.job,snapshot?.active]) if(job?.action==='train' && job.course===course) tasks.set(job.id,job);
 return [...tasks.values()].sort((a,b)=>Date.parse(b.startedAt)-Date.parse(a.startedAt));
}
