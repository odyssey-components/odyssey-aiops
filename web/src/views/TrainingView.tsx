import {tablePagination} from '../utilities/list-pagination';
import {getTrainingSnapshot,submitTrainingCommand} from '../apis/training';
import {Alert} from '@odyssey/ui/primitives/alert';
import {Button} from '@odyssey/ui/primitives/button';
import {EmptyState,EmptyStateTitle} from '@odyssey/ui/primitives/empty-state';
import {Table,TableHeader,TableBody,TableRow,TableHead,TableCell} from '@odyssey/ui/primitives/table';
import {useEffect,useRef,useState} from 'react';
import type {LessonId} from '../../core';
import {busyJob,type TrainingCommand,type TrainingSnapshot} from '../utilities/training-contract';
import {courseTasks,trainingStates,trainingTime} from '../utilities/training-tasks';
import {NewTrainingDialog} from '../components/training/NewTrainingDialog';
import {TrainingTaskDetail} from '../components/training/TrainingTaskDetail';

export function TrainingView({course,onFinished,selectedTask,onSelectTask}:{course:LessonId;onFinished:()=>Promise<void>;selectedTask?:string;onSelectTask:(id?:string)=>void}) {
 const [pageSize,setPageSize]=useState(20);
 const [page,setPage]=useState(1);
 const [snapshot,setSnapshot]=useState<TrainingSnapshot|null>(null),[open,setOpen]=useState(false);
 const [readyTask,setReadyTask]=useState<string|undefined>();
 const [error,setError]=useState(''),[pollError,setPollError]=useState(''),[pending,setPending]=useState(false);
 const completed=useRef(''),finished=useRef(onFinished);finished.current=onFinished;
 useEffect(()=>{
  setReadyTask(undefined);
  let disposed=false;let timer:ReturnType<typeof setTimeout>;const controller=new AbortController();
  async function poll() {
   try {
    const value=await getTrainingSnapshot(course,controller.signal,selectedTask,{kind:"tasks",page,pageSize});
    if(disposed) return;setSnapshot(value);setReadyTask(selectedTask);setPollError('');
    const success=courseTasks(value,course).find(job=>job.state==='succeeded' && job.run);
    if(success && completed.current!==success.id) {await finished.current();if(!disposed) completed.current=success.id;}
   } catch(error) {if(!disposed) setPollError(error instanceof Error?error.message:'无法读取任务');}
   finally {if(!disposed) timer=setTimeout(()=>void poll(),1500);}
  }
  void poll();return ()=>{disposed=true;controller.abort();clearTimeout(timer);};
 },[course,selectedTask,page,pageSize]);
 const tasks=courseTasks(snapshot,course);
 const detail=selectedTask && readyTask===selectedTask ? (snapshot?.detail?.id===selectedTask ? snapshot.detail : snapshot?.detail===null ? undefined : tasks.find(job=>job.id===selectedTask)) : undefined;
 const blocked=pending || busyJob(snapshot?.active) || !snapshot || !!pollError;
 async function submit(command:TrainingCommand) {
  setPending(true);setError('');
  try {
   const result=await submitTrainingCommand(command);
   setSnapshot(previous=>({datasets:previous?.datasets ?? [],job:result,active:busyJob(result)?result:null,
    tasks:[result,...(previous?.tasks ?? []).filter(job=>job.id!==result.id)],detail:result}));
   if(command.action==='train') {setOpen(false);onSelectTask(result.id);}
  } catch(error) {setError(error instanceof Error?error.message:'任务请求失败');}
  finally {setPending(false);}
 }
 return <section className="training-tasks-workspace" aria-label="本课训练任务">
  {selectedTask ? detail ? <TrainingTaskDetail job={detail} onBack={()=>onSelectTask()} pending={pending} onCancel={()=>void submit({action:'cancel',id:detail.id})} />
   : <><Button id="back-training-tasks" onClick={()=>onSelectTask()}>返回任务列表</Button><p role="status">{readyTask===selectedTask?'本课找不到此训练任务。':'正在读取任务详情…'}</p></>
   : <>
    <div className="dataset-heading"><h2>训练任务</h2><Button id="add-training-task" variant="solid" onClick={()=>{setError('');setOpen(true);}}>添加训练任务</Button></div>
    {!snapshot && !pollError && <p role="status">正在读取训练任务…</p>}
    {snapshot && !tasks.length && <EmptyState><EmptyStateTitle>本课暂无训练任务，点击「添加训练任务」创建。</EmptyStateTitle></EmptyState>}
    <Table pagination={tablePagination(snapshot?.pagination,setPage,size=>{setPageSize(size);setPage(1);})} aria-label="本课训练任务列表" containerClassName="training-task-list" containerProps={{tabIndex:0}}>
     <TableHeader><TableRow>{['任务','地图组','训练轮数','初始化种子','状态','开始时间','完成时间','操作'].map(label=><TableHead key={label} scope="col">{label}</TableHead>)}</TableRow></TableHeader>
     <TableBody>{tasks.map(job=><TableRow key={job.id} data-task-id={job.id}>
      <TableHead scope="row">{job.id.slice(0,8)}</TableHead>
      <TableCell>{job.datasetTitle ?? snapshot?.datasets.find(entry=>entry.id===job.dataset)?.title ?? job.dataset ?? '—'}</TableCell>
      <TableCell>{job.epochs ?? '—'}</TableCell><TableCell>{job.seed ?? '—'}</TableCell><TableCell>{trainingStates[job.state]}</TableCell>
      <TableCell><time dateTime={job.startedAt}>{trainingTime(job.startedAt)}</time></TableCell><TableCell>{trainingTime(job.finishedAt)}</TableCell>
      <TableCell><Button size="sm" variant="ghost" aria-label={`查看训练任务 ${job.id}`} onClick={()=>onSelectTask(job.id)}>详情</Button></TableCell>
     </TableRow>)}</TableBody>
    </Table>
    {snapshot?.active && <p role="status">{snapshot.active.course===course?'有任务正在运行，可进入详情查看。':'其他课时有任务正在运行，请等待完成后再添加。'}</p>}
   </>}
  {(error || pollError) && !open && <Alert tone="danger" role="alert">{error || pollError}</Alert>}
  <NewTrainingDialog key={course} open={open} onOpenChange={setOpen} course={course} datasets={snapshot?.datasets ?? []} blocked={blocked} error={error || pollError || (busyJob(snapshot?.active)?'已有任务正在运行，请等待完成后再添加。':'')}
   onSubmit={command=>void submit(command)} />
 </section>;
}
