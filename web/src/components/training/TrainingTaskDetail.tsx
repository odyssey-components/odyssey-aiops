import {Button} from '@odyssey/ui/primitives/button';
import {Card} from '@odyssey/ui/primitives/card';
import {Code} from '@odyssey/ui/primitives/code';
import {Progress} from '@odyssey/ui/primitives/progress';
import type {TrainingJob} from '../../utilities/training-contract';
import {busyJob} from '../../utilities/training-contract';
import {trainingStates,trainingTime} from '../../utilities/training-tasks';
import {TrainingSuccessChart} from './TrainingSuccessChart';

export function TrainingTaskDetail({job,onBack,onCancel,pending}:{job:TrainingJob;onBack:()=>void;onCancel:()=>void;pending:boolean}) {
 const progress=[...job.logs.matchAll(/(?:epoch\s+|轮次\s*)(\d+)\s*\/\s*(\d+)/g)].at(-1);
 return <section className="training-task-detail" aria-label="训练任务详情">
  <div className="dataset-heading"><div><h2>训练任务详情</h2><p>{job.id}</p></div><Button id="back-training-tasks" onClick={onBack}>返回任务列表</Button></div>
  <Card className="card">
   <div id="training-status" role="status">模型训练：{trainingStates[job.state]}</div>
   <dl className="training-task-metadata">
    <div><dt>地图组</dt><dd>{job.datasetTitle ?? job.dataset ?? '—'}</dd></div>
    <div><dt>训练轮数</dt><dd>{job.epochs ?? '—'}</dd></div><div><dt>初始化种子</dt><dd>{job.seed ?? '—'}</dd></div>
    <div><dt>开始时间</dt><dd>{trainingTime(job.startedAt)}</dd></div><div><dt>完成时间</dt><dd>{trainingTime(job.finishedAt)}</dd></div>
   </dl>
   {progress && <Progress label="训练轮次进度" value={Number(progress[1])} max={Number(progress[2])} />}
   {job.error && <p role="alert">{job.error}</p>}
   {job.run && <p>模型已保存：{job.run}。可在「实验」中查看和运行。</p>}
   {busyJob(job) && job.state!=='finalizing' && <Button id="cancel-training" disabled={pending || job.state==='cancelling'} onClick={onCancel}>取消任务</Button>}
  </Card>
  <TrainingSuccessChart key={job.id} course={job.course} metrics={job.metrics ?? []} />
  <Card className="card"><h2>训练日志</h2><Code block className="training-logs" aria-label="训练日志">{job.logs || (busyJob(job)?'等待 Python 任务输出…':'此任务没有保存可用日志。')}</Code></Card>
 </section>;
}
