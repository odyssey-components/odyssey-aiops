import type {Go1RunMetrics,Go1Simulation} from '../contract';

export function RunMetrics({jobs,summary}:{jobs:Go1Simulation[];summary?:Go1RunMetrics}) {
 const completed=jobs.filter(job=>job.state==='succeeded'||job.state==='failed');
 const failed=completed.filter(job=>job.state==='failed').length;
 const times=completed.flatMap(job=>{
  const elapsed=Date.parse(job.finishedAt ?? '')-Date.parse(job.startedAt);
  return Number.isFinite(elapsed)&&elapsed>=0?[elapsed/1000]:[];
 }).sort((a,b)=>a-b);
 const p95=summary?summary.p95??undefined:times[Math.ceil(times.length*.95)-1];
 const count=summary?.completed ?? completed.length,failures=summary?.failed ?? failed;
 const cancelled=summary?.cancelled ?? jobs.filter(job=>job.state==='cancelled').length;
 return <dl aria-label="任务指标" className="go1-scene-facts">
  <div><dt>执行异常率</dt><dd>{count?`${(100*failures/count).toFixed(1)}%（${failures}/${count}）`:'暂无数据'}</dd></div>
  <div><dt>提交到结束 P95（含排队）</dt><dd>{p95===undefined?'暂无数据':`${p95.toFixed(2)} 秒（${summary?.sampleCount ?? times.length} 项）`}</dd></div>
  <div><dt>已取消</dt><dd>{cancelled}</dd></div>
 </dl>;
}

export function ReplayPairMetrics({job,history}:{job:Go1Simulation;history:Go1Simulation[]}) {
 const ids=job.testCaseSet?.cases ?? (job.recording?[job.recording]:[]);
 if(!ids.length||job.state!=='succeeded')return null;
 const cases=Array.isArray(job.result?.cases)?job.result.cases:[job.result];
 let paired=0,comparable=0,newFailures=0;
 ids.forEach((id,index)=>{
  const original=history.find(item=>item.id===id&&item.state==='succeeded');
  const replay=cases[index];
  const scene=job.cases?.[index]??job.scene;
  if(!original||original.scene.id!==scene.id||!replay||(job.testCaseSet&&replay.scene!==scene.id))return;
  paired++;
  const old=original.result?.passed;
  const next=replay.passed ?? (replay.outcome==='success'?true:
   ['fallen','fall','collision','bypass','timeout'].includes(replay.outcome)?false:undefined);
  if(!job.visionInjected||scene.kind==='flat'||typeof old!=='boolean'||typeof next!=='boolean')return;
  comparable++;
  if(old&&!next)newFailures++;
 });
 return <dl aria-label="回灌配对指标" className="go1-scene-facts">
  <div><dt>原录制配对覆盖率</dt><dd>{(100*paired/ids.length).toFixed(1)}%（{paired}/{ids.length}）</dd></div>
  <div><dt>新增未通过</dt><dd>{comparable?`${newFailures} / ${comparable} 个可判定用例`:'无可比闭环判定'}</dd></div>
 </dl>;
}
