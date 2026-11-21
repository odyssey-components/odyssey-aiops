import {Button} from '@odyssey/ui/primitives/button';
import {Card} from '@odyssey/ui/primitives/card';
import {Alert} from '@odyssey/ui/primitives/alert';
import {activeGo1,go1States,type Go1Simulation} from '../contract';
import {sceneKinds,sceneSuccess} from '../scenes';
import {go1Evaluation,go1SimulationEvaluation} from '../outcomes';

function ObstacleEvents({result}:{result:Record<string,unknown>}) {
 const fields=[['first_airborne','首次离地'],['first_obstacle_contact','首次触碰障碍']] as const;
 return <>{fields.map(([key,label])=>{
  const value=result[key];
  if(!value||typeof value!=='object')return null;
  const event=value as Record<string,unknown>;
  if(typeof event.time_seconds!=='number'||typeof event.x_m!=='number')return null;
  return <div key={key}><dt>{label}</dt><dd>{event.time_seconds.toFixed(2)} 秒 · 前进 {event.x_m.toFixed(2)} m{typeof event.robot_body==='string'&&event.robot_body? ` · ${event.robot_body}`:''}</dd></div>;
 })}</>;
}
export function SimulationDetail({job,pending,onCancel,onReplay,modelTitle,modelUnavailable=false}:{modelTitle?:string;modelUnavailable?:boolean;job:Go1Simulation;pending:boolean;onCancel:(id:string)=>void;onReplay:(id:string,caseId?:string)=>void}){
 const savedTitle=job.modelManifest.title;
 const displayModel=modelTitle?.trim() || (typeof savedTitle==='string' && savedTitle.trim()) || job.model;
 if(job.sceneSet||job.testCaseSet){
  const memberLabel=job.testCaseSet?'用例':'场景';
  const results=Array.isArray(job.result?.cases)?job.result.cases as {scene:string;outcome:string;steps:number}[]:[];
  const failures=new Map<string,number>();
  for(const result of results){
   if(result.outcome==='success'||result.outcome==='observed')continue;
   const reason=go1Evaluation(result);
   failures.set(reason,(failures.get(reason)??0)+1);
  }
  const total=job.result?.total,successes=job.result?.successes;
  const diagnostics=results.filter(result=>result.outcome==='observed').length;
  return <Card className="card dataset-detail">
   <h3>{job.title}</h3><p role="status">{go1States[job.state]}</p>
   <dl><dt>模型</dt><dd>{displayModel}</dd><dt>{job.testCaseSet?'测试用例集':'历史场景集'}</dt><dd>{job.testCaseSet?.title ?? job.sceneSet?.title}</dd></dl>
   <progress aria-label="集合执行进度" max={job.cases?.length ?? 1} value={job.progress}/>
   <p>已执行 {job.progress} / {job.cases?.length ?? 0} 个{memberLabel}</p>
   {job.error&&<Alert role="alert" tone="danger">{job.error}</Alert>}
   <h4>评估结果</h4><p>{go1SimulationEvaluation(job.state,job.result)}</p>
   {job.testCaseSet&&typeof total==='number'&&total>0&&typeof successes==='number'&&
    <p>通过率 {(successes/total*100).toFixed(1)}%（{successes}/{total}）</p>}
   {job.testCaseSet&&diagnostics>0&&<p>诊断用例 {diagnostics} 个</p>}
   {job.testCaseSet&&failures.size>0&&<p>失败原因：{[...failures].map(([reason,count])=>`${reason}（${count}）`).join('、')}</p>}
   <ul>{job.cases?.map((scene,index)=>{
    const result=results.find(item=>item.scene===scene.id);
    return <li key={scene.id}>{job.testCaseSet?`${index+1}. ${scene.title}`:scene.title}{result?` · ${go1Evaluation(result)} · ${result.steps} 步`:''}{job.testCaseSet&&result&&job.state==='succeeded'&&<Button size="sm" onClick={()=>onReplay(job.id,scene.id)}>查看指标</Button>}</li>;
   })}</ul>
   {activeGo1(job.state)&&job.state!=='finalizing'&&<Button disabled={pending||job.state==='cancelling'} onClick={()=>onCancel(job.id)}>取消仿真</Button>}
   {job.state==='succeeded'&&<Button onClick={()=>onReplay(job.id)}>查看仿真回放</Button>}
   <pre className="training-logs" aria-label="仿真日志">{job.logs || '等待仿真输出…'}</pre>
  </Card>;
 }
 return <Card className="card dataset-detail"><h3>{job.title}</h3>{modelUnavailable&&<p role="status">已移除模型 · 历史快照回放</p>}<p role="status">{job.state==='succeeded'?'执行完成':go1States[job.state]}</p><dl><dt>任务 ID</dt><dd>{job.id}</dd><dt>模型</dt><dd>{displayModel}</dd><dt>模型 SHA-256</dt><dd>{job.modelFiles['policy.npz']}</dd><dt>场景</dt><dd>{job.scene.title} · {sceneKinds[job.scene.kind]}</dd><dt>场景参数</dt><dd>{Object.entries(job.scene.parameters).map(([key,value])=>`${key}: ${value}`).join(' / ')}</dd><dt>成功条件</dt><dd>{sceneSuccess(job.scene)}</dd><dt>随机种子</dt><dd>{job.seed}</dd><dt>仿真步数上限</dt><dd>{job.steps}</dd></dl>
 <progress aria-label="仿真进度" max={job.steps} value={job.progress}/><p>已执行 {job.progress} / {job.steps} 步</p>
 {job.error && <Alert role="alert" tone="danger">{job.error}</Alert>}
 <h4>评估结果</h4>
 {!job.result&&<p>{go1SimulationEvaluation(job.state,job.result)}</p>}
 {job.result && <>{job.visionInjected&&<p>{job.result.vision_input_exhausted===true?'录制视觉输入已用完':'视觉输入来自 MCAP 录制'}</p>}<dl><dt>目标判定</dt><dd>{go1Evaluation(job.result)}</dd><dt>物理后端</dt><dd>{String(job.result.backend)}</dd><dt>实际时长</dt><dd>{String(job.result.duration_seconds)} 秒</dd><dt>最低机身高度</dt><dd>{String(job.result.minimum_height_m)} m</dd>{typeof job.result.final_yaw_deg==='number'&&<><dt>最终朝向</dt><dd>{job.result.final_yaw_deg.toFixed(1)}°</dd></>}<dt>最终位置</dt><dd>{Array.isArray(job.result.final_position_m)?job.result.final_position_m.join(' / '):'未记录'}</dd><ObstacleEvents result={job.result}/></dl></>}
 <div className="training-actions">{activeGo1(job.state) && job.state!=='finalizing' && <Button disabled={pending || job.state==='cancelling'} onClick={()=>onCancel(job.id)}>取消仿真</Button>}{job.state==='succeeded' && <Button onClick={()=>onReplay(job.id)}>查看仿真回放</Button>}</div>
 <pre className="training-logs" aria-label="仿真日志">{job.logs || '等待仿真输出…'}</pre></Card>;
}
