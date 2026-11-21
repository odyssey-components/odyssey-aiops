import type {TableProps} from '@odyssey/ui/primitives/table';
import {useEffect,useState} from 'react';
import {Button} from '@odyssey/ui/primitives/button';
import {Badge} from '@odyssey/ui/primitives/badge';
import {Drawer,DrawerBody,DrawerContent} from '@odyssey/ui/primitives/drawer';
import {Card} from '@odyssey/ui/primitives/card';
import {Alert} from '@odyssey/ui/primitives/alert';
import {Table,TableHeader,TableBody,TableRow,TableHead,TableCell} from '@odyssey/ui/primitives/table';
import {Choice} from '../../../components/common/Choice';
import {MetricSeriesChart} from '../../../components/training/MetricSeriesChart';
import {activeGo1,go1States,type Go1Snapshot,type Go1Command,type Go1Metric} from '../contract';
import {NewGo1Training} from './NewTraining';
const metrics={mean_reward:['平均奖励',''],loss:['PPO 损失',''],mean_height_m:['平均机身高度','m'],episode_ends:['采样回合结束数','次'],elapsed_seconds:['训练用时','秒']} as const;
export function Go1Tasks({pagination,snapshot,selected,onSelect,onSubmit,scene,onSceneUsed=()=>{}}:{pagination?:TableProps['pagination'];scene?:string;onSceneUsed?:()=>void;snapshot:Go1Snapshot|null;selected?:string;onSelect:(id?:string)=>void;onSubmit:(command:Go1Command)=>Promise<void>}) {
 const [open,setOpen]=useState(false),[metric,setMetric]=useState<keyof typeof metrics>('mean_reward'),[error,setError]=useState('');
 useEffect(()=>{if(scene)setOpen(true);},[scene]);
 const job=snapshot?.taskDetail?.id===selected?snapshot?.taskDetail:snapshot?.tasks.find(job=>job.id===selected),environment=snapshot?.environment;
 const deviceLabel=environment?.device?.toUpperCase();
 return <section className="training-workspace" aria-label="Go1 训练任务">
 <div className="dataset-heading">
  <div className="go1-training-heading">
   <h1>Go1 训练任务</h1>
   <Badge><span role="status">{!environment?'正在检测…':environment.ready?`可用 · ${deviceLabel ?? environment.devices.join('，')}`:'环境未就绪'}</span></Badge>
  </div>
  <Button variant="solid" disabled={!environment?.ready} onClick={()=>setOpen(true)}>添加训练任务</Button>
 </div>
 {environment?.errors.map((message,index)=><Alert key={index} tone="danger">{message}</Alert>)}
 {snapshot?.activity.active && <p>当前执行：{{maze:'Maze',go1:'Go1 训练','go1-simulation':'Go1 仿真'}[snapshot.activity.active.module] ?? snapshot.activity.active.module} · {snapshot.activity.active.id.slice(0,8)}；新任务按顺序排队。</p>}
 <Table pagination={pagination} aria-label="Go1 训练任务列表" containerClassName="dataset-list"><TableHeader><TableRow>{['名称','状态','创建时间','详情'].map(label=><TableHead scope="col" key={label}>{label}</TableHead>)}</TableRow></TableHeader><TableBody>{snapshot?.tasks.map(job=><TableRow key={job.id}><TableHead scope="row">{job.title}</TableHead><TableCell>{go1States[job.state]}</TableCell><TableCell>{new Date(job.startedAt).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false})}</TableCell><TableCell><Button size="sm" onClick={()=>onSelect(job.id)}>详情</Button></TableCell></TableRow>)}</TableBody></Table>{snapshot && !snapshot.tasks.length && <p>暂无 Go1 训练任务</p>}
 <Drawer open={Boolean(selected)} onOpenChange={value=>{if(!value){setError('');onSelect();}}}>
 {selected&&<DrawerContent className="go1-simulation-drawer" title="Go1 训练任务详情" closeLabel="关闭训练任务详情"><DrawerBody>
 {error && <Alert role="alert" tone="danger">{error}</Alert>}
 {selected && !snapshot && <p>正在读取任务…</p>}{selected && snapshot && !job && <p>未找到 Go1 训练任务</p>}
 {job && <>
 <Card className="card go1-training-summary">
  <div className="dataset-heading"><h3>{job.title}</h3><span role="status">{go1States[job.state]}</span></div>
  <p>设备：{job.device ?? '未记录'}</p>
  <p>基础模型：{job.base?.title ?? '从头训练'}</p>
  <p>场景：{job.scene?.title ?? '历史状态训练'}</p>
  {job.error && <Alert role="alert" tone="danger">{job.error}</Alert>}
  <details>
   <summary>训练参数与标识</summary>
   <dl>
    <div><dt>任务 ID</dt><dd>{job.id}</dd></div>
    {job.base && <div><dt>基础模型 ID</dt><dd>{job.base.id}</dd></div>}
    {job.scene && <div><dt>场景 ID</dt><dd>{job.scene.id}</dd></div>}
    {Object.entries(job.config)
     .filter(([key])=>(!job.base||key!=='checkpoint_interval')&&(!job.scene||!['target_speed_m_s','eval_steps','checkpoint_interval'].includes(key)))
     .map(([key,value])=><div key={key}><dt>{key}</dt><dd>{value}</dd></div>)}
   </dl>
  </details>
  {activeGo1(job.state) && job.state!=='finalizing' && <Button disabled={job.state==='cancelling'} onClick={()=>void onSubmit({action:'cancel',id:job.id}).catch(error=>setError(String(error)))}>取消任务</Button>}
  {job.run && <p>训练产物记录：{job.run}</p>}
 </Card>
 <div className="go1-training-metrics">
  <Choice id="go1-metric" label="训练指标" value={metric} onChange={key=>setMetric(key as keyof typeof metrics)} options={Object.entries(metrics).map(([value,[label]])=>({value,label}))}/>
  <MetricSeriesChart title={metrics[metric][0]} unit={metrics[metric][1]} points={(job.metrics ?? []).map((point:Go1Metric)=>({step:point.iteration,value:point[metric]}))}/>
 </div>
 <Card className="card go1-training-log">
  <h3>训练日志</h3>
  <pre className="training-logs" aria-label="Go1 训练日志">{job.logs || (activeGo1(job.state)?'等待训练输出…':'暂无训练日志')}</pre>
 </Card>
 </>}
 </DrawerBody></DrawerContent>}
 </Drawer>
 <NewGo1Training bases={snapshot?.baseModels ?? []} scenes={snapshot?.scenes ?? []} selectedScene={scene} open={open && environment?.ready===true} onOpen={value=>{setOpen(value);if(!value)onSceneUsed();}} onSubmit={onSubmit}/>
 </section>;
}
