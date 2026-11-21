import type {TableProps} from '@odyssey/ui/primitives/table';
import {Drawer,DrawerBody,DrawerContent} from '@odyssey/ui/primitives/drawer';
import {Alert} from '@odyssey/ui/primitives/alert';
import {SimulationDetail} from './SimulationDetail';
import {SimulationReplayDialog} from './SimulationReplayDialog';
import {RunMetrics} from './RunMetrics';
import {useState} from 'react';
import {CreateSimulationDialog} from './CreateSimulationDialog';
import {RecordingFrames} from './RecordingFrames';
import {Button} from '@odyssey/ui/primitives/button';
import {Table,TableHeader,TableBody,TableRow,TableHead,TableCell} from '@odyssey/ui/primitives/table';
import {activeGo1,go1States,type Go1Snapshot,type Go1SimulationCommand} from '../contract';

export function Go1Recordings({pagination,snapshot,onCreate,onCancel}:{pagination?:TableProps['pagination'];
 snapshot:Go1Snapshot|null;
 onCreate:(command:Go1SimulationCommand)=>void|Promise<void>;
 onCancel:(id:string)=>Promise<void>;
}) {
 const tasks=(snapshot?.simulations ?? []).filter(task=>!task.visionInjected && !task.recording);
 const [creating,setCreating]=useState(false);
 const [frames,setFrames]=useState<string>();
 const [detail,setDetail]=useState<string>();
 const [replay,setReplay]=useState<{id:string;caseId?:string}>();
 const [pending,setPending]=useState(false);
 const [error,setError]=useState('');
 const detailTask=tasks.find(item=>item.id===detail);
 const replayTask=tasks.find(item=>item.id===replay?.id) ?? null;
 async function cancel(id:string){
  setPending(true);setError('');
  try{await onCancel(id);}
  catch(error){setError(error instanceof Error?error.message:String(error));}
  finally{setPending(false);}
 }
 const task=tasks.find(item=>item.id===frames);
 const canCreate=Boolean(snapshot?.scenes?.length&&(snapshot.models.length||snapshot.baseModels?.length));
 return <section className="training-workspace" aria-label="Go1 测试用例">
  <div className="dataset-heading"><h1>测试用例</h1><Button variant="solid" disabled={!canCreate} onClick={()=>setCreating(true)}>创建测试用例</Button></div>
  {!snapshot?<p role="status">正在读取测试用例…</p>:<>
   {tasks.length>0&&<RunMetrics jobs={tasks} summary={snapshot?.runMetrics}/>}
   {!canCreate&&<p>请先准备模型和仿真场景。</p>}
   {tasks.length===0?<p>暂无测试用例。</p>:<Table pagination={pagination} aria-label="Go1 测试用例列表" containerClassName="dataset-list">
    <TableHeader><TableRow>{['仿真任务','场景','录制状态','操作'].map(label=><TableHead scope="col" key={label}>{label}</TableHead>)}</TableRow></TableHeader>
    <TableBody>{tasks.map(task=><TableRow key={task.id}>
     <TableHead scope="row">{task.title}</TableHead><TableCell>{task.scene.title}</TableCell>
     <TableCell>{(task.sceneSet||task.testCaseSet)&&task.state==='succeeded'?`${task.cases?.length ?? 0} 份 MCAP 已就绪`:task.recorded&&task.state==='succeeded'?'MCAP 已就绪':task.state==='succeeded'?'无 MCAP 录制':activeGo1(task.state)?'录制中 · '+go1States[task.state]:go1States[task.state]}</TableCell>
     <TableCell><div className="training-actions"><Button size="sm" onClick={()=>{setError('');setDetail(task.id);}}>仿真详情</Button>{task.state==='succeeded'&&(task.recorded||task.sceneSet||task.testCaseSet)&&<Button size="sm" onClick={()=>setFrames(task.id)}>查看录制帧</Button>}</div></TableCell>
    </TableRow>)}</TableBody>
   </Table>}
  </>}
  <Drawer open={Boolean(detailTask)} onOpenChange={open=>{if(!open)setDetail(undefined);}}>
   {detailTask&&<DrawerContent className="go1-simulation-drawer" title="仿真任务详情" closeLabel="关闭仿真详情"><DrawerBody>
    {error&&<Alert role="alert" tone="danger">{error}</Alert>}
    <SimulationDetail job={detailTask} modelTitle={snapshot?.models.find(model=>model.run===detailTask.model)?.title ?? snapshot?.baseModels?.find(model=>model.id===detailTask.model)?.title} pending={pending} onCancel={id=>void cancel(id)}
     onReplay={(id,caseId)=>setReplay({id,caseId})}/>
   </DrawerBody></DrawerContent>}
  </Drawer>
  <SimulationReplayDialog task={replayTask} selectedCase={replay?.caseId}
   onCase={caseId=>{if(replay)setReplay({...replay,caseId});}} onClose={()=>setReplay(undefined)}/>
  {creating&&<CreateSimulationDialog snapshot={snapshot} open={creating} setOpen={setCreating} onCreate={async command=>{await onCreate(command);}} testCase/>}
  {task&&<RecordingFrames key={task.id} task={task} onClose={()=>setFrames(undefined)}/>}
 </section>;
}
