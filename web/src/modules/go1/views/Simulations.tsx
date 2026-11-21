import type {TableProps} from '@odyssey/ui/primitives/table';
import {ReplayPairMetrics,RunMetrics} from './RunMetrics';
import {useEffect,useState} from 'react';
import {Button} from '@odyssey/ui/primitives/button';
import {Drawer,DrawerContent,DrawerBody} from '@odyssey/ui/primitives/drawer';
import {CreateSimulationDialog} from './CreateSimulationDialog';
import {CreateVisionTestDialog} from './CreateVisionTestDialog';
import {SimulationDetail} from './SimulationDetail';
import {SimulationReplayDialog} from './SimulationReplayDialog';
import {Alert} from '@odyssey/ui/primitives/alert';
import {Table,TableHeader,TableBody,TableRow,TableHead,TableCell} from '@odyssey/ui/primitives/table';
import {go1States,type Go1Snapshot,type Go1SimulationCommand} from '../contract';
import {go1SimulationEvaluation} from '../outcomes';
export function Go1Simulations({pagination,visionInjection=false,snapshot,selected,onSelect,onCreate,onCancel,initialModel,onPresetUsed,replayId,replayCaseId,onReplay}:{pagination?:TableProps['pagination'];visionInjection?:boolean;replayId?:string;replayCaseId?:string;onReplay?:(id?:string,caseId?:string)=>void;initialModel?:string;onPresetUsed?:()=>void;snapshot:Go1Snapshot|null;selected?:string;onSelect:(id?:string)=>void;onCreate:(command:Go1SimulationCommand)=>Promise<void>;onCancel:(id:string)=>Promise<void>;}) {
 const label=visionInjection?'回灌测试':'仿真任务';
 const [creatingTest,setCreatingTest]=useState(false);
 const [open,setOpen]=useState(false),[title,setTitle]=useState('Go1 仿真'),[model,setModel]=useState('');
 const [pending,setPending]=useState(false),[error,setError]=useState('');
 const models=[...(snapshot?.models ?? []),...(snapshot?.baseModels ?? []).filter(base=>!snapshot?.models.some(model=>model.run===base.id)).map(base=>({run:base.id,title:base.title}))],scenes=snapshot?.scenes ?? [],tasks=(snapshot?.simulations ?? []).filter(task=>visionInjection ? task.visionInjected === true : !task.visionInjected && !task.recording);
 const targets=scenes.map(scene=>({value:scene.id,label:scene.title}));
 const chosenModel=models[0]?.run ?? '',chosenScene=targets[0]?.value ?? '';
 const preset=models.find(entry=>entry.run===initialModel)?.run;
 useEffect(()=>{if(preset){setTitle('Go1 仿真');setModel(preset);setError('');setOpen(true);onPresetUsed?.();}},[preset,onPresetUsed]);
 const details=snapshot?.simulationDetails ?? tasks;
 const replay=details.find(task=>task.id===replayId && task.state==='succeeded') ?? null;
 function showReplay(id:string,caseId?:string){onReplay?.(id,caseId);}
 const job=details.find(job=>job.id===selected);
 async function cancel(id:string){setPending(true);setError('');try{await onCancel(id);}catch(error){setError(String(error));}finally{setPending(false);}}
 return <section className="training-workspace" aria-label={`Go1 ${label}`}>
 <div className="dataset-heading"><h1>{label}</h1>{visionInjection?<Button variant="solid" disabled={!snapshot} onClick={()=>setCreatingTest(true)}>新建回灌测试</Button>:<Button variant="solid" disabled={!chosenModel || !chosenScene} onClick={()=>{setTitle('Go1 仿真');setModel(chosenModel);setError('');setOpen(true);}}>添加仿真任务</Button>}</div>
 {error && !open && <Alert role="alert" tone="danger">{error}</Alert>}
 <>{!visionInjection && snapshot && (!models.length || !targets.length) && <p>请先准备已有模型和场景，再创建仿真任务。</p>}
 {visionInjection&&!selected&&tasks.length>0&&<RunMetrics jobs={tasks} summary={snapshot?.runMetrics}/>}
 <Table pagination={pagination} aria-label={`Go1 ${label}列表`} containerClassName="dataset-list"><TableHeader><TableRow>{['名称','模型','场景','执行状态','评估结果','创建时间','操作'].map(label=><TableHead key={label} scope="col">{label}</TableHead>)}</TableRow></TableHeader>
 <TableBody>{tasks.map(task=><TableRow key={task.id}><TableHead scope="row">{task.title}</TableHead><TableCell>{models.find(model=>model.run===task.model)?.title ?? `${task.model}（已移除模型 · 历史快照）`}</TableCell><TableCell>{task.testCaseSet?.title ?? task.sceneSet?.title ?? task.scene.title}</TableCell><TableCell>{task.state==='succeeded'?'执行完成':go1States[task.state]}</TableCell><TableCell>{go1SimulationEvaluation(task.state,task.result)}</TableCell><TableCell>{new Date(task.startedAt).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false})}</TableCell><TableCell><div className="training-actions"><Button size="sm" onClick={()=>onSelect(task.id)}>详情</Button><Button size="sm" disabled={task.state!=='succeeded'} onClick={()=>showReplay(task.id)}>回放</Button></div></TableCell></TableRow>)}</TableBody></Table>
 {snapshot && !tasks.length && <p>暂无{label}</p>}</>
 {!snapshot && <p role="status">正在读取仿真任务…</p>}
 <Drawer open={Boolean(selected)} onOpenChange={value=>{if(!value)onSelect();}}>
 {selected&&<DrawerContent className="go1-simulation-drawer" title={`${label}详情`} closeLabel="关闭仿真详情"><DrawerBody>
 {!snapshot?<p role="status">正在读取仿真任务…</p>:!job?<p>未找到仿真任务</p>:<SimulationDetail job={job} modelTitle={models.find(model=>model.run===job.model)?.title} modelUnavailable={!models.some(model=>model.run===job.model)} pending={pending} onCancel={id=>void cancel(id)} onReplay={showReplay}/>}
 {visionInjection&&job&&<ReplayPairMetrics job={job} history={snapshot?.caseOptions ?? snapshot?.simulations ?? []}/>}
 </DrawerBody></DrawerContent>}
 </Drawer>
 {snapshot && replayId && !replay && <Alert role="alert" tone="danger">该仿真没有完整回放 <Button onClick={()=>onReplay?.()}>关闭</Button></Alert>}
 <SimulationReplayDialog task={replay} selectedCase={replayCaseId} onCase={caseId=>onReplay?.(replayId,caseId)} onClose={()=>onReplay?.()}/>
 {visionInjection&&creatingTest&&snapshot&&<CreateVisionTestDialog snapshot={snapshot} onCreate={onCreate} onClose={()=>setCreatingTest(false)}/>}
 {!visionInjection&&open&&<CreateSimulationDialog snapshot={snapshot} open={open} setOpen={setOpen} onCreate={onCreate} initialModel={model} initialTitle={title}/>}
 </section>;
}
