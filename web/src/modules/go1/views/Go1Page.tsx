import {tablePagination} from '../../../utilities/list-pagination';
import {Go1TestCaseSets} from './TestCaseSets';
import {Alert} from '@odyssey/ui/primitives/alert';
import {Button} from '@odyssey/ui/primitives/button';
import {MazeShell} from '../../../components/layout/MazeShell';
import type {MazeSearch} from '../../../utilities/search';
import {Go1Recordings} from './Recordings';
import {Go1Simulations} from './Simulations';
import {Go1Scenes} from './Scenes';
import {cancelGo1Simulation,createGo1Simulation,createGo1TestCaseSet,go1Command} from '../api';
import type {Go1Command,Go1SimulationCommand} from '../contract';
import {Go1Tasks} from './Tasks';
import {Go1Models} from './Models';
import {Go1Replay} from './Replay';
import {useGo1Snapshot} from '../useGo1Snapshot';
export function Go1Page({search,onChange}:{search:MazeSearch;onChange:(search:MazeSearch)=>void}) {
 const page=typeof search.listPage==='number'&&Number.isSafeInteger(search.listPage)&&search.listPage>0&&search.listPage<=1000000?search.listPage:1;
 const pageSize=typeof search.listSize==='number'&&[10,20,50,100].includes(search.listSize)?search.listSize:20;
 const selected=search.go1Tab==='models'?search.go1Model:search.go1Tab==='scenes'?search.go1Scene:!search.go1Tab?search.go1Task:search.go1Simulation;
 const {snapshot,error,refresh}=useGo1Snapshot(search.go1Tab,page,selected,search.go1Replay,pageSize);
 const simulations=snapshot?.simulationDetails ?? snapshot?.simulations;

 const publish=(patch:Partial<MazeSearch>)=>onChange({...search,...(Object.hasOwn(patch,'go1Tab')&&patch.go1Tab!==search.go1Tab?{listPage:undefined}:{}),...patch});
 const pagination=tablePagination(snapshot?.pagination,listPage=>publish({listPage}),listSize=>publish({listSize,listPage:undefined}));
 async function submit(command:Go1Command){const result=await go1Command(command) as {id?:string};await refresh();if(command.action==='train' && result.id)publish({go1Task:result.id});}

 async function createSimulation(command:Go1SimulationCommand){
  const result=await createGo1Simulation(command);
  await refresh();
  publish(command.action==='inject-vision'
   ?{go1Tab:'vision-injection',go1Simulation:result.id,go1Replay:undefined,go1ReplayCase:undefined}
   :{go1Simulation:result.id});
 }
 const legacyInjection=search.go1Tab==='simulation' && simulations?.some(task=>task.visionInjected &&
  (task.id===search.go1Simulation||task.id===search.go1Replay));
 const simulationSection=search.go1Tab==='vision-injection'||legacyInjection?'vision-injection':'simulation';
 return <MazeShell module="go1" onModule={module=>publish({module:module==='maze'?undefined:'go1'})} go1Section={search.go1Tab==='replay'?(search.go1Simulation?'simulation':'models'):legacyInjection?'vision-injection':search.go1Tab ?? 'training'} onGo1Section={section=>{publish({go1Tab:section==='training'?undefined:section,go1Task:undefined,go1Model:undefined,go1Simulation:undefined,go1Replay:undefined,go1ReplayCase:undefined,go1Scene:undefined});}} course={search.course} onCourse={()=>{}}>
 {error && <Alert role="alert" tone="danger">{error}<Button onClick={()=>void refresh().catch(()=>{})}>重试</Button></Alert>}
 {!search.go1Tab && <><Go1Tasks pagination={pagination} scene={search.go1Scene} onSceneUsed={()=>publish({go1Scene:undefined})} snapshot={snapshot} selected={search.go1Task} onSelect={go1Task=>publish({go1Task})} onSubmit={submit}/></>}
 {(search.go1Tab==='simulation'||search.go1Tab==='vision-injection') && <Go1Simulations pagination={pagination} key={simulationSection} visionInjection={simulationSection==='vision-injection'} replayId={search.go1Replay} replayCaseId={search.go1ReplayCase} onReplay={(go1Replay,go1ReplayCase)=>publish({go1Replay,go1ReplayCase,go1Tab:simulationSection})} initialModel={simulationSection==='simulation'?search.go1Model:undefined} onPresetUsed={()=>publish({go1Model:undefined})} snapshot={snapshot} selected={search.go1Simulation} onSelect={go1Simulation=>publish({go1Simulation,go1Tab:simulationSection})} onCreate={createSimulation} onCancel={async id=>{await cancelGo1Simulation(id);await refresh();}}/>}
 {search.go1Tab==='recording' && <Go1Recordings pagination={pagination} snapshot={snapshot} onCreate={async command=>{await createGo1Simulation(command);await refresh();}} onCancel={async id=>{await cancelGo1Simulation(id);await refresh();}}/>}
 {search.go1Tab==='models' && <><Go1Models pagination={pagination} snapshot={snapshot} selected={search.go1Model} onSelect={go1Model=>publish({go1Model})} onSimulate={go1Model=>publish({go1Model,go1Tab:'simulation',go1Simulation:undefined})}/></>}
 {search.go1Tab==='replay' && <><Go1Replay simulation={search.go1Simulation?{id:search.go1Simulation,title:simulations?.find(task=>task.id===search.go1Simulation)?.title ?? '仿真任务',scene:simulations?.find(task=>task.id===search.go1Simulation)?.scene}:undefined} onBack={()=>publish({go1Tab:search.go1Simulation?'simulation':'models',...(!search.go1Simulation?{go1Model:search.go1Model ?? snapshot?.models[0]?.run}:{})})} models={snapshot?.models ?? []} selected={search.go1Model} onSelect={go1Model=>publish({go1Model})}/></>}
 {search.go1Tab==='scenes'&&<Go1Scenes pagination={pagination} loading={!snapshot} selected={search.go1Scene} onSelect={go1Scene=>publish({go1Scene})} scenes={snapshot?.scenes ?? []} onTrain={go1Scene=>publish({go1Tab:undefined,go1Scene,go1Task:undefined,go1Model:undefined,go1Simulation:undefined})}/>}
 {search.go1Tab==='test-case-sets' && (snapshot?<Go1TestCaseSets pagination={pagination} cases={(snapshot.caseOptions ?? snapshot.simulations ?? []).filter(task=>task.state==='succeeded'&&task.recorded&&!task.sceneSet&&!task.testCaseSet&&(!task.recording||task.visionInjected))} sets={snapshot.testCaseSets ?? []} onCreate={async input=>{await createGo1TestCaseSet(input);await refresh();}}/>:<p role="status">正在读取测试用例集…</p>)}
 </MazeShell>;
}
