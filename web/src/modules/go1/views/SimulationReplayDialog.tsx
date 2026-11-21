import {useState} from 'react';
import {Dialog} from '@odyssey/ui/primitives/dialog';
import {Choice} from '../../../components/common/Choice';
import type {Go1Simulation} from '../contract';
import {Go1Replay} from './Replay';

function TaskReplay({task,onClose,selectedCase,onCase}:{task:Go1Simulation;onClose:()=>void;selectedCase?:string;onCase?:(id:string)=>void}) {
 const [selected,setSelected]=useState(task.cases?.[0]?.id);
 const scene=task.cases?.find(item=>item.id===(selectedCase ?? selected)) ?? task.scene;
 return <Dialog open onOpenChange={open=>{if(!open)onClose();}} className="go1-simulation-replay-dialog"
  title={`仿真回放 · ${task.title}`} description={`场景：${scene.title}`}
  content={<>
  {task.cases&&<Choice id="simulation-replay-case" label={task.testCaseSet?'回放用例':'回放场景'} value={scene.id} onChange={id=>{setSelected(id);onCase?.(id);}} options={task.cases.map((item,index)=>({value:item.id,label:task.testCaseSet?`${index+1}. ${item.title}`:item.title}))}/>}
  <Go1Replay key={scene.id} embedded simulation={{id:task.id,title:task.title,scene,visionInjected:task.visionInjected,originalId:task.visionInjected?(task.testCaseSet?scene.id:task.recording):undefined,caseId:task.cases?scene.id:undefined}} models={[]} onSelect={()=>{}}/>
  </>} cancel={{label:'关闭回放'}}/>;
}

export function SimulationReplayDialog({task,onClose,selectedCase,onCase}:{task:Go1Simulation|null;onClose:()=>void;selectedCase?:string;onCase?:(id:string)=>void}){
 return task?<TaskReplay key={task.id} task={task} onClose={onClose} selectedCase={selectedCase} onCase={onCase}/>:null;
}
