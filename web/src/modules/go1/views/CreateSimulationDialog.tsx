import {useRef,useState} from 'react';
import {Dialog} from '@odyssey/ui/primitives/dialog';
import {Field,FieldLabel} from '@odyssey/ui/primitives/field';
import {Input} from '@odyssey/ui/primitives/input';
import {Choice} from '../../../components/common/Choice';
import {sceneSuccess} from '../scenes';
import type {Go1Snapshot,Go1SimulationCommand} from '../contract';

export function CreateSimulationDialog({snapshot,open,setOpen,onCreate,initialModel,initialTitle,testCase=false}:{
 snapshot:Go1Snapshot|null;open:boolean;setOpen:(open:boolean)=>void;
 onCreate:(command:Go1SimulationCommand)=>Promise<void>;initialModel?:string;
 initialTitle?:string;testCase?:boolean;
}) {
 const formRef=useRef<HTMLFormElement>(null);
 const [title,setTitle]=useState(initialTitle ?? (testCase?'Go1 测试用例':'Go1 仿真'));
 const [model,setModel]=useState(initialModel ?? ''),[scene,setScene]=useState('');
 const [seed,setSeed]=useState('7'),[steps,setSteps]=useState('500'),[pending,setPending]=useState(false),[error,setError]=useState('');
 const models=[...(snapshot?.models ?? []),...(snapshot?.baseModels ?? []).filter(base=>!snapshot?.models.some(model=>model.run===base.id)).map(base=>({run:base.id,title:base.title}))],scenes=snapshot?.scenes ?? [];
 const targets=scenes.map(scene=>({value:scene.id,label:scene.title}));
 const chosenModel=models.find(entry=>entry.run===model)?.run ?? models[0]?.run ?? '',chosenScene=targets.find(entry=>entry.value===scene)?.value ?? targets[0]?.value ?? '';
 const chosenScenario=scenes.find(item=>item.id===chosenScene);
 async function create() {
  if(pending || !chosenModel || !chosenScene || !formRef.current?.reportValidity()) throw Error('无法提交仿真表单');
  setPending(true);
  setError('');
  try {
   await onCreate({action:'simulate',title,model:chosenModel,scene:chosenScene,seed:Number(seed),steps:Number(steps)});
  } catch(error) {
   setError(error instanceof Error?error.message:'创建失败');
   throw error;
  } finally {
   setPending(false);
  }
 }
 return <Dialog className="go1-task-dialog" open={open}
   onOpenChange={setOpen}
   title={testCase?'创建测试用例':'添加仿真任务'}
   description="选择模型和场景，保存评估结果与回放。"
   form={<form ref={formRef} className="task-dialog-form training-fields" onSubmit={event=>{
   event.preventDefault();
   void create().then(()=>setOpen(false),()=>{});
  }}>
   <Field className="task-dialog-field-wide"><FieldLabel>任务名称</FieldLabel><Input required maxLength={80} value={title} disabled={pending} onChange={event=>setTitle(event.currentTarget.value)}/></Field>
   <Field>
    <FieldLabel htmlFor="simulation-model">仿真模型</FieldLabel>
    <Choice id="simulation-model" label="仿真模型" value={chosenModel} disabled={pending} onChange={setModel} options={models.map(model=>({value:model.run,label:model.title}))}/>
   </Field>
    <Field>
     <FieldLabel htmlFor="simulation-scene">仿真场景</FieldLabel>
     <Choice id="simulation-scene" label="仿真场景" value={chosenScene} disabled={pending} onChange={setScene} options={targets}/>
    </Field>
    {chosenScenario&&<p className="task-dialog-field-wide go1-simulation-criterion">{sceneSuccess(chosenScenario)}</p>}
    <Field><FieldLabel>随机种子</FieldLabel><Input type="number" required min="0" max="2147483647" step="1" value={seed} disabled={pending} onChange={event=>setSeed(event.currentTarget.value)}/></Field>
    <Field><FieldLabel>仿真步数</FieldLabel><Input type="number" required min="50" max="500" step="1" value={steps} disabled={pending} onChange={event=>setSteps(event.currentTarget.value)}/></Field>
   <button type="submit" hidden disabled={pending}/>
  </form>}
   pending={pending}
   error={error}
   cancel={{label:'关闭'}}
   confirm={{label:testCase?'创建测试用例':'创建仿真任务',disabled:!chosenModel || !chosenScene,onConfirm:create}}/>;
}
