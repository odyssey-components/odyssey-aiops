import {useRef,useState} from 'react';
import {Dialog} from '@odyssey/ui/primitives/dialog';
import {Field,FieldLabel} from '@odyssey/ui/primitives/field';
import {Input} from '@odyssey/ui/primitives/input';
import {Choice} from '../../../components/common/Choice';
import type {Go1SimulationCommand,Go1Snapshot} from '../contract';
import {visionTestModel,visionTestModelName} from '../vision-test-policy';

export function CreateVisionTestDialog({snapshot,onClose,onCreate,initialRecording}:{
 initialRecording?:string;
 snapshot:Go1Snapshot;
 onClose:()=>void;
 onCreate:(command:Go1SimulationCommand)=>Promise<void>;
}) {
 const recordings=(snapshot.caseOptions ?? snapshot.simulations ?? []).filter(task=>task.recorded&&task.state==='succeeded'&&!task.sceneSet&&!task.testCaseSet&&(!task.recording||task.visionInjected)&&
  !['go1-robot-parkour/v1','go1-walk-these-ways/v1'].includes(String(task.modelManifest?.format)));
 const sets=(snapshot.testCaseSets ?? []).filter(set=>set.cases.length&&set.cases.every(id=>recordings.some(task=>task.id===id)));
 const [target,setTarget]=useState('case');
 const [setId,setSetId]=useState('');
 const selectedSet=sets.find(set=>set.id===setId) ?? sets[0];
 const batch=target==='set';
 const model=visionTestModel(snapshot);
 const formRef=useRef<HTMLFormElement>(null);
 const [selection,setSelection]=useState(initialRecording ?? '');
 const [title,setTitle]=useState('Go1 视觉回灌测试');
 const [pending,setPending]=useState(false);
 const [error,setError]=useState('');
 const recording=recordings.find(task=>task.id===selection) ?? recordings[0];
 const canCreate=Boolean((batch?selectedSet:recording)&&model&&title.trim());

 async function create(){
  if(!canCreate||!model||pending||!formRef.current?.reportValidity())throw Error('无法提交回灌表单');
  setPending(true);setError('');
  try{
   const input={action:'inject-vision' as const,title:title.trim(),model:model.run};
   if(batch&&selectedSet)await onCreate({...input,testCaseSet:selectedSet.id});
   else if(recording)await onCreate({...input,recording:recording.id});
  }catch(error){setError(error instanceof Error?error.message:String(error));throw error;}
  finally{setPending(false);}
 }

 return <Dialog open onOpenChange={open=>{if(!open&&!pending)onClose();}} className="go1-task-dialog"
  title="新建回灌测试" description="使用录制深度帧驱动 Go1 500 万步视觉策略，比较原轨迹与本次仿真。"
  form={<form ref={formRef} className="task-dialog-form training-fields" onSubmit={event=>{
   event.preventDefault();void create().then(onClose,()=>{});
  }}>
   <Field className="task-dialog-field-wide"><FieldLabel>任务名称</FieldLabel><Input required maxLength={80} value={title} disabled={pending} onChange={event=>setTitle(event.currentTarget.value)}/></Field>
   <Field className="task-dialog-field-wide">
    <FieldLabel htmlFor="vision-test-scope">测试范围</FieldLabel>
    <Choice id="vision-test-scope" label="测试范围" value={target} disabled={pending} onChange={setTarget} options={[{label:'测试用例',value:'case'},{label:'测试用例集',value:'set'}]}/>
   </Field>
   <Field className="task-dialog-field-wide">
    <FieldLabel htmlFor="vision-recording">{batch?'测试用例集':'测试用例'}</FieldLabel>
    {batch?<Choice id="vision-recording" label="测试用例集" value={selectedSet?.id ?? ''} disabled={pending||!sets.length} onChange={setSetId} placeholder="暂无兼容用例集" options={sets.map(set=>({value:set.id,label:set.title}))}/>:
     <Choice id="vision-recording" label="测试用例" value={recording?.id ?? ''} disabled={pending||!recordings.length} onChange={setSelection} placeholder="暂无兼容录制" options={recordings.map(task=>({value:task.id,label:task.title}))}/>}
   </Field>
   <Field className="task-dialog-field-wide">
    <FieldLabel>待测模型</FieldLabel><p>{model?.title ?? visionTestModelName}</p>
   </Field>
   {!recordings.length&&<p className="task-dialog-field-wide">请先用 Go1 力矩模型运行一次仿真，生成完整录制。</p>}
   {!model&&<p className="task-dialog-field-wide">500 万步视觉模型未登记，暂时无法创建回灌。</p>}
   <button type="submit" hidden disabled={pending||!canCreate}/>
  </form>}
  pending={pending} error={error} cancel={{label:'关闭'}} confirm={{label:'新建回灌测试',disabled:!canCreate,onConfirm:create}}/>;
}
