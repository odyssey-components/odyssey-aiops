import {Dialog} from '@odyssey/ui/primitives/dialog';
import {Field,FieldLabel,FieldDescription} from '@odyssey/ui/primitives/field';
import {Input} from '@odyssey/ui/primitives/input';
import {Button} from '@odyssey/ui/primitives/button';
import {Alert} from '@odyssey/ui/primitives/alert';
import {useState} from 'react';
import type {LessonId} from '../../../core';
import type {DatasetInfo,TrainingCommand} from '../../utilities/training-contract';
import {Choice} from '../common/Choice';
import {trainingTime} from '../../utilities/training-tasks';

interface FormProps {course:LessonId;datasets:DatasetInfo[];blocked:boolean;error:string;onClose:()=>void;onSubmit:(command:Extract<TrainingCommand,{action:'train'}>)=>void;}
export function TrainingTaskForm({course,datasets,blocked,error,onSubmit,onClose}:FormProps) {
 const [dataset,setDataset]=useState(''),[seed,setSeed]=useState('7');
 const [epochs,setEpochs]=useState(course==='fixed-map-v1'?'400':course==='local-map-v1'?'120':'160');
 const selected=datasets.find(entry=>entry.id===dataset) ?? datasets[0];
 return <form onSubmit={event=>{event.preventDefault();if(selected && !blocked) onSubmit({action:'train',course,dataset:selected.id,seed:Number(seed),epochs:Number(epochs)});}}>
   <Field><FieldLabel>地图组</FieldLabel><Choice id="training-dataset" label="预生成地图集" value={selected?.id ?? ''} disabled={!datasets.length || blocked}
    placeholder="请先在「地图组」中添加本课地图组" onChange={setDataset}
    options={datasets.map(entry=>({value:entry.id,label:`${entry.title} · ${entry.maps} 图 · ${trainingTime(entry.createdAt)}`}))} /></Field>
   {selected && <p id="dataset-summary">{selected.description} · 训练 {selected.maps} 张 · 验证 {selected.validation} 张 · 测试 {selected.test} 张</p>}
   {!datasets.length && <p>本课暂无地图组，请先到「地图组」添加。</p>}
   <div className="training-fields">
    <Field><FieldLabel>初始化种子</FieldLabel><Input id="training-seed" type="number" required min="0" max="2147483647" step="1" value={seed} disabled={blocked} onChange={event=>setSeed(event.currentTarget.value)} /></Field>
    <Field><FieldLabel>训练轮数</FieldLabel><Input id="training-epochs" type="number" required min="0" max="1000" step="1" value={epochs} disabled={blocked} onChange={event=>setEpochs(event.currentTarget.value)} /><FieldDescription>0 轮用于未训练基线</FieldDescription></Field>
   </div>
   {error && <Alert tone="danger" role="alert">{error}</Alert>}
   <div className="training-actions"><Button type="button" onClick={onClose}>关闭</Button><Button type="submit" id="start-training" variant="solid" disabled={blocked || !selected}>开始训练</Button></div>
  </form>;
}
export function NewTrainingDialog({open,onOpenChange,...props}:Omit<FormProps,'onClose'> & {open:boolean;onOpenChange:(open:boolean)=>void}) {
 return <Dialog open={open} onOpenChange={onOpenChange} className="dataset-dialog" bodyClassName="task-dialog-form"
  title="添加训练任务" description="选择本课地图组和训练参数。创建后可在详情页查看进度。"
  form={<TrainingTaskForm {...props} onClose={()=>onOpenChange(false)}/>} cancel={false}/>;
}
