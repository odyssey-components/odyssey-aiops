import {useEffect,useRef,useState} from 'react';
import {Dialog} from '@odyssey/ui/primitives/dialog';
import {Field,FieldLabel} from '@odyssey/ui/primitives/field';
import {Input} from '@odyssey/ui/primitives/input';
import {Choice} from '../../../components/common/Choice';
import {sceneKinds,type Go1Scene} from '../scenes';
import {go1Defaults,parseGo1Config,type Go1Base,type Go1Command} from '../contract';

const labels={iterations:'PPO 迭代次数',rollout_steps:'每次采样步数',seed:'训练种子',target_speed_m_s:'目标速度（m/s）',learning_rate:'学习率',eval_seed:'独立评估种子',eval_steps:'评估步数',checkpoint_interval:'检查点间隔（迭代）'};

export function NewGo1Training({open,onOpen,onSubmit,scenes,selectedScene,bases=[]}:{bases?:Go1Base[];scenes:Go1Scene[];selectedScene?:string;open:boolean;onOpen:(open:boolean)=>void;onSubmit:(command:Go1Command)=>Promise<void>}) {
 const formRef=useRef<HTMLFormElement>(null);
 const [base,setBase]=useState(bases[0]?.id ?? 'random');
 useEffect(()=>{if(open)setBase(bases[0]?.id ?? 'random');},[open]);
 const [scene,setScene]=useState(selectedScene ?? 'legacy');
 useEffect(()=>{setScene(selectedScene ?? 'legacy');},[selectedScene,open]);
 const [title,setTitle]=useState('Go1 PPO');
 const [config,setConfig]=useState(Object.fromEntries(Object.entries(go1Defaults).map(([key,value])=>[key,String(value)])));
 const [pending,setPending]=useState(false);
 const [error,setError]=useState('');

 async function submit() {
  if(pending || !formRef.current?.reportValidity()) throw Error('无法提交训练表单');
  setPending(true);
  setError('');
  try {
   if(Object.values(config).some(value=>!value.trim())) throw Error('请填写完整参数');
   const trainingConfig=parseGo1Config(Object.fromEntries(Object.entries(config).map(([key,value])=>[key,Number(value)])));
   await onSubmit({action:'train',title,...(base!=='random'?{base}:{}),...(scene!=='legacy'?{scene}:{}),config:trainingConfig});
  } catch(error) {
   setError(error instanceof Error?error.message:'无法启动');
   throw error;
  } finally {
   setPending(false);
  }
 }

 const form=<form ref={formRef} className="task-dialog-form training-fields" onSubmit={event=>{
  event.preventDefault();
  void submit().then(()=>onOpen(false),()=>{});
 }}>
  <Field className="task-dialog-field-wide">
   <FieldLabel>任务名称</FieldLabel>
   <Input required maxLength={80} value={title} disabled={pending} onChange={event=>setTitle(event.currentTarget.value)}/>
  </Field>
  <Field>
   <FieldLabel>基础模型</FieldLabel>
   <Choice id="go1-base-model" label="基础模型" value={base} disabled={pending} onChange={setBase} options={[{value:'random',label:'从头训练'},...bases.map(item=>({value:item.id,label:item.title}))]}/>
  </Field>
  <Field>
   <FieldLabel>场景</FieldLabel>
   <Choice id="go1-training-scene" label="训练场景" value={scene} disabled={pending} onChange={setScene} options={[{value:'legacy',label:'原状态训练（无场景）'},...scenes.filter(item=>item.kind!=='flat').map(item=>({value:item.id,label:`${item.title} · ${sceneKinds[item.kind]}`}))]}/>
  </Field>
  {Object.entries(labels).filter(([key])=>(base==='random'||key!=='checkpoint_interval')&&(scene==='legacy'||!['target_speed_m_s','eval_steps','checkpoint_interval'].includes(key))).map(([key,label])=><Field key={key}>
   <FieldLabel>{label}</FieldLabel>
   <Input required type="number" step={key==='learning_rate'||key==='target_speed_m_s'?'any':'1'} value={config[key]} disabled={pending} onChange={event=>{
    const value=event.currentTarget.value;
    setConfig(previous=>({...previous,[key]:value}));
   }}/>
  </Field>)}
  <button type="submit" hidden disabled={pending}/>
 </form>;

 return <Dialog className="go1-task-dialog" open={open}
   onOpenChange={onOpen}
   title="添加 Go1 训练任务"
   description="PPO"
   form={form}
   pending={pending}
   error={error}
   cancel={{label:'取消'}}
   confirm={{label:'提交训练',onConfirm:submit}}/>;
}
