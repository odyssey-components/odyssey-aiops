import type {TableProps} from '@odyssey/ui/primitives/table';
import {useState} from 'react';
import {Alert} from '@odyssey/ui/primitives/alert';
import {Button} from '@odyssey/ui/primitives/button';
import {Dialog} from '@odyssey/ui/primitives/dialog';
import {Input} from '@odyssey/ui/primitives/input';
import {Field,FieldLabel} from '@odyssey/ui/primitives/field';
import {Table,TableHeader,TableBody,TableRow,TableHead,TableCell} from '@odyssey/ui/primitives/table';
import type {Go1Simulation,Go1TestCaseSet} from '../contract';
import {RecordingFrames} from './RecordingFrames';

export function Go1TestCaseSets({pagination,cases,sets,onCreate}:{pagination?:TableProps['pagination'];
 cases:Go1Simulation[];
 sets:Go1TestCaseSet[];
 onCreate:(input:{title:string;cases:string[]})=>Promise<void>;
}) {
 const [frameCase,setFrameCase]=useState<Go1Simulation>();
 const [open,setOpen]=useState(false);
 const [title,setTitle]=useState('');
 const [selected,setSelected]=useState<string[]>([]);
 const [pending,setPending]=useState(false);
 const [error,setError]=useState('');
 async function save(){
  if(pending||!title.trim()||!selected.length)return;
  setPending(true);setError('');
  try{await onCreate({title:title.trim(),cases:selected});setOpen(false);}
  catch(error){setError(error instanceof Error?error.message:String(error));}
  finally{setPending(false);}
 }
 return <section className="training-workspace" aria-label="Go1 测试用例集">
  <div className="dataset-heading"><h1>测试用例集</h1><Button disabled={!cases.length} onClick={()=>{setTitle('');setSelected([]);setError('');setOpen(true);}}>添加测试用例集</Button></div>
  {sets.length?<Table pagination={pagination} aria-label="Go1 测试用例集列表" containerClassName="dataset-list">
   <TableHeader><TableRow><TableHead>名称</TableHead><TableHead>测试用例</TableHead></TableRow></TableHeader>
   <TableBody>{sets.map(set=><TableRow key={set.id}>
    <TableHead scope="row">{set.title}</TableHead>
    <TableCell>{set.cases.map(id=>{
     const item=cases.find(entry=>entry.id===id);
     return item?<Button key={id} size="sm" onClick={()=>setFrameCase(item)}>{item.title}</Button>:<span key={id}>{id}</span>;
    })}</TableCell>
   </TableRow>)}</TableBody>
  </Table>:<p>暂无测试用例集。</p>}
  <Dialog open={open} onOpenChange={value=>{if(!pending)setOpen(value);}} className="go1-task-dialog" bodyClassName="task-dialog-form"
   title="添加测试用例集" description="选择 1–16 个已完成的测试用例。"
   form={
    <form id="go1-case-set-form" onSubmit={event=>{event.preventDefault();void save();}}>
     <Field><FieldLabel>测试用例集名称</FieldLabel><Input required maxLength={80} value={title} disabled={pending} onChange={event=>setTitle(event.currentTarget.value)}/></Field>
     <fieldset disabled={pending}><legend>测试用例（已选 {selected.length}）</legend>
      {cases.map(scene=><div key={scene.id}><label><input type="checkbox" checked={selected.includes(scene.id)}
       disabled={selected.length>=16&&!selected.includes(scene.id)}
       onChange={event=>{const checked=event.currentTarget.checked;setSelected(items=>checked?[...items,scene.id]:items.filter(id=>id!==scene.id));}}/> {scene.title}</label></div>)}
     </fieldset>
     {error&&<Alert role="alert" tone="danger">{error}</Alert>}
    </form>
   }
   footer={<>
    <Button type="button" disabled={pending} onClick={()=>setOpen(false)}>关闭</Button>
    <Button type="submit" form="go1-case-set-form" variant="solid" disabled={pending||!title.trim()||!selected.length}>保存测试用例集</Button>
   </>}/>
  {frameCase&&<RecordingFrames key={frameCase.id} task={frameCase} onClose={()=>setFrameCase(undefined)}/>}
 </section>;
}
