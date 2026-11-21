import {tablePagination} from '../utilities/list-pagination';
import {useEffect,useRef,useState} from 'react';
import {Alert} from '@odyssey/ui/primitives/alert';
import {Field,FieldLabel} from '@odyssey/ui/primitives/field';
import {EmptyState,EmptyStateTitle} from '@odyssey/ui/primitives/empty-state';
import {Table,TableHeader,TableBody,TableRow,TableHead,TableCell} from '@odyssey/ui/primitives/table';
import {Input} from '@odyssey/ui/primitives/input';
import {Textarea} from '@odyssey/ui/primitives/textarea';
import {Button} from '@odyssey/ui/primitives/button';
import {Dialog} from '@odyssey/ui/primitives/dialog';
import {DatasetDetail} from './DatasetDetail';
import type {LessonId} from '../../core';
import {getTrainingSnapshot,submitTrainingCommand} from '../apis/training';
import {busyJob,type TrainingSnapshot} from '../utilities/training-contract';

export function DatasetView({course,dataset,onDataset}:{course:LessonId;dataset?:string;onDataset:(id:string|undefined)=>void}) {
 const [pageSize,setPageSize]=useState(20);
 const [page,setPage]=useState(1);
  const fixed=course==='fixed-map-v1';
  const [snapshot,setSnapshot]=useState<TrainingSnapshot|null>(null);
  const [open,setOpen]=useState(false),[pending,setPending]=useState(false);
  const [title,setTitle]=useState(''),[description,setDescription]=useState('');
  const [seed,setSeed]=useState('7'),[maps,setMaps]=useState(course==='local-map-v1'?'128':'256');
  const [error,setError]=useState(''),[pollError,setPollError]=useState('');
  const submitted=useRef('');
  useEffect(()=>{
    let disposed=false;let timer:ReturnType<typeof setTimeout>;
    const controller=new AbortController();
    async function poll() {
      try {
        const value=await getTrainingSnapshot(course,controller.signal,undefined,{kind:"datasets",page,pageSize,dataset});
        if(disposed) return;
        setSnapshot(value);setPollError('');
        if(submitted.current && value.job?.id===submitted.current && !busyJob(value.job)) {
          submitted.current='';
          if(value.job.state==='succeeded') {setOpen(false);setTitle('');setDescription('');}
          else setError(value.job.error ?? (value.job.state==='cancelled'?'地图组生成已取消':'地图组生成失败，请重试'));
        }
      } catch(error) {if(!disposed) setPollError(error instanceof Error?error.message:'无法读取地图组');}
      finally {if(!disposed) timer=setTimeout(()=>void poll(),1500);}
    }
    void poll();return ()=>{disposed=true;controller.abort();clearTimeout(timer);};
  },[course,page,pageSize,dataset]);
  const blocked=pending || busyJob(snapshot?.active) || !snapshot || !!pollError;
  const job=snapshot?.job;
  async function create() {
    setPending(true);setError('');
    try {
      const result=await submitTrainingCommand({action:'prepare',course,title,description,
        seed:Number(seed),maps:fixed?16:Number(maps)});
      submitted.current=result.id;
      setSnapshot(previous=>({datasets:previous?.datasets ?? [],job:result,active:result}));
    } catch(error) {setError(error instanceof Error?error.message:'无法添加地图组');}
    finally {setPending(false);}
  }
  if(dataset) return <DatasetDetail dataset={snapshot?.datasetDetail?.id===dataset?snapshot.datasetDetail:snapshot?.datasets.find(entry=>entry.id===dataset)} loading={!snapshot} error={pollError} onBack={()=>onDataset(undefined)} />;
  return <section className="training-workspace" aria-label="本课地图组">
    <div className="dataset-heading"><h2>地图组</h2>
      <Button id="add-dataset" variant="solid" onClick={()=>{setError('');setOpen(true);}}>添加地图组</Button>
    </div>
    {pollError && <Alert tone="danger" role="alert">{pollError}</Alert>}
    {!snapshot && !pollError && <p role="status">正在读取地图组…</p>}
    {snapshot && !snapshot.datasets.length && <EmptyState><EmptyStateTitle>本课暂无地图组，点击「添加地图组」创建。</EmptyStateTitle></EmptyState>}
    <Table pagination={tablePagination(snapshot?.pagination,setPage,size=>{setPageSize(size);setPage(1);})} aria-label="本课地图组列表" containerClassName="dataset-list" containerProps={{tabIndex:0}}>
      <TableHeader><TableRow>
        {['名称','说明','训练地图','验证地图','测试地图','种子','创建时间','详情'].map(label=><TableHead key={label} scope="col">{label}</TableHead>)}
      </TableRow></TableHeader>
      <TableBody>{snapshot?.datasets.map(entry=><TableRow key={entry.id} data-dataset-id={entry.id}>
        <TableHead scope="row">{entry.title}</TableHead>
        <TableCell className="dataset-description">{entry.description || '—'}</TableCell>
        <TableCell>{entry.maps} 张</TableCell><TableCell>{entry.validation} 张</TableCell><TableCell>{entry.test} 张</TableCell>
        <TableCell>{entry.seed}</TableCell>
        <TableCell><time dateTime={entry.createdAt}>{new Date(entry.createdAt).toLocaleString('zh-CN',{
          timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23',
        })}</time></TableCell>
        <TableCell><Button size="sm" onClick={()=>onDataset(entry.id)}>详情</Button></TableCell>
      </TableRow>)}</TableBody>
    </Table>
    {job?.action==='prepare' && <p id="dataset-status" role="status">{busyJob(job)?'地图组生成中…':job.state==='succeeded'?'地图组已添加':job.error ?? '地图组生成未完成'}</p>}
    <Dialog open={open} onOpenChange={setOpen} className="dataset-dialog" bodyClassName="task-dialog-form"
      title="添加地图组" description={fixed?'第一课使用固定地图。':'为当前课时生成训练地图，另含 64 张验证地图和 100 张测试地图。'}
      form={
        <form id="maze-dataset-form" onSubmit={event=>{event.preventDefault();if(!blocked) void create();}}>
          <div className="training-fields">
            <Field><FieldLabel>地图组名称</FieldLabel><Input id="dataset-title" maxLength={80} value={title} disabled={blocked} onChange={event=>setTitle(event.currentTarget.value)} /></Field>
            <Field><FieldLabel>地图组说明</FieldLabel><Textarea id="dataset-description" maxLength={500} value={description} disabled={blocked} onChange={event=>setDescription(event.currentTarget.value)} /></Field>
            <Field><FieldLabel>随机种子</FieldLabel><Input id="dataset-seed" type="number" required min="0" max="2147483647" step="1" value={seed} disabled={blocked} onChange={event=>setSeed(event.currentTarget.value)} /></Field>
            <Field><FieldLabel>训练地图数量</FieldLabel><Input id="training-maps" type="number" required min={fixed?1:16} max="512" step="1" value={fixed?'1':maps} disabled={fixed || blocked} onChange={event=>setMaps(event.currentTarget.value)} /></Field>
          </div>
          {(error || pollError) && <Alert tone="danger" role="alert">{error || pollError}</Alert>}
          {snapshot?.active && <p role="status">{snapshot.active.action==='prepare'?'正在生成地图组，请等待完成。':'有训练任务正在运行，请完成后再添加。'}</p>}
        </form>
      }
      footer={<>
        <Button type="button" onClick={()=>setOpen(false)}>关闭</Button>
        <Button id="prepare-dataset" type="submit" form="maze-dataset-form" variant="solid" disabled={blocked}>添加地图组</Button>
      </>}/>
  </section>;
}
