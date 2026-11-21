import {tablePagination} from '../utilities/list-pagination';
import {useEffect,useRef,useState} from 'react';
import {Alert} from '@odyssey/ui/primitives/alert';
import {Button} from '@odyssey/ui/primitives/button';
import {EmptyState,EmptyStateTitle} from '@odyssey/ui/primitives/empty-state';
import {Table,TableHeader,TableBody,TableRow,TableHead,TableCell} from '@odyssey/ui/primitives/table';
import type {LessonId} from '../../core';
import {ModelDetail} from './ModelDetail';
import {listModelPage,type ModelEntry} from '../apis/models';

export interface ModelViewProps {
  course:LessonId;paginated?:boolean;catalog:ModelEntry[]|null;selectedRun?:string;loading:boolean;
  onRefresh:()=>Promise<void>;detail?:string;onDetail:(run:string|undefined)=>void;
}
export function ModelView({course,paginated=false,catalog,selectedRun,loading,onRefresh,detail,onDetail}:ModelViewProps) {
  const [refreshing,setRefreshing]=useState(false),[error,setError]=useState('');
  const alive=useRef(true);
  useEffect(()=>{alive.current=true;return ()=>{alive.current=false;};},[]);
  const [pageSize,setPageSize]=useState(20);
 const [page,setPage]=useState(1),[version,setVersion]=useState(0);
  const [result,setResult]=useState<Awaited<ReturnType<typeof listModelPage>>>();
  useEffect(()=>{
    if(!paginated||detail)return;
    const controller=new AbortController();
    setError('');
    listModelPage(course,page,controller.signal,pageSize).then(data=>{
      if(!controller.signal.aborted)setResult(data);
    }).catch(()=>{
      if(!controller.signal.aborted)setError('无法读取模型列表，请刷新重试。');
    });
    return ()=>controller.abort();
  },[course,page,pageSize,version,paginated,detail]);
  const models=paginated?result?.items ?? []:catalog?.filter(entry=>entry.lesson===course) ?? [];
  async function refresh() {
    setRefreshing(true);setError('');
    try {if(paginated)setVersion(value=>value+1);else await onRefresh();} catch {if(alive.current) setError('无法读取模型列表，请重试。');}
    finally {if(alive.current) setRefreshing(false);}
  }
  if(detail) return <ModelDetail key={`${course}:${detail}`} course={course} run={detail} onBack={()=>onDetail(undefined)} />;
  return <section className="training-workspace" aria-label="本课模型">
    <div className="dataset-heading"><h2>模型</h2>
      <Button disabled={refreshing || loading} onClick={()=>void refresh()}>刷新模型列表</Button>
    </div>
    {(error || (!paginated&&catalog===null)) && <Alert tone="danger" role="alert">{error || '无法读取模型列表，请刷新重试。'}</Alert>}
    {refreshing && <p role="status">正在读取模型列表…</p>}
    {(paginated?!!result:catalog!==null) && !models.length && <EmptyState><EmptyStateTitle>本课暂无有效模型</EmptyStateTitle></EmptyState>}
    <Table pagination={tablePagination(result?.pagination,setPage,size=>{setPageSize(size);setPage(1);})} aria-label="本课模型列表" containerClassName="dataset-list" containerProps={{tabIndex:0}}>
      <TableHeader><TableRow><TableHead scope="col">模型</TableHead><TableHead scope="col">状态</TableHead><TableHead scope="col">操作</TableHead></TableRow></TableHeader>
      <TableBody>{models.map(entry=><TableRow key={entry.run}>
        <TableHead scope="row">{entry.run}/model.onnx</TableHead>
        <TableCell>{entry.run===selectedRun?'当前使用':'未使用'}</TableCell>
        <TableCell><Button size="sm" onClick={()=>onDetail(entry.run)}>详情</Button></TableCell>
      </TableRow>)}</TableBody>
    </Table>
  </section>;
}
