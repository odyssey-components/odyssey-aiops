import {tablePagination,useLocalTablePagination,type PageInfo} from '../utilities/list-pagination';
import {importLocalRecordings,listRecordings,removeSharedRecording,saveSharedRecording} from '../apis/recordings';
import {AlertDialog,AlertDialogTrigger,AlertDialogContent,AlertDialogTitle,AlertDialogDescription,AlertDialogFooter,AlertDialogClose} from '@odyssey/ui/primitives/alert-dialog';
import {useEffect,useMemo,useRef,useState} from 'react';
import {Button} from '@odyssey/ui/primitives/button';
import {Dialog} from '@odyssey/ui/primitives/dialog';
import {Drawer,DrawerContent,DrawerBody} from '@odyssey/ui/primitives/drawer';
import {Card} from '@odyssey/ui/primitives/card';
import {Alert} from '@odyssey/ui/primitives/alert';
import {Table,TableHeader,TableBody,TableRow,TableHead,TableCell} from '@odyssey/ui/primitives/table';
import {Input} from '@odyssey/ui/primitives/input';
import {startGame} from '../utilities/game';
import {ACTIONS,type LessonId} from '../../core';
import {Board} from '../components/maze/Board';
import {Choice} from '../components/common/Choice';
import {loadOnnxModel} from '../utilities/onnx-runtime';
import {compareRecording,recordingFrames,simulateMaze,type MazeComparisonStep,type MazeRecording} from '../utilities/recordings';

function DeleteRecordingButton({label,onDelete}:{label:string;onDelete:()=>void}) {
 return <AlertDialog>
  <AlertDialogTrigger render={<Button>删除{label}</Button>}/>
  <AlertDialogContent>
   <AlertDialogTitle>删除这条{label}？</AlertDialogTitle>
   <AlertDialogDescription>这条记录将从当前浏览器移除。</AlertDialogDescription>
   <AlertDialogFooter>
    <AlertDialogClose render={<Button>取消</Button>}/>
    <AlertDialogClose render={<Button variant="solid" tone="danger" onClick={onDelete}>确认删除</Button>}/>
   </AlertDialogFooter>
  </AlertDialogContent>
 </AlertDialog>;
}

export function MazeRecordings({lesson,models,view,initialSeed=7}:{initialSeed?:number;view:'recordings'|'playback'|'reinjection'|'simulation';lesson:LessonId;models:{value:string;label:string}[]}) {
 const actionPagination=useLocalTablePagination(1);
 const [records,setRecords]=useState<MazeRecording[]>([]);
 const [pageSize,setPageSize]=useState(20);
 const [page,setPage]=useState(1),[version,setVersion]=useState(0);
 const [pagination,setPagination]=useState<PageInfo>();
 const [loading,setLoading]=useState(true);
 const [importError,setImportError]=useState('');
 const [created,setCreated]=useState<MazeRecording>();
 const [seed,setSeed]=useState(String(initialSeed));
 const [selected,setSelected]=useState('');
 const [replayId,setReplayId]=useState('');
 const [index,setIndex]=useState(0),[playing,setPlaying]=useState(false);
 const [model,setModel]=useState(''),[result,setResult]=useState<{model:string;steps:MazeComparisonStep[]}>(),[busy,setBusy]=useState(false);
 const [comparisonIndex,setComparisonIndex]=useState(0);
 const comparison=result?.steps[comparisonIndex];
 const [error,setError]=useState('');
 const controller=useRef<AbortController|null>(null);
 const label={recordings:'录制记录',playback:'回放',reinjection:'回灌',simulation:'仿真任务'}[view];
 const record=created?.id===selected?created:records.find(item=>item.id===selected);
 const candidate=models.find(item=>item.value===model) ?? models[0];
 const frames=useMemo(()=>record?recordingFrames(record):[],[record]);
 const replay=created?.id===replayId?created:records.find(item=>item.id===replayId);
 const replayFrames=useMemo(()=>replay?recordingFrames(replay):[],[replay]);
 const frame=replayFrames[index];
 function openReplay(id:string) {
  setPlaying(false);setIndex(0);setReplayId(id);
 }
 useEffect(()=>{
  const request=new AbortController();
  setLoading(true);
  async function load(){
   try {
    await importLocalRecordings();
    if(!request.signal.aborted)setImportError('');
   }catch {
    if(!request.signal.aborted)setImportError('本地旧录制尚未导入，共享列表仍可查看。');
   }
   if(request.signal.aborted)return;
   try {
    const result=await listRecordings(lesson,view,page,request.signal,pageSize);
    if(!request.signal.aborted){setRecords(result.items);setPagination(result.pagination);}
   }catch(reason){if(!request.signal.aborted)setError(String(reason));}
   finally{if(!request.signal.aborted)setLoading(false);}
  }
  void load();
  return ()=>request.abort();
 },[lesson,view,page,pageSize,version]);
 useEffect(()=>()=>controller.current?.abort(),[]);
 useEffect(()=>{
  if (!playing || index>=replayFrames.length-1) return;
  const timer=window.setTimeout(()=>{
   setIndex(index+1);
   if(index+1===replayFrames.length-1)setPlaying(false);
  },250);
  return ()=>window.clearTimeout(timer);
 },[playing,index,replayFrames.length]);

 function select(id:string) {
  controller.current?.abort();controller.current=null;
  setPlaying(false);setIndex(0);setSelected(id);setResult(undefined);setBusy(false);setError('');
 }
 async function compare() {
  if(!record||!candidate||controller.current)return;
  const request=new AbortController();
  controller.current=request;setBusy(true);setError('');setResult(undefined);setComparisonIndex(0);
  try {
   const session=await loadOnnxModel(candidate.value);
   try {
    const steps=await compareRecording(record,session,request.signal);
    if(!request.signal.aborted)setResult({model:candidate.label,steps});
   } finally {await session.dispose();}
  } catch(reason){if(!request.signal.aborted)setError(String(reason));}
  finally {if(controller.current===request){controller.current=null;setBusy(false);}}
 }
 async function simulate() {
  if (!candidate || controller.current) return;
  const mapSeed=Number(seed);
  if (!seed.trim() || !Number.isInteger(mapSeed) || mapSeed<0 || mapSeed>0xffffffff) {
   setError('种子必须是 0 到 4294967295 的整数');return;
  }
  const request=new AbortController();
  controller.current=request;setBusy(true);setError('');
  try {
   const session=await loadOnnxModel(candidate.value);
   try {
    const {maze}=startGame(lesson,mapSeed);
    const {actions}=await simulateMaze(lesson,maze,session,request.signal);
    if(request.signal.aborted)return;
    const task={format:'maze-recording/v1' as const,id:crypto.randomUUID(),createdAt:new Date().toISOString(),
     lesson,seed:mapSeed,maze,actions,simulationModel:candidate.value};
    await saveSharedRecording(task);setCreated(task);setSelected(task.id);setPage(1);setVersion(value=>value+1);
   } finally {await session.dispose();}
  } catch(reason){if(!request.signal.aborted)setError(String(reason));}
  finally {if(controller.current===request){controller.current=null;setBusy(false);}}
 }
 function cancelRun() {
  controller.current?.abort();controller.current=null;setBusy(false);
 }
 async function remove(id:string) {
  try {await removeSharedRecording(id);if(created?.id===id)setCreated(undefined);select('');setVersion(value=>value+1);}
  catch(reason){setError(String(reason));}
 }
 return <section className="training-task-detail" aria-label={`Maze ${label}`}>
  <div className="dataset-heading"><h2>{label}</h2>{record&&view!=='simulation'&&<Button onClick={()=>select('')}>返回录制列表</Button>}</div>
  {importError&&<p role="status">{importError}</p>}
  {loading&&<p role="status">正在读取共享记录…</p>}
  {error&&<Alert role="alert" tone="danger">{error}</Alert>}
  {view==='simulation'&&<Card className="card">
   <div className="training-actions">
    <Choice id="maze-simulation-model" label="仿真模型" value={candidate?.value ?? ''} onChange={setModel} options={models} disabled={busy || !candidate} placeholder="暂无同课模型"/>
    {lesson!=='fixed-map-v1'&&<Input aria-label="仿真地图种子" type="number" min={0} max={4294967295} step={1} value={seed} onChange={event=>setSeed(event.currentTarget.value)} disabled={busy}/>}
    <Button disabled={!candidate||busy} onClick={()=>void simulate()}>{busy?'仿真运行中…':'创建并运行仿真'}</Button>
    {busy&&<Button onClick={cancelRun}>取消仿真</Button>}
   </div>
  </Card>}
  {(!record||view==='simulation')&&<><Table pagination={tablePagination(pagination,value=>{select('');setReplayId('');setPage(value);},size=>{select('');setReplayId('');setPageSize(size);setPage(1);})} aria-label={view==='simulation'?'Maze 仿真任务列表':'Maze 录制列表'} containerClassName="dataset-list">
   <TableHeader><TableRow>{['时间','来源','地图种子','动作数','操作'].map(label=><TableHead scope="col" key={label}>{label}</TableHead>)}</TableRow></TableHeader>
   <TableBody>{records.map(item=><TableRow key={item.id}>
    <TableCell>{new Date(item.createdAt).toLocaleString('zh-CN')}</TableCell>
    <TableCell>{item.simulationModel?'仿真':'实验录制'}</TableCell><TableCell>{item.seed}</TableCell><TableCell>{item.actions.length}</TableCell>
    <TableCell><div className="training-actions"><Button size="sm" onClick={()=>view==='playback'?openReplay(item.id):view==='simulation'?setSelected(item.id):select(item.id)}>{view==='playback'?'打开回放':view==='reinjection'?'选择录制':view==='simulation'?'查看结果':'查看录制'}</Button>{view==='simulation'&&<Button size="sm" onClick={()=>openReplay(item.id)}>回放</Button>}</div></TableCell>
   </TableRow>)}</TableBody>
  </Table>{!records.length&&<p>{view==='simulation'?'暂无已完成仿真':'暂无录制，可在「录制」标签从起点开始录制。'}</p>}</>}
  {replay&&frame&&<Dialog open onOpenChange={open=>{if(!open)openReplay('');}} className="dataset-dialog"
   title="Maze 回放" description={`地图种子 ${replay.seed} · ${replay.actions.length} 步`} cancel={{label:'关闭回放'}}
   content={<>
    <Board id="maze-replay-board" {...frame} maze={replay.maze}/>
    <div className="training-actions">
     <Button onClick={()=>{if(index===replayFrames.length-1)setIndex(0);setPlaying(!playing);}}>{playing?'暂停回放':'播放录制'}</Button>
     <Button onClick={()=>{setPlaying(false);setIndex(0);}}>回到起点</Button>
    </div>
    <label>回放步数 {index} / {replay.actions.length}
     <input type="range" min={0} max={replay.actions.length} value={index} onChange={event=>{setPlaying(false);setIndex(Number(event.currentTarget.value));}}/>
    </label>
   </>}/>}

  {record&&(view==='reinjection'||view==='recordings')&&<div>
   {view==='reinjection'&&<Card className="card">
    <h3>模型回灌</h3>
    <Choice id="maze-recording-model" label="待测模型" value={candidate?.value ?? ''} onChange={setModel} options={models} disabled={busy || !candidate} placeholder="暂无同课模型"/>
    <div className="training-actions">
     <Button disabled={!candidate || busy} onClick={()=>void compare()}>{busy?'回灌中…':'运行回灌'}</Button>
     {busy&&<Button onClick={cancelRun}>取消回灌</Button>}
    </div>
    {result&&comparison&&<>
     <p>{result.model} · 动作一致 {result.steps.filter((row,i)=>row.action===record.actions[i]).length} / {record.actions.length}</p>
     <div className="training-actions">
      <Button disabled={comparisonIndex===0} onClick={()=>setComparisonIndex(comparisonIndex-1)}>上一步</Button>
      <label>回灌步数 <input type="number" min={1} max={result.steps.length} value={comparisonIndex+1}
       onChange={event=>{const value=event.currentTarget.valueAsNumber;if(Number.isInteger(value)&&value>=1&&value<=result.steps.length)setComparisonIndex(value-1);}}/></label>
      <Button disabled={comparisonIndex===result.steps.length-1} onClick={()=>setComparisonIndex(comparisonIndex+1)}>下一步</Button>
     </div>
     <p>原录制动作 {ACTIONS[record.actions[comparisonIndex]]} · 待测动作 {ACTIONS[comparison.action]}</p>
     <Table pagination={actionPagination.options} aria-label="待测动作概率"><TableHeader><TableRow>{ACTIONS.map(action=><TableHead key={action} scope="col">{action}</TableHead>)}</TableRow></TableHeader>
      <TableBody><TableRow>{comparison.values.map((value,i)=><TableCell key={i}>{value.toPrecision(5)}</TableCell>)}</TableRow></TableBody>
     </Table>
     <details open><summary>输入观测（{comparison.observation.length} 维）</summary>
      <pre className="training-logs">{JSON.stringify(comparison.observation)}</pre>
     </details>
     {comparison.hidden&&<details><summary>输入 GRU 隐藏状态</summary><pre className="training-logs">{JSON.stringify(comparison.hidden)}</pre></details>}
    </>}
   </Card>}
   {view==='recordings'&&<Card className="card"><p>地图种子 {record.seed} · {record.actions.length} 步</p>
    <DeleteRecordingButton label="录制" onDelete={()=>remove(record.id)}/>
   </Card>}
  </div>}
  <Drawer open={view==='simulation'&&Boolean(record)} onOpenChange={open=>{if(!open)setSelected('');}}>
   {view==='simulation'&&record&&<DrawerContent title="仿真结果" closeLabel="关闭仿真结果" className="go1-simulation-drawer">
    <DrawerBody className="dataset-detail">
    <p>{frames.at(-1)?.state.status==='success'?'已到达出口':'达到步数上限'}</p>
    <p>模型 {record.simulationModel} · 地图种子 {record.seed} · {record.actions.length} 步</p>
    <p>轨迹已保存，可在「回放」中查看。</p>
    <DeleteRecordingButton label="仿真" onDelete={()=>remove(record.id)}/>
    </DrawerBody>
   </DrawerContent>}
  </Drawer>

 </section>;
}
