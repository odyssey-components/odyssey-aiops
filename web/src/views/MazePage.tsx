import {Page,PageHeader,PageTitle} from '@odyssey/ui/primitives/page';
import {ToggleGroup} from '@odyssey/ui/primitives/toggle-group';
import {ToggleButton} from '@odyssey/ui/primitives/toggle-button';
import {Stat,StatLabel,StatValue} from '@odyssey/ui/primitives/stat';
import {EmptyState,EmptyStateTitle} from '@odyssey/ui/primitives/empty-state';
import { Tabs, TabsList, TabsTab, TabsPanel } from '@odyssey/ui/primitives/tabs';
import { ModelView } from './ModelView';
import { DatasetView } from './DatasetView';
import {MazeRecordings} from './MazeRecordings';
import {saveSharedRecording} from '../apis/recordings';
import { TrainingView } from './TrainingView';
import { Board } from "../components/maze/Board";
import {Decisions} from "../components/maze/Decisions";
import {loadOnnxModel,type OnnxRuntimeModel} from "../utilities/onnx-runtime";
import {useOnnxPrediction} from "../utilities/useOnnxPrediction";
import { Button } from "@odyssey/ui/primitives/button";
import { Card } from "@odyssey/ui/primitives/card";
import { useEffect, useReducer, useRef, useState } from "react";
import { ACTIONS, HEADINGS, terminal, modelLesson, type LessonId } from "../../core";
import { Input } from "@odyssey/ui/primitives/input";
import { Badge } from "@odyssey/ui/primitives/badge";
import { Choice } from "../components/common/Choice";
import { MazeShell } from "../components/layout/MazeShell";
import { RecurrentMemory } from "../components/maze/RecurrentMemory";
import { Exploration } from "../components/maze/Exploration";
import { startGame, updateGame } from "../utilities/game";
import {DEFAULT_SEARCH,searchKey,type MazeSearch} from "../utilities/search";
import {ModelPicker} from '../components/models/ModelPicker';
import {defaultModel,listModels,type ModelSource,type ModelEntry} from '../apis/models';
import {firstCourseModel} from '../utilities/models';

const ARROWS = ["→", "↓", "←", "↑"];
const MODES = [
  { id: "manual", label: "手动", name: "手动控制", description: "点击下方动作，观察位置和朝向的变化。" },
  { id: "expert", label: "专家", name: "专家示范", description: "专家查看整张地图，沿最短路径选择动作。" },
  { id: "model", label: "模型", name: "模型推理", description: "模型根据本课观测编码，选择概率最大的动作。" },
] as const;


export function MazePage({ initialRuns,search=DEFAULT_SEARCH,onSearchChange }: { initialRuns: ModelEntry[] | null; search?:MazeSearch; onSearchChange?:(search:MazeSearch)=>void }) {
  const [game, dispatch] = useReducer(updateGame, undefined, () => updateGame(startGame(search.course,search.seed),{type:"configure",lesson:search.course,seed:search.seed,mode:search.mode,speed:search.speed}));
  const [seedInput, setSeedInput] = useState(String(search.seed));
  const [info, setInfo] = useState("尚未加载模型");
  const [message, setMessage] = useState<{ text: string; kind: "neutral" | "error" | "success" }>({ text: "", kind: "neutral" });
  const version = useRef(0);
  const applied = useRef<string | null>(null);
  const currentSearch = useRef(search);
  const loadedRun = useRef<string | undefined>(undefined);
  const runtime = useRef<OnnxRuntimeModel|null>(null);
  function clearRuntime() {const session=runtime.current;runtime.current=null;void session?.dispose().catch(()=>{});}
  useOnnxPrediction(game,runtime,dispatch);
  const [autoCatalog,setAutoCatalog]=useState({entries:initialRuns,revision:0});
  const handledRefresh=useRef(0);
  useEffect(()=>{setAutoCatalog(previous=>previous.entries===initialRuns ? previous : {...previous,entries:initialRuns});},[initialRuns]);
  function publish(patch:Partial<MazeSearch>) {
    const next={...currentSearch.current,...patch}; delete next.modelError;
    currentSearch.current=next; applied.current=searchKey(next); onSearchChange?.(next);
  }

  const { state, mode, model, running, loading, speed, visited, log, maze } = game;
  const random = game.lesson === "random-map-v1";
  const recurrent = game.lesson === "memory-map-v1";
  const local = game.lesson === "local-map-v1";
  const ended = terminal(state);
  const pending = !!game.inferenceRequest;
  const values = model && !ended ? game.prediction : null;
  const manual = mode === "manual";
  const playbackDisabled = manual || ended || loading || pending || !!game.inferenceError;

  const [recordingEpisode,setRecordingEpisode]=useState<object|null>(null);
  const [recordingStatus,setRecordingStatus]=useState('');
  const [recordingVersion,setRecordingVersion]=useState(0);
  const recording=recordingEpisode===game.episode;
  async function toggleRecording() {
    if (!recording) {
      if (game.state.steps!==0 || running || loading) return;
      setRecordingStatus('');setRecordingEpisode(game.episode);
      return;
    }
    dispatch({type:'stop'});
    if (!game.actions.length) {
      setRecordingEpisode(null);setRecordingStatus('录制已取消，尚无动作。');
      return;
    }
    try {
      await saveSharedRecording({
        format:'maze-recording/v1',id:crypto.randomUUID(),createdAt:new Date().toISOString(),
        lesson:game.lesson,seed:game.seed,maze:game.maze,actions:game.actions,
      });
      setRecordingVersion(value=>value+1);setRecordingEpisode(null);setRecordingStatus('录制已保存，可在「回放」或「回灌」查看。');
    } catch(error) {setRecordingStatus(error instanceof Error?error.message:'保存录制失败');}
  }

  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => dispatch({ type: "tick" }), speed);
    return () => window.clearInterval(timer);
  }, [running, speed]);

  useEffect(() => () => {
    ++version.current;
    // StrictMode replays mount effects: cancelled URL reads must be restartable.
    applied.current = null;
    clearRuntime();
  }, []);

  useEffect(() => {
    const stop = () => dispatch({ type: "stop" });
    const onVisibility = () => { if (document.hidden) stop(); };
    window.addEventListener("pagehide", stop);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("pagehide", stop);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  useEffect(() => {
    const key=searchKey(search);
    if(applied.current===key) return;
    applied.current=key; currentSearch.current=search;
    ++version.current;
    const clearModel=loadedRun.current!==search.model;
    if(clearModel) {clearRuntime();loadedRun.current=undefined;setInfo("尚未加载该 URL 指定的模型。");}
    dispatch({type:"configure",lesson:search.course,seed:search.seed,mode:search.mode,speed:search.speed,clearModel});
    setSeedInput(String(search.seed));
    if(search.modelError) setMessage({kind:"error",text:search.modelError});
    else if(search.model && (loadedRun.current!==search.model || game.lesson!==search.course)) {
      void loadModel(defaultModel(search.model),search);
    } else if(search.mode==="model" && !game.model && !search.model) {
      setMessage({kind:"neutral",text:"本课模型加载后可使用模型控制。"});
    }
  },[searchKey(search)]);

  useEffect(()=>{
    const refreshed=autoCatalog.revision!==handledRefresh.current;
    handledRefresh.current=autoCatalog.revision;
    if(!refreshed && (currentSearch.current.model || game.model)) return;
    const first=firstCourseModel(autoCatalog.entries,game.lesson);
    if(!first) return;
    void loadModel(defaultModel(first.run),{...currentSearch.current,course:game.lesson,mode:currentSearch.current.mode},true);
  },[game.lesson,game.seed,autoCatalog,search.model]);

  async function loadModel(source: ModelSource,restore?:MazeSearch,automatic=false) {

    const request = ++version.current;
    dispatch({ type: "loading" });
    setMessage({ kind: "neutral", text: "正在读取模型…" });
    try {
      const candidate=await loadOnnxModel(source.run);
      if(request!==version.current) {await candidate.dispose();return;}
      const loaded=candidate.policy;
      if(modelLesson(loaded)!==(restore?.course ?? game.lesson)) {await candidate.dispose();throw new Error("模型与当前课时不同");}
      const previous=runtime.current;runtime.current=candidate;
      void previous?.dispose().catch(()=>{});
      loadedRun.current=source.run;
      dispatch({ type: "loaded", model: loaded,mode:restore?.mode,external:true });
      if(!restore || automatic) publish({course:modelLesson(loaded),mode:restore?.mode ?? "model",model:source.run});
      const meta = loaded.metadata;
      setInfo(`${modelLesson(loaded) === "memory-map-v1" ? "第四课" : modelLesson(loaded) === "local-map-v1" ? "第三课" : modelLesson(loaded) === "random-map-v1" ? "第二课" : "第一课"} · ${source.name} · ${meta.epochs} 轮 · ${meta.samples} 条样本 · 训练集正确率 ${(meta.accuracy * 100).toFixed(1)}%${meta.test_success_rate === undefined ? "" : ` · 独立测试地图通关率 ${(meta.test_success_rate * 100).toFixed(1)}%`}`);
      setMessage({ kind: "success", text: "已加载 ONNX 模型" });
    } catch (error) {
      if (request !== version.current) return;
      dispatch({ type: "failed" });
      const detail = error instanceof Error ? error.message : "无法读取模型";
      setMessage({ kind: "error", text: `${detail}${restore && !automatic ? "；未启用该 URL 模型。" : model ? "；已保留上一个有效模型。" : "。"}` });
    }
  }

  function changeCourse(course: string) {
    ++version.current;
    loadedRun.current=undefined;
    clearRuntime();
    dispatch({ type: "course", lesson: course as LessonId });
    publish({course:course as LessonId,mode:"manual",model:undefined,modelDetail:undefined,dataset:undefined,task:undefined});
    setInfo("正在查找本课可用模型。");
    setMessage({kind:"neutral",text:""});
  }

  function newMaze() {
    const seed = Number(seedInput);
    if (!seedInput.trim() || !Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
      setMessage({kind:"error",text:"种子必须是 0 到 4294967295 的整数"}); return;
    }
    ++version.current;
    dispatch({type:"maze",seed});
    publish({seed});
    setMessage({kind:"neutral",text:"已生成迷宫；本课模型仍可用于当前地图。"});
  }

  function reset() {
    ++version.current;
    if (loading) setMessage({ kind: "neutral", text: "" });
    dispatch({ type: "reset" });
  }

  async function refreshTrainedModels() {
    const entries=await listModels().catch(()=>{throw new Error('模型已训练完成，但列表刷新失败，请在实验页刷新模型列表');});
    setAutoCatalog(previous=>({entries,revision:previous.revision+1}));
  }

  return <MazeShell onModule={module=>publish({module:module==='go1'?'go1':undefined})} course={game.lesson} onCourse={changeCourse}><Page>
      <PageHeader><PageTitle>从第一步，理解模型。</PageTitle></PageHeader>
      <Tabs value={search.tab ?? 'experiment'} onValueChange={value=>{dispatch({type:'stop'});publish({tab:value==='training' || value==='datasets' || value==='models' || value==='recordings' || value==='playback' || value==='reinjection' || value==='simulation'?value:undefined,dataset:undefined,task:undefined,modelDetail:undefined});}}>
      <TabsList aria-label="课时工作区"><TabsTab value="experiment">实验</TabsTab><TabsTab value="training">训练任务</TabsTab><TabsTab value="datasets">地图组</TabsTab><TabsTab value="models">模型</TabsTab><TabsTab value="recordings">录制</TabsTab><TabsTab value="playback">回放</TabsTab><TabsTab value="reinjection">回灌</TabsTab><TabsTab value="simulation">仿真任务</TabsTab></TabsList>
      {(['playback','reinjection','simulation'] as const).map(tab=><TabsPanel key={tab} value={tab}>
        {search.tab===tab&&<MazeRecordings key={game.lesson} view={tab} lesson={game.lesson} initialSeed={game.seed}
          models={(autoCatalog.entries ?? []).filter(entry=>entry.lesson===game.lesson).map(entry=>({value:entry.run,label:entry.run}))}/>}
      </TabsPanel>)}
      <TabsPanel value="models"><ModelView paginated key={game.lesson} course={game.lesson} catalog={autoCatalog.entries}
        selectedRun={model ? loadedRun.current : undefined} loading={loading} onRefresh={refreshTrainedModels}
        detail={search.modelDetail} onDetail={modelDetail=>publish({modelDetail})} /></TabsPanel>
      <TabsPanel value="datasets"><DatasetView key={game.lesson} course={game.lesson} dataset={search.dataset} onDataset={dataset=>publish({dataset})} /></TabsPanel>
      <TabsPanel value="training" keepMounted><TrainingView key={game.lesson} course={game.lesson} onFinished={refreshTrainedModels} selectedTask={search.task} onSelectTask={task=>publish({task})} /></TabsPanel>
      <TabsPanel value={search.tab==='recordings'?'recordings':'experiment'} keepMounted>
      <section className="workspace" aria-label="迷宫实验">
        <Card className="card maze-card">
          <div className="card-heading"><h2>一张地图，三种走法</h2><Badge id="map-kind">7 × 7 · {recurrent ? "模型记忆" : local ? "局部探索" : random ? "随机地图" : "固定地图"}</Badge></div>
          <ToggleGroup block aria-label="控制方式" value={[mode]} onValueChange={values=>{
            const next=values[0];if(next==='manual' || next==='expert' || next==='model') {dispatch({type:'mode',mode:next});publish({mode:next});}
          }}>
            {MODES.map(({ id, label }) => <ToggleButton key={id} value={id} size="sm" id={`mode-${id}`} data-mode={id}
              disabled={loading || (id === "model" && (!model || (pending && mode!=="model") || !!game.inferenceError))} >{label}</ToggleButton>)}
          </ToggleGroup>
          <Board state={state} visited={visited} maze={maze} memory={game.memory} />
          {game.memory && <Exploration memory={game.memory} state={state} />}
          <div className="legend"><span><i className="legend-agent" />角色</span><span><i className="legend-goal" />出口</span><span><i className="legend-trail" />走过的格子</span>{(local || recurrent) && <span>？ 未知区域 · 描边为当前视野</span>}</div>
          <div className="stats">
            <Stat><StatLabel>当前位置</StatLabel><StatValue id="position">({state.x}, {state.y})</StatValue></Stat>
            <Stat><StatLabel>朝向</StatLabel><StatValue id="heading">{HEADINGS[state.direction]} {ARROWS[state.direction]}</StatValue></Stat>
            <Stat><StatLabel>已走步数</StatLabel><StatValue id="step-count">{state.steps} / {maze.max_steps}</StatValue></Stat>
          </div>
          <div className="game-feedback">
            <span id="game-status" role="status" data-status={state.status}>{state.status === "success" ? "已到达出口"
              : state.status === "timeout" ? "达到步数上限" : state.steps ? "继续向出口前进" : "准备好，迈出第一步"}</span>
            <span id="running-state" aria-busy={pending}>{loading ? "读取模型中" : running ? "自动运行中" : "已暂停"}</span>
          </div>
          <div id="manual-controls" className="manual-controls" role="group" aria-label="手动动作" hidden={!manual}>
            {[0, 2, 1].map((action) => <Button key={action} data-action={action} disabled={loading || pending || !!game.inferenceError || ended}
              onClick={() => dispatch({ type: "step", action })}><span aria-hidden="true">{["↶", "↷", "↑"][action]}</span>{ACTIONS[action]}</Button>)}
          </div>
          <div className="playback-controls">
            <Button id="single-step" disabled={playbackDisabled || running} onClick={() => dispatch({ type: "step" })}>执行一步</Button>
            <Button id="auto" variant="solid" disabled={playbackDisabled && !running}
              onClick={() => dispatch({ type: "toggle" })}>{running ? "暂停" : "自动运行"}</Button>
            <Button id="reset" onClick={reset}>重置</Button>
            <div className="speed-label">速度<Choice id="speed" label="速度" value={String(speed)} onChange={value=>{dispatch({type:"speed",speed:Number(value)});publish({speed:Number(value)});}} options={[{value:"650",label:"慢速"},{value:"250",label:"正常"},{value:"40",label:"快速"}]} /></div>
          </div>
          {search.tab==='recordings'&&<div className="training-actions">
            <Button onClick={toggleRecording} disabled={!recording && (state.steps!==0 || running || loading)}>{recording?'结束并保存录制':'开始录制'}</Button>
            <span role="status">{recording?`录制中 · ${game.actions.length} 步`:recordingStatus || (recordingEpisode?'地图或模型已重置，录制已结束。':state.steps>0?'重置到起点后可开始录制。':'')}</span>
          </div>}
        </Card>
        <aside className="inspector">
          <ModelPicker onLoad={source=>void loadModel(source,{...currentSearch.current},true)} initialRuns={autoCatalog.entries} course={game.lesson} selectedRun={model ? loadedRun.current : undefined} loading={loading} onCatalogChange={entries=>setAutoCatalog(previous=>({entries,revision:previous.revision+1}))} info={info} message={game.inferenceError ? {kind:"error",text:game.inferenceError} : message} />
          {(random || local || recurrent) && <div className="seed-controls"><Input id="maze-seed" aria-label="地图种子" title="地图种子" size="sm" type="number" min="0" max="4294967295" step="1" value={seedInput} onChange={event=>setSeedInput(event.currentTarget.value)} /><Button size="sm" id="new-maze" onClick={newMaze}>生成新迷宫</Button></div>}
          <Decisions state={state} mode={mode} values={values} maze={maze} course={game.lesson} memory={game.memory} previousAction={game.previousAction} pending={pending} predictionIdentity={model} />
          {recurrent && <RecurrentMemory state={game.recurrent} disabled={loading || pending} onForget={()=>dispatch({type:"forget"})} />}
        </aside>
      </section>
      <section className="bottom-grid">
        <Card className="card log-card">
          <div className="card-heading"><h2>刚刚发生了什么</h2><span className="tiny-label">最近 8 步</span></div>
          <ol id="action-log">{log.length ? [...log].reverse().map((entry) => <li key={entry.step}>
            <span className="log-count">{String(entry.step).padStart(2, "0")}</span><span>{ACTIONS[entry.action]} · {entry.description}</span>
          </li>) : <li><EmptyState><EmptyStateTitle>暂无记录</EmptyStateTitle></EmptyState></li>}</ol>
        </Card>
      </section>
      {search.tab==='recordings'&&<MazeRecordings key={`${game.lesson}-${recordingVersion}`} view="recordings" lesson={game.lesson} models={[]}/>}
      </TabsPanel></Tabs>
  </Page></MazeShell>;
}
