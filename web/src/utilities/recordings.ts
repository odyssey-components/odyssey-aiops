import {bestAction,initialState,modelLesson,step,terminal,type LessonId,type Maze} from '../../core';
import {initialMemory,observe} from '../lessons/local_map_v1';
import {encodeGameObservation} from './useOnnxPrediction';
import type {OnnxRuntimeModel} from './onnx-runtime';

export interface MazeRecording {
 format: 'maze-recording/v1';
 id: string;
 createdAt: string;
 lesson: LessonId;
 seed: number;
 maze: Maze;
 actions: number[];
 simulationModel?: string;
}
const key='odymaze-recordings/v1';
const lessons=['fixed-map-v1','random-map-v1','local-map-v1','memory-map-v1'];
function validRecording(record:MazeRecording):boolean {
 const maze=record?.maze;
 const valid=record?.format==='maze-recording/v1' && typeof record.id==='string' &&
  typeof record.createdAt==='string' && Number.isFinite(Date.parse(record.createdAt)) &&
  lessons.includes(record.lesson) && Number.isInteger(record.seed) && record.seed>=0 && record.seed<=0xffffffff &&
  !!maze && Array.isArray(maze.rows) && maze.rows.length===7 &&
  maze.rows.every(row=>typeof row==='string' && row.length===7 && /^[.#SG]+$/.test(row)) &&
  Number.isInteger(maze.start?.x) && Number.isInteger(maze.start?.y) &&
  /^[.SG]$/.test(maze.rows[maze.start.y]?.[maze.start.x] ?? '') &&
  [0,1,2,3].includes(maze.start.direction) && Number.isInteger(maze.max_steps) &&
  maze.max_steps>0 && maze.max_steps<=200 && Array.isArray(record.actions) &&
  record.actions.length>0 && record.actions.length<=maze.max_steps &&
  record.actions.every(action=>[0,1,2].includes(action)) &&
  (record.simulationModel===undefined || (typeof record.simulationModel==='string' && record.simulationModel.length>0));
 if (!valid) return false;
 let state=initialState(maze);
 for (const action of record.actions) {
  if (terminal(state)) return false;
  state=step(maze,state,action);
 }
 return record.simulationModel===undefined || terminal(state);
}
export function readRecordings():MazeRecording[] {
 const raw=localStorage.getItem(key);
 if (!raw) return [];
 const records:MazeRecording[]=JSON.parse(raw);
 if (!Array.isArray(records)) throw Error('录制记录格式无效');
 return records.slice(0,100).filter(validRecording);
}
export function saveRecording(record:MazeRecording) {
 if (!validRecording(record)) throw Error('录制数据无效');
 const records=readRecordings();
 if (records.length>=100) throw Error('已保存 100 条录制，请先删除一条');
 localStorage.setItem(key,JSON.stringify([record,...records]));
}
export function deleteRecording(id:string) {
 localStorage.setItem(key,JSON.stringify(readRecordings().filter(record=>record.id!==id)));
}
export function recordingFrames(record:MazeRecording) {
 const {maze,lesson}=record;
 let state=initialState(maze);
 let memory=lesson==='local-map-v1'||lesson==='memory-map-v1'?initialMemory(maze,state):null;
 const visited=new Set([`${state.x},${state.y}`]);
 const frames=[{state,memory,visited:new Set(visited),previousAction:-1}];
 for (const action of record.actions) {
  if (terminal(state)) throw Error('录制包含结束后的动作');
  state=step(maze,state,action);
  if (memory) memory=observe(maze,state,memory);
  visited.add(`${state.x},${state.y}`);
  frames.push({state,memory,visited:new Set(visited),previousAction:action});
 }
 return frames;
}
type ObservationFrame = Pick<ReturnType<typeof recordingFrames>[number],'state'|'memory'|'previousAction'>;
function policyRunner(lesson:LessonId,maze:Maze,session:OnnxRuntimeModel,signal?:AbortSignal) {
 if (modelLesson(session.policy)!==lesson) throw Error('模型与当前课时不同');
 let hidden=lesson==='memory-map-v1'?Array<number>(32).fill(0):undefined;
 return async (frame:ObservationFrame)=>{
  signal?.throwIfAborted();
  const observation=encodeGameObservation({...frame,lesson,maze});
  const inputHidden=hidden;
  const result=await session.predict(observation,hidden);
  signal?.throwIfAborted();
  if (result.values.length!==3 || !result.values.every(Number.isFinite)) throw Error('模型动作输出无效');
  if (hidden) {
   if (result.hidden?.length!==32 || !result.hidden.every(Number.isFinite)) throw Error('模型记忆输出无效');
   hidden=result.hidden;
  }
  return {action:bestAction(result.values),observation,hidden:inputHidden,values:result.values};
 };
}

export interface MazeComparisonStep {action:number;observation:number[];hidden?:number[];values:number[]}

export async function compareRecording(record:MazeRecording,session:OnnxRuntimeModel,signal?:AbortSignal) {
 const predict=policyRunner(record.lesson,record.maze,session,signal);
 const predictions:MazeComparisonStep[]=[];
 for (const frame of recordingFrames(record).slice(0,-1)) predictions.push(await predict(frame));
 return predictions;
}

export async function simulateMaze(lesson:LessonId,maze:Maze,session:OnnxRuntimeModel,signal?:AbortSignal) {
 const predict=policyRunner(lesson,maze,session,signal);
 let state=initialState(maze),previousAction=-1;
 let memory=lesson==='local-map-v1'||lesson==='memory-map-v1'?initialMemory(maze,state):null;
 const actions:number[]=[];
 while(!terminal(state)) {
  previousAction=(await predict({state,memory,previousAction})).action;
  actions.push(previousAction);
  state=step(maze,state,previousAction);
  if(memory)memory=observe(maze,state,memory);
 }
 return {actions,state};
}
