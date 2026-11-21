import {apiPath} from './path';
import type {LessonId} from '../../core';
import type {PageInfo} from '../utilities/list-pagination';
import {readRecordings,type MazeRecording} from '../utilities/recordings';

type RecordingView='recordings'|'playback'|'reinjection'|'simulation';

async function request(query:string,signal?:AbortSignal,command?:unknown) {
 const response=await fetch(apiPath('/api/maze-recordings'+query),command?
  {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(command),signal}:{signal});
 if(!response.ok){
  const error=await response.json().catch(()=>null);
  throw Error(error?.error ?? '无法访问共享录制');
 }
 return response.json();
}

export const saveSharedRecording=(record:MazeRecording)=>request('',undefined,{action:'save',record});
export const removeSharedRecording=(id:string)=>request('',undefined,{action:'delete',id});

export function listRecordings(course:LessonId,view:RecordingView,page:number,signal:AbortSignal,pageSize=20):Promise<{items:MazeRecording[];pagination:PageInfo}> {
 return request('?'+new URLSearchParams({course,view,page:String(page),pageSize:String(pageSize)}),signal);
}

const migrationKey='aiops/shared-recordings-imported/v1';
let migration:Promise<void>|undefined;

export function importLocalRecordings() {
 if(localStorage.getItem(migrationKey))return Promise.resolve();
 if(!migration) {
  migration=(async()=>{
   for(const record of readRecordings())await saveSharedRecording(record);
   localStorage.setItem(migrationKey,'1');
  })().finally(()=>{migration=undefined;});
 }
 return migration;
}
