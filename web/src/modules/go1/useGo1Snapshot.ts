import {useCallback,useEffect,useRef,useState} from 'react';
import type {MazeSearch} from '../../utilities/search';
import {go1Snapshot} from './api';
import type {Go1Snapshot} from './contract';
import {go1NeedsPolling,reuseGo1Snapshot} from './snapshot';

export function useGo1Snapshot(section:MazeSearch['go1Tab'],page:number,selected?:string,replay?:string,pageSize=20) {
 const [snapshot,setSnapshot]=useState<Go1Snapshot|null>(null),[error,setError]=useState('');
 const mounted=useRef(false),sequence=useRef(0);
 useEffect(()=>{mounted.current=true;return ()=>{mounted.current=false;sequence.current++;};},[]);
 const refresh=useCallback(async(signal?:AbortSignal)=>{
  const request=++sequence.current;
  try {
   const query=new URLSearchParams({list:section ?? 'training',page:String(page),pageSize:String(pageSize)});
   if(selected)query.set('selected',selected);
   if(replay)query.set('replayId',replay);
   const next=await go1Snapshot(signal,query);
   if(mounted.current&&!signal?.aborted&&request===sequence.current){setSnapshot(previous=>reuseGo1Snapshot(previous,next));setError('');}
  }catch(error){
   if(mounted.current&&!signal?.aborted&&request===sequence.current)setError(error instanceof Error?error.message:String(error));
   throw error;
  }
 },[section,page,pageSize,selected,replay]);
 // Load once when entering a section; static pages stay stable until an action or navigation.
 useEffect(()=>{const controller=new AbortController();void refresh(controller.signal).catch(()=>{});return ()=>controller.abort();},[section,refresh]);
 const live=go1NeedsPolling(snapshot,section);
 useEffect(()=>{
  if(!live)return;
  let disposed=false,timer:ReturnType<typeof setTimeout>;
  const controller=new AbortController();
  async function poll(){try{await refresh(controller.signal);}catch{/* Keep the last snapshot and retry active work. */}finally{if(!disposed)timer=setTimeout(()=>void poll(),1500);}}
  timer=setTimeout(()=>void poll(),1500);
  return ()=>{disposed=true;controller.abort();clearTimeout(timer);};
 },[live,section,refresh]);
 return {snapshot,error,refresh};
}
