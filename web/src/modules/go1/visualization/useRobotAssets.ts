import {useEffect,useState} from 'react';
import {loadRobotAssets,type RobotAssets} from './assets';
export function useRobotAssets(){
 const [assets,setAssets]=useState<RobotAssets>(),[error,setError]=useState(''),[attempt,setAttempt]=useState(0);
 useEffect(()=>{
  let active=true;setError('');
  void loadRobotAssets().then(value=>{if(active)setAssets(value);},reason=>{if(active)setError(reason instanceof Error?reason.message:'Go1 模型加载失败');});
  return ()=>{active=false;};
 },[attempt]);
 return {assets,error,retry:()=>setAttempt(value=>value+1)};
}
