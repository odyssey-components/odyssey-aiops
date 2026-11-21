import {createContext,useContext,useEffect,useState,type ReactNode} from 'react';
import {ProAdminLogin,type ProAdminLoginValue} from '@odyssey/ui/patterns/admin-login';
import {loginDestination,loginURL} from './login-url';
import {apiPath} from '../apis/path';

async function authCommand(action:'login'|'logout',body:object){
 return fetch(apiPath('/api/auth/'+action),{
  method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),
 });
}

const UsernameContext=createContext<string|undefined>(undefined);

export function useSessionAccount(){
 const username=useContext(UsernameContext);
 if(!username)return undefined;
 return {user:{displayName:username,roleLabel:''},onLogout:async()=>{
  if(!(await authCommand('logout',{})).ok)throw new Error('退出登录失败');
  window.location.replace(loginURL(window.location));
 }};
}

export function SessionGate({children,loginRoute=false}:{children?:ReactNode;loginRoute?:boolean}){
 const [session,setSession]=useState<{username:string}|'loading'|'signed-out'|'unavailable'>('loading');
 const [busy,setBusy]=useState(false);
 const [error,setError]=useState('');
 useEffect(()=>{
  let controller:AbortController|undefined;
  const refresh=async()=>{
   controller?.abort();
   const request=new AbortController();
   controller=request;
   try{
    const response=await fetch(apiPath('/api/session'),{signal:request.signal});
    const payload=response.ok?await response.json():null;
    if(!request.signal.aborted){
     if(typeof payload?.username==='string' && payload.username) setSession({username:payload.username});
     else setSession(response.status===401?'signed-out':'unavailable');
    }
   }catch{
    if(!request.signal.aborted)setSession('unavailable');
   }
  };
  void refresh();
  window.addEventListener('focus',refresh);
  return ()=>{controller?.abort();window.removeEventListener('focus',refresh);};
 },[]);
 useEffect(()=>{
  if(loginRoute && typeof session==='object')window.location.replace(loginDestination(window.location));
  if(!loginRoute && session==='signed-out')window.location.replace(loginURL(window.location));
 },[loginRoute,session]);
 async function login({username,password}:ProAdminLoginValue){
  setBusy(true);
  setError('');
  try{
   const response=await authCommand('login',{username,password});
   if(response.status===401){setError('用户名或密码错误。');return;}
   if(!response.ok)throw new Error();
   window.location.replace(loginDestination(window.location));
  }catch{setError('登录服务暂不可用。');}
  finally{setBusy(false);}
 }
 if(loginRoute && session==='signed-out')return <ProAdminLogin appName="AIOps" busy={busy} error={error} onSubmit={login}/>;
 if(session==='signed-out' || (loginRoute && typeof session==='object'))return null;
 if(typeof session==='string')return <main><p role="status">{session==='loading'?'正在读取登录状态…':'登录服务暂不可用。'}</p></main>;
 return <UsernameContext.Provider value={session.username}>{children}</UsernameContext.Provider>;
}
