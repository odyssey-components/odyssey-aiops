import { product } from "../../scripts/workspace.mjs";
import {spawn,execFileSync} from 'node:child_process';
import {once} from 'node:events';
import {createInterface} from 'node:readline';
import {join} from 'node:path';

export async function startGoAPI(root,home,python=join(root,'.venv/bin/python'),source) {
  const binary=join(home,'odymaze-api');
  execFileSync('go',['build','-o',binary,'./cmd/aiops-server'],{cwd:join(product,'server'),stdio:'pipe'});
  const child=spawn(binary,['--listen','127.0.0.1:0',...(python?['--python',python]:[]),...(source?['--go1-source',source]:[])],{cwd:home,env:{...process.env,HOME:home},stdio:['ignore','ignore','pipe']});
  const closed=once(child,'close');
  const timer=setTimeout(()=>child.kill(),15000);
  let origin;
  for await(const line of createInterface({input:child.stderr})) {
    const match=line.match(/http:\/\/127\.0\.0\.1:\d+/);
    if(match){origin=match[0];break;}
  }
  clearTimeout(timer);
  if(!origin)throw new Error('Go API failed to start');
  return {origin,async close(){child.kill();await closed;}};
}
