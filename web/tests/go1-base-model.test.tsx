import {test,expect} from 'bun:test';
import {renderToStaticMarkup} from 'react-dom/server';
import {go1Defaults,parseGo1Command} from '../src/modules/go1/contract';
import {Go1Tasks} from '../src/modules/go1/views/Tasks';
const id='00000000-0000-4000-8000-000000000001';
test('base model accepts server ID and rejects path reference',()=>{
 expect(parseGo1Command({action:'train',title:'warm',config:go1Defaults,base:id})).toMatchObject({base:id});
 expect(()=>parseGo1Command({action:'train',title:'bad',config:go1Defaults,base:'../../policy'})).toThrow();
});
test('task list stays mounted when a task is selected',()=>{
 const base={id,title:'Upstream walk',policySha256:'a'.repeat(64),manifestSha256:'b'.repeat(64)};
 const snapshot={tasks:[{id,title:'warm',base,config:go1Defaults,format:'go1-local-job/v1' as const,state:'succeeded' as const,startedAt:'2026-11-19T00:00:00Z',logs:'',metrics:[]}],models:[],environment:null,activity:{active:null,queued:[]}};
 const html=renderToStaticMarkup(<Go1Tasks snapshot={snapshot} selected={id} onSelect={()=>{}} onSubmit={async()=>{}}/>);
 expect(html).toContain('Go1 训练任务列表');expect(html).toContain('warm');
});
