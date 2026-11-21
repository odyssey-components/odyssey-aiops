import {ReplayPairMetrics,RunMetrics} from '../src/modules/go1/views/RunMetrics';
import {test,expect} from 'bun:test';
import {renderToStaticMarkup} from 'react-dom/server';
import {go1Navigation} from '../src/components/layout/MazeShell';
import {Go1Models} from '../src/modules/go1/views/Models';
import {SimulationDetail} from '../src/modules/go1/views/SimulationDetail';
import {Go1Simulations} from '../src/modules/go1/views/Simulations';
import {Go1Recordings} from "../src/modules/go1/views/Recordings";
import type {Go1Snapshot} from '../src/modules/go1/contract';
const id='aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee';
const snapshot:Go1Snapshot={tasks:[],models:[{run:id,task:id,title:'模型 A',createdAt:'2026-11-19',manifest:{}}],baseModels:[{id, title:'基础模型',policySha256:'a',manifestSha256:'b'}],environment:null,activity:{active:null,queued:[]},simulations:[{id,title:'仿真 A',model:id,modelManifest:{},modelFiles:{},state:'succeeded',startedAt:'2026-11-19',scene:{id,title:'坡面快照',kind:'slope',format:'go1-scene/v1',createdAt:'',parameters:{angle_deg:8,length_m:1.2}},seed:7,steps:50,progress:50,logs:'',result:{outcome:'fallen'}}]};
test('simulation and visual replay test have separate navigation',()=>{
 expect(go1Navigation.map(item=>item.label)).toEqual(['场景','训练任务','仿真任务','测试用例','测试用例集','回灌测试','模型']);
});
test('open-source model list offers simulation',()=>{
 const html=renderToStaticMarkup(<Go1Models snapshot={snapshot} onSelect={()=>{}} onSimulate={()=>{}}/>);
 expect(html).toContain('创建仿真任务');expect(html).not.toContain('训练评估记录');expect(html).not.toContain('查看评估回放');
 expect(html.match(/<table\b/g)).toHaveLength(1);
});
test('simulation list distinguishes execution completion from evaluation failure',()=>{
 const html=renderToStaticMarkup(<Go1Simulations snapshot={snapshot} onSelect={()=>{}} onCreate={async()=>{}} onCancel={async()=>{}}/>);
 expect(html).toContain('回放');expect(html).toContain('执行状态');expect(html).toContain('评估结果');expect(html).toContain('执行完成');expect(html).toContain('未通过 · 跌倒');
 const detail=renderToStaticMarkup(<SimulationDetail job={snapshot.simulations![0]} pending={false} onCancel={()=>{}} onReplay={()=>{}}/>);
 expect(detail).toContain('成功条件');expect(detail).toContain('机器人整体通过坡顶');
});

test('base model list can start simulation without a training replay',()=>{
 const html=renderToStaticMarkup(<Go1Models snapshot={{...snapshot,models:[]}} onSelect={()=>{}} onSimulate={()=>{}}/>);
 expect(html).toContain('创建仿真任务');expect(html).not.toContain('训练评估记录');
});

test('removed model simulations remain clearly marked historical snapshots',()=>{
 const html=renderToStaticMarkup(<Go1Simulations snapshot={{...snapshot,models:[],baseModels:[]}} onSelect={()=>{}} onCreate={async()=>{}} onCancel={async()=>{}}/>);
 expect(html).toContain('已移除模型 · 历史快照');expect(html).toContain('回放');
 const detail=renderToStaticMarkup(<SimulationDetail modelUnavailable job={snapshot.simulations![0]} pending={false} onCancel={()=>{}} onReplay={()=>{}}/>);
 expect(detail).toContain('已移除模型 · 历史快照回放');
});

test('single and set-based vision tests appear only in the vision test list',()=>{
 const ordinary=snapshot.simulations![0];
 const data={...snapshot,simulations:[
  ordinary,
  {...ordinary,id:'bbbbbbbb-bbbb-cccc-dddd-eeeeeeeeeeee',title:'单用例回灌',recording:id,visionInjected:true},
  {...ordinary,id:'cccccccc-bbbb-cccc-dddd-eeeeeeeeeeee',title:'集合回灌',visionInjected:true,
   testCaseSet:{id,title:'集合',cases:[id],createdAt:''}},
  {...ordinary,id:'dddddddd-bbbb-cccc-dddd-eeeeeeeeeeee',title:'离线动作比较',recording:id},
 ]};
 const render=(visionInjection:boolean)=>renderToStaticMarkup(<Go1Simulations visionInjection={visionInjection}
  snapshot={data} onSelect={()=>{}} onCreate={async()=>{}} onCancel={async()=>{}}/>);
 const simulations=render(false),visionTests=render(true);
 expect(simulations).toContain('仿真 A');
 expect(simulations).not.toContain('单用例回灌');
 expect(simulations).not.toContain('集合回灌');
 expect(visionTests).toContain('单用例回灌');
 expect(visionTests).toContain('新建回灌测试');
 expect(visionTests).toContain('集合回灌');
 expect(visionTests).not.toContain('仿真 A');
 expect(visionTests).not.toContain('离线动作比较');
});

test('test case inventory excludes both single and set-based vision test jobs',()=>{
 const ordinary=snapshot.simulations![0];
 const data={...snapshot,simulations:[
  ordinary,
  {...ordinary,id:'bbbbbbbb-bbbb-cccc-dddd-eeeeeeeeeeee',title:'单用例回灌',recording:id,visionInjected:true},
  {...ordinary,id:'cccccccc-bbbb-cccc-dddd-eeeeeeeeeeee',title:'集合回灌',visionInjected:true,
   testCaseSet:{id,title:'集合',cases:[id],createdAt:''}},
 ]};
 const html=renderToStaticMarkup(<Go1Recordings snapshot={data} onCreate={()=>{}} onCancel={async()=>{}}/>);
 expect(html).toContain('仿真 A');
 expect(html).not.toContain('单用例回灌');
 expect(html).not.toContain('集合回灌');
});

test('GO1 metrics distinguish abnormal execution from a paired closed-loop regression',()=>{
 const original={...snapshot.simulations![0],result:{passed:true}};
 const replay={...original,id:'bbbbbbbb-cccc-dddd-eeee-ffffffffffff',recording:original.id,
  visionInjected:true,result:{outcome:'fallen',passed:false}};
 const summary=renderToStaticMarkup(<ReplayPairMetrics job={replay} history={[original]}/>);
 expect(summary).toContain('100.0%');
 expect(summary).toContain('1 / 1 个可判定用例');
 const openLoop=renderToStaticMarkup(<ReplayPairMetrics job={{...replay,visionInjected:false}} history={[original]}/>);
 expect(openLoop).toContain('无可比闭环判定');
 const runs=renderToStaticMarkup(<RunMetrics jobs={[original,replay,{...replay,state:'failed'},{...replay,state:'cancelled'}]}/>);
 expect(runs).toContain('33.3%');
});
