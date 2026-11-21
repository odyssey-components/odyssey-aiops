import {expect,test} from 'bun:test';
import {sceneCriteria,sceneSuccess,validateSceneInput} from '../src/modules/go1/scenes';
import {go1Evaluation,go1SimulationEvaluation} from '../src/modules/go1/outcomes';
import {go1Defaults} from '../src/modules/go1/contract';
test('criteria validate physical progress and custom stability, preserving legacy defaults',()=>{
 const parameters={angle_deg:90,road_width_m:.8,distance_m:1.2};
 const input=validateSceneInput('turn','turn',parameters,{exit_distance_m:2,heading_tolerance_deg:5,stable_seconds:2});
 expect(input.criteria.exit_distance_m).toBe(2);
 expect(sceneCriteria({kind:'turn'}).stable_seconds).toBe(1);
 expect(sceneSuccess({...input,id:'fixture',format:'go1-scene/v1',createdAt:''})).toContain('稳定 2 秒');
 expect(()=>validateSceneInput('bad','turn',parameters,{exit_distance_m:0,heading_tolerance_deg:5,stable_seconds:2})).toThrow();
 expect(()=>validateSceneInput('bad','uturn',{road_width_m:.8,turn_at_m:1.2},{return_tolerance_m:1,heading_tolerance_deg:5,stable_seconds:2})).toThrow();
 expect(go1Defaults.target_speed_m_s).toBe(.3);
});
test('execution completion cannot be mistaken for passing a scene',()=>{
 expect(go1Evaluation({outcome:'fallen'})).toBe('未通过 · 跌倒');
 expect(go1Evaluation({outcome:'success'})).toBe('通过');
 expect(go1Evaluation({successes:0,total:1})).toBe('未通过 · 0/1');
 expect(go1Evaluation({episode_ends:7})).toBe('未通过 · 跌倒');
 expect(go1Evaluation({})).toBe('未评估');
});

test('available diagnostic data and missing run results have explicit evaluation labels',()=>{
 expect(go1Evaluation({format:'go1-command-evaluation/v2',steps:150,capability_verified:false})).toBe('已记录诊断 · 150 步，能力未验证');
 expect(go1Evaluation({outcome:'observed',steps:500,passed:false})).toBe('诊断已录制 · 500 步');
 expect(go1SimulationEvaluation('succeeded',{outcome:'collision'})).toBe('未通过 · 碰撞');
 expect(go1SimulationEvaluation('failed',null)).toBe('执行失败 · 无评估结果');
 expect(go1SimulationEvaluation('running',null)).toBe('待评估');
});
