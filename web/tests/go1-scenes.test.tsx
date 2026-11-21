import {expect,test} from 'bun:test';
import {renderToStaticMarkup} from 'react-dom/server';
import {Go1Scenes} from '../src/modules/go1/views/Scenes';
import {sceneSuccess,validateSceneInput,type Go1Scene} from '../src/modules/go1/scenes';
import {validateMazeSearch} from '../src/utilities/search';
import {parseGo1Command} from '../src/modules/go1/contract';
const id='12345678-1234-1234-1234-123456789abc';
test('scenes restore through module navigation and training commands retain the scene ID',()=>{
 expect(validateMazeSearch({module:'go1',go1Tab:'scenes'}).go1Tab).toBe('scenes');
 expect(parseGo1Command({action:'train',title:'目标任务',scene:id,config:{}})).toMatchObject({scene:id});
 expect(()=>parseGo1Command({action:'train',title:'目标任务',scene:'../bad',config:{}})).toThrow();
});
test('scene configuration bounds and task goal wording are explicit',()=>{
 const value=validateSceneInput('转弯练习','turn',{angle_deg:90,road_width_m:.8,distance_m:1.2});
 expect(()=>validateSceneInput('bad','uturn',{road_width_m:.1,turn_at_m:1.2})).toThrow();
 const scene:Go1Scene={...value,id,format:'go1-scene/v1',createdAt:'2026-11-19T00:00:00Z'};
 const html=renderToStaticMarkup(<Go1Scenes scenes={[scene]} onTrain={()=>{}}/>);
 expect(html).toContain('转弯练习');expect(html).toContain('90°');expect(html).toContain('创建训练任务');
 expect(sceneSuccess(scene)).toContain('稳定 1 秒');
 expect(html).toContain('>查看<');
 expect(html).not.toContain('添加场景');expect(html).not.toContain('复制并调整条件');expect(html).not.toContain('<input');
 const detail=renderToStaticMarkup(<Go1Scenes scenes={[scene]} selected={id} onTrain={()=>{}}/>);
 expect(detail).toContain('Go1 场景列表');expect(detail).not.toContain('正在加载三维场景');expect(detail).not.toContain('go1-scene-viewport');
});

test('scene distances, widths and slope lengths have no fixed maximum',()=>{
 for(const [kind,parameters] of [
  ['turn',{angle_deg:90,road_width_m:100,distance_m:1000}],
  ['uturn',{road_width_m:100,turn_at_m:1000}],
  ['slope',{angle_deg:8,length_m:1000}],
 ] as const) expect(validateSceneInput('大范围场景',kind,parameters).parameters).toEqual(parameters);
 for(const value of [NaN,Infinity,-Infinity,0,-1]) expect(()=>validateSceneInput('无效','turn',{angle_deg:90,road_width_m:.8,distance_m:value})).toThrow();
});
