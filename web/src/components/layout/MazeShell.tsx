import { ProAdminShell } from '@odyssey/ui/patterns/admin-shell';
import type { ReactNode } from 'react';
import { useSessionAccount } from '../../access/session';
import { mazeVisible } from '../../maze-visibility';
import type { LessonId } from '../../../core';

export const go1Navigation=[{key:'go1-scenes',label:'场景'},{key:'go1-training',label:'训练任务'},{key:'go1-simulation',label:'仿真任务'},{key:'go1-recording',label:'测试用例'},{key:'go1-test-case-sets',label:'测试用例集'},{key:'go1-vision-injection',label:'回灌测试'},{key:'go1-models',label:'模型'}];
const courses = [
  {key:'fixed-map-v1',label:'第一课：固定地图'},
  {key:'random-map-v1',label:'第二课：随机地图'},
  {key:'local-map-v1',label:'第三课：局部探索'},
  {key:'memory-map-v1',label:'第四课：模型记忆'},
];

export function MazeShell({course,onCourse,children,module='maze',onModule,go1Section='training',onGo1Section}: {
  go1Section?:'scenes'|'test-case-sets'|'training'|'models'|'replay'|'simulation'|'recording'|'vision-injection';onGo1Section?:(section:'scenes'|'test-case-sets'|'training'|'models'|'replay'|'simulation'|'recording'|'vision-injection')=>void;
  module?:'maze'|'go1';onModule?:(module:'maze'|'go1')=>void;course:LessonId; onCourse:(course:LessonId)=>void; children:ReactNode;
}) {
  const account=useSessionAccount();
  function navigate(key:string) {
    if(key.startsWith('go1-')) {const section=key.slice(4);if(section==='scenes'||section==='test-case-sets'||section==='training'||section==='models'||section==='replay'||section==='simulation'||section==='recording'||section==='vision-injection')onGo1Section?.(section);return;}
    if(key==='maze' || key==='go1') {onModule?.(key);return;}
    if(key==='fixed-map-v1' || key==='random-map-v1' || key==='local-map-v1' || key==='memory-map-v1') onCourse(key);
  }
  return <ProAdminShell className="maze-shell" classNames={{content:'maze-content'}} options={{
    brand:{name:'AIOps',subtitle:'本地模型训练'},
    navigation:{items:[...(mazeVisible || module==='maze'?[{key:"maze",label:"Maze"}]:[]),{key:"go1",label:"Go1"}],activeKey:module,onChange:navigate},
    contextualNavigation:module==='maze'?courses:go1Navigation,
    contextualNavigationLabel:module==='maze'?'课时':'Go1',contextualActiveKey:module==='maze'?course:`go1-${go1Section}`,
    appearance:true,account,children,
  }} />;
}
