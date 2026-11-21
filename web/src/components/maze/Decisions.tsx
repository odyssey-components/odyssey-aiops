import {useEffect,useRef} from 'react';
import {Progress} from '@odyssey/ui/primitives/progress';
import {EmptyState,EmptyStateTitle} from '@odyssey/ui/primitives/empty-state';
import {Badge} from '@odyssey/ui/primitives/badge';
import {Card} from '@odyssey/ui/primitives/card';
import {ACTIONS,HEADINGS,bestAction,encode,type GameState,type Maze,type LessonId} from '../../../core';
import {encode as randomEncode} from '../../lessons/random_map_v1';
import {encode as localEncode,type Memory} from '../../lessons/local_map_v1';
import {encode as memoryEncode} from '../../lessons/memory_map_v1';
import type {Mode} from '../../utilities/game';

export function Decisions({state,mode,values,maze,course,memory,previousAction,pending,predictionIdentity}: {
  state:GameState;mode:Mode;values:number[]|null;maze:Maze;course:LessonId;memory:Memory|null;previousAction:number;pending:boolean;predictionIdentity?:object|null;
}) {
  const observation=course==='memory-map-v1' ? memoryEncode(maze,state,previousAction)
    : course==='local-map-v1' && memory ? localEncode(memory,state)
    : course==='random-map-v1' ? randomEncode(maze,state) : encode(maze,state);
  const previous=useRef<{values:number[];maze:Maze;identity:object|null|undefined;steps:number}|null>(null);
  useEffect(()=>{
    if(values) previous.current={values,maze,identity:predictionIdentity,steps:state.steps};
    else if(!pending || previous.current?.maze!==maze || previous.current?.identity!==predictionIdentity || state.steps<=previous.current.steps) previous.current=null;
  },[values,maze,predictionIdentity,state.steps,pending]);
  const cached=previous.current;
  const displayed=values ?? (pending && cached?.maze===maze && cached.identity===predictionIdentity && cached.steps<state.steps ? cached.values : null);
  const stale=!values && !!displayed;
  const winner=displayed?bestAction(displayed):-1;
  return <Card className="card decision-card">
    <div className="card-heading"><h2>动作预测</h2><Badge id="decision-mode">{{manual:'手动',expert:'专家',model:'模型'}[mode]}</Badge></div>
    <div className="input-label"><span>输入</span><span>{observation.length} 维</span></div>
    <div className="input-state" id="input-state">x = {state.x}　y = {state.y}　{HEADINGS[state.direction]}</div>
    <div id="encoding" className={`encoding ${course==='fixed-map-v1'?'':'grid-encoding'}`} aria-label="观测编码">
      {observation.map((bit,index)=><span key={index} className={`bit ${bit?'on':''}`}>{bit}</span>)}
    </div>
    <div className="input-label">动作概率</div>
    <div id="probabilities" data-values={values?.join(",")} data-stale={stale} aria-busy={pending}>{displayed?displayed.map((value,index)=><div key={ACTIONS[index]} className={`probability-row ${winner===index?'selected':''}`}>
      <span>{ACTIONS[index]}</span><Progress value={value*100} showValue={false} aria-label={`${ACTIONS[index]}概率`} /><strong>{(value*100).toFixed(1)}%</strong>
    </div>):<EmptyState><EmptyStateTitle>{pending?'推理中…':'暂无预测'}</EmptyStateTitle></EmptyState>}</div>
  </Card>;
}
