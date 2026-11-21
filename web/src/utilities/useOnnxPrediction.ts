import {useEffect, type Dispatch, type RefObject} from 'react';
import {encode} from '../../core';
import {encode as randomEncode} from '../lessons/random_map_v1';
import {encode as localEncode} from '../lessons/local_map_v1';
import {encode as memoryEncode} from '../lessons/memory_map_v1';
import {type startGame,type updateGame} from './game';
import type {OnnxRuntimeModel} from './onnx-runtime';

type Game=ReturnType<typeof startGame>;
export function encodeGameObservation(game:Pick<Game,"lesson"|"maze"|"state"|"memory"|"previousAction">) {
 return game.lesson==='memory-map-v1' ? memoryEncode(game.maze,game.state,game.previousAction)
      : game.lesson==='local-map-v1' ? localEncode(game.memory!,game.state)
      : game.lesson==='random-map-v1' ? randomEncode(game.maze,game.state) : encode(game.maze,game.state);
}
type Command=Parameters<typeof updateGame>[1];
export function useOnnxPrediction(game:Game,runtime:RefObject<OnnxRuntimeModel|null>,dispatch:Dispatch<Command>) {
  useEffect(()=>{
    const request=game.inferenceRequest,session=runtime.current;
    if(!request || game.loading || game.modelBackend!=='onnx' || !session || session.policy!==game.model) return;
    let cancelled=false;
    const observation=encodeGameObservation(game);
    const hidden=game.lesson==='memory-map-v1' ? game.recurrent?.hidden ?? Array<number>(32).fill(0) : undefined;
    void session.predict(observation,hidden).then(result=>{
      if(!cancelled)dispatch({type:'prediction',request,values:result.values,hidden:result.hidden});
    },error=>{
      if(!cancelled)dispatch({type:'prediction-failed',request,message:error instanceof Error ? error.message : 'ONNX 推理失败，请重新加载模型'});
    });
    return ()=>{cancelled=true;};
  },[game.inferenceRequest,game.loading,game.model,runtime,dispatch]);
}
