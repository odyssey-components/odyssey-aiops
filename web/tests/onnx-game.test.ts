import { expect, test } from 'bun:test';
import type { LessonId, Model } from '../core';
import { startGame, updateGame } from '../src/utilities/game';

type Game = ReturnType<typeof startGame>;
function model(lesson:LessonId='memory-map-v1'):Model {
  return {format:'odymaze-policy/v4',lesson,maze:startGame(lesson).maze,encoding:'test',actions:['left','right','forward'],layers:[],metadata:{epochs:0,seed:7,samples:0,accuracy:0,success:false}};
}
function loaded(lesson:LessonId='memory-map-v1') {
  return updateGame(startGame(lesson),{type:'loaded',model:model(lesson),external:true,mode:'manual'});
}
function finish(game:Game, values=[0,1,0], request=game.inferenceRequest!) {
  return updateGame(game,{type:'prediction',request,values,hidden:game.lesson==='memory-map-v1' ? Array(32).fill(0.25) : null});
}

test('external metadata needs no JSON parameters; async observations serialize every controller including terminal',()=>{
  let game=loaded();
  expect(game.modelBackend).toBe('onnx');expect(game.prediction).toBeNull();expect(game.recurrent).toBeNull();
  expect(updateGame(game,{type:'step',action:0})).toBe(game);
  game=finish(game);expect(game.recurrent!.observations).toBe(1);
  for(const mode of ['manual','expert','model'] as const) {
    game=updateGame(game,{type:'mode',mode});
    const before=game;
    game=updateGame(game,{type:'step',action:0});
    expect(game.state.steps).toBe(before.state.steps+1);
    if(mode==='model') expect(game.previousAction).toBe(1);
    expect(game.recurrent).toBe(before.recurrent);expect(game.prediction).toBeNull();
    expect(updateGame(game,{type:'step',action:0})).toBe(game);
    expect(updateGame(game,{type:'tick'})).toBe(game);
    expect(updateGame(game,{type:'toggle'})).toBe(game);
    game=finish(game);expect(game.recurrent!.observations).toBe(before.recurrent!.observations+1);
  }
  game={...game,state:{...game.state,steps:199}};
  const terminal=updateGame(game,{type:'step'});
  expect(terminal.state.status).toBe('timeout');expect(terminal.inferenceRequest).not.toBeNull();
  const observed=finish(terminal);
  expect(observed.recurrent!.observations).toBe(game.recurrent!.observations+1);
  expect(updateGame(observed,{type:'step'})).toBe(observed);
});

test('tick preserves running across async results without duplicate movement',()=>{
  let game=updateGame(finish(loaded()),{type:'mode',mode:'model'});
  game=updateGame(game,{type:'toggle'});
  const next=updateGame(game,{type:'tick'});
  expect(next.running).toBe(true);expect(next.state.steps).toBe(1);
  expect(updateGame(next,{type:'tick'})).toBe(next);
  const ready=finish(next);
  expect(updateGame(ready,{type:'tick'}).state.steps).toBe(2);
});

test('request identity rejects same-seed reset, replacement, course, forget and cleared-model races',async()=>{
  const pending=loaded();
  const result=Promise.resolve({type:'prediction' as const,request:pending.inferenceRequest!,values:[1,0,0],hidden:Array(32).fill(1)});
  for(const command of [{type:'reset'},{type:'maze',seed:7},{type:'loaded',model:model(),external:true},
    {type:'course',lesson:'random-map-v1'},{type:'forget'},
    {type:'configure',lesson:'memory-map-v1',seed:8,mode:'manual',speed:10},
    {type:'configure',lesson:'memory-map-v1',seed:7,mode:'manual',speed:10,clearModel:true}] as const) {
    const next=updateGame(pending,command);
    expect(next.inferenceRequest).not.toBe(pending.inferenceRequest);
    expect(updateGame(next,await result)).toBe(next);
    expect(updateGame(next,{type:'prediction-failed',request:pending.inferenceRequest!,message:'stale'})).toBe(next);
  }
  const cleared=updateGame(pending,{type:'configure',lesson:'memory-map-v1',seed:7,mode:'expert',speed:10,clearModel:true});
  expect(cleared.model).toBeNull();expect(cleared.inferenceRequest).toBeNull();expect(cleared.recurrent).toBeNull();
  expect(updateGame(cleared,{type:'toggle'}).running).toBe(true);
});

test('mode, speed, stop and same-episode configuration preserve in-flight context and loading failure can retry',()=>{
  let game=updateGame(finish(loaded()),{type:'step',action:0});
  const request=game.inferenceRequest,recurrent=game.recurrent;
  for(const command of [{type:'mode',mode:'expert'},{type:'speed',speed:15},{type:'stop'},
    {type:'configure',lesson:'memory-map-v1',seed:7,mode:'manual',speed:80},
    {type:'loading'},{type:'failed'}] as const) {
    game=updateGame(game,command);expect(game.inferenceRequest).toBe(request);expect(game.recurrent).toBe(recurrent);
  }
  game=finish(game);expect(game.recurrent!.observations).toBe(2);
  const forgotten=updateGame(game,{type:'forget'});
  expect(forgotten.state).toBe(game.state);expect(forgotten.memory).toBe(game.memory);
  expect(forgotten.visited).toBe(game.visited);expect(forgotten.log).toBe(game.log);
  expect(forgotten.previousAction).toBe(game.previousAction);expect(forgotten.recurrent).toBeNull();
  expect(finish(forgotten).recurrent!.observations).toBe(1);
});

test('invalid results fail closed and stale or duplicate completions never advance hidden state',()=>{
  for(const values of [[1,1,1],[NaN,0,1],[-1,1,1],[0,1],Object.assign(Array<number>(3),{1:1})]) {
    const pending=loaded();const failed=finish(pending,values);
    expect(failed.inferenceError).toBeTruthy();expect(failed.inferenceRequest).toBeNull();
    expect(updateGame(failed,{type:'mode',mode:'model'})).toBe(failed);
    expect(updateGame(failed,{type:'toggle'})).toBe(failed);
    expect(updateGame(failed,{type:'step',action:0})).toBe(failed);
  }
  for(const hidden of [null,Array(31).fill(0),Array(32).fill(Infinity),Array<number>(32)]) {
    const pending=loaded();
    expect(updateGame(pending,{type:'prediction',request:pending.inferenceRequest!,values:[1,0,0],hidden}).inferenceError).toBeTruthy();
  }
  const pending=loaded(),ready=finish(pending);
  expect(finish(ready,[1,0,0],pending.inferenceRequest!)).toBe(ready);
  const failed=updateGame(pending,{type:'prediction-failed',request:pending.inferenceRequest!,message:'runtime failed'});
  expect(failed.running).toBe(false);expect(failed.inferenceError).toBe('runtime failed');
  expect(updateGame(failed,{type:'reset'}).inferenceError).toBeNull();
});

test('lessons one to three use external policy predictions without recurrent state',()=>{
  for(const lesson of ['fixed-map-v1','random-map-v1','local-map-v1'] as const) {
    let game=finish(loaded(lesson));
    expect(game.recurrent).toBeNull();
    game=updateGame(game,{type:'mode',mode:'model'});
    game=updateGame(game,{type:'step'});
    expect(game.previousAction).toBe(1);expect(game.inferenceRequest).not.toBeNull();
    expect(finish(game).recurrent).toBeNull();
  }
});

test('pause takes effect while ONNX inference is pending',()=>{
  let game=updateGame(finish(loaded()),{type:'mode',mode:'model'});
  game=updateGame(game,{type:'toggle'});
  game=updateGame(game,{type:'tick'});
  expect(game.running).toBe(true);
  expect(game.inferenceRequest).not.toBeNull();
  const paused=updateGame(game,{type:'toggle'});
  expect(paused.running).toBe(false);
  expect(paused.inferenceRequest).toBe(game.inferenceRequest);
  const completed=finish(paused);
  expect(completed.running).toBe(false);
  expect(updateGame(completed,{type:'tick'})).toBe(completed);
});
