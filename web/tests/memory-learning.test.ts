import { expect, test } from 'bun:test';
import { initialState, predictRecurrent, probabilities, validateModel } from '../core';
import { startGame, updateGame } from '../src/utilities/game';
import { encode, generateMaze } from '../src/lessons/memory_map_v1';

function sample() {
  const matrix=(rows:number,cols:number)=>Array.from({length:rows},()=>Array<number>(cols).fill(0));
  return {format:'odymaze-policy/v4',lesson:'memory-map-v1',generator:'lcg-corridor-7/v1',
    maze:generateMaze(7),encoding:'local-view-xy-direction-previous-action/v1',actions:['left','right','forward'],
    gru:{hidden_size:32,weight_ih:matrix(96,40),weight_hh:matrix(96,32),bias_ih:Array<number>(96).fill(0),bias_hh:Array<number>(96).fill(0)},
    layers:[{activation:'linear',weight:matrix(3,32),bias:[0,0,0]}],
    metadata:{epochs:0,seed:7,samples:0,accuracy:0,success:false}};
}

test('40 inputs include only current local view, xy, direction and actual previous action',()=>{
  const maze=generateMaze(7),state=initialState(maze),bits=encode(maze,state);
  expect(bits.length).toBe(40);
  expect(bits.slice(36)).toEqual([1,0,0,0]);
  expect(encode(maze,state,2).slice(36)).toEqual([0,0,0,1]);
  expect(bits.slice(18,25)).toEqual([0,1,0,0,0,0,0]);
  const changed={...maze,rows:maze.rows.map((row,y)=>y===5 ? row.replace('G','.') : row)};
  expect(encode(changed,state)).toEqual(bits);
  expect(encode(maze,{...state,x:0,y:0}).slice(0,3)).toEqual([1,1,1]);
  expect(()=>encode(maze,state,3)).toThrow();
});

test('GRU matches PyTorch r,z,n equations with reset applied after recurrent affine; immutable state and context',()=>{
  const source=sample();
  source.gru.bias_ih[0]=0.2;source.gru.bias_hh[0]=0.3;
  source.gru.bias_ih[32]=-0.4;source.gru.bias_hh[32]=0.1;
  source.gru.bias_ih[64]=0.6;source.gru.bias_hh[64]=0.7;
  source.gru.weight_hh[64][0]=0.8;source.gru.weight_ih[64][36]=0.9;
  source.layers[0].weight[0][0]=2;
  const model=validateModel(source,source.maze),hidden=Array<number>(32).fill(0);hidden[0]=0.25;
  const old=[...hidden],r=1/(1+Math.exp(-0.5)),z=1/(1+Math.exp(0.3));
  const expected=(1-z)*Math.tanh(1.5+r*(0.7+0.8*0.25))+z*0.25;
  const result=predictRecurrent(model,source.maze,initialState(source.maze),hidden);
  expect(result.hidden[0]).toBeCloseTo(expected,12);
  expect(result.probabilities[0]).toBeCloseTo(Math.exp(2*expected)/(Math.exp(2*expected)+2),12);
  expect(hidden).toEqual(old);
  expect(predictRecurrent(model,source.maze,initialState(source.maze),Array(32).fill(0)).probabilities).not.toEqual(result.probabilities);
  expect(predictRecurrent(model,source.maze,initialState(source.maze),hidden,0).probabilities).not.toEqual(result.probabilities);
  expect(()=>probabilities(model,source.maze,initialState(source.maze))).toThrow(/predictRecurrent/);
});

test('all controllers process one observation per action, including terminal, and controls never advance GRU',()=>{
  const source=sample();source.gru.bias_ih[64]=1;source.layers[0].weight[0][0]=1;
  const model=validateModel(source,source.maze);
  let game=updateGame(startGame('memory-map-v1'),{type:'loaded',model,mode:'manual'});
  expect(game.recurrent!.observations).toBe(1);
  for (const mode of ['manual','expert','model'] as const) {
    game=updateGame(game,{type:'mode',mode});
    const previous=game;
    game=updateGame(game,{type:'step',action:0});
    expect(game.recurrent!.observations).toBe(previous.recurrent!.observations+1);
    expect(previous.recurrent!.hidden).not.toBe(game.recurrent!.hidden);
    expect(game.previousAction).toBe(game.log.at(-1)!.action);
  }
  const recurrent=game.recurrent;
  for(const command of [{type:'mode',mode:'manual'},{type:'speed',speed:40},
    {type:'configure',lesson:'memory-map-v1',seed:7,mode:'expert',speed:90}] as const) {
    game=updateGame(game,command);expect(game.recurrent).toBe(recurrent);
  }
  game={...game,state:{...game.state,steps:199},mode:'manual'};
  const terminal=updateGame(game,{type:'step',action:0});
  expect(terminal.state.status).toBe('timeout');
  expect(terminal.recurrent!.observations).toBe(game.recurrent!.observations+1);
  expect(updateGame(terminal,{type:'step',action:0})).toBe(terminal);
  for(const command of [{type:'reset'},{type:'maze',seed:42},{type:'loaded',model},
    {type:'configure',lesson:'memory-map-v1',seed:42,mode:'manual',speed:50}] as const) {
    const reset=updateGame(game,command);
    expect(reset.recurrent!.observations).toBe(1);expect(reset.previousAction).toBe(-1);
  }
  const forgotten=updateGame(game,{type:'forget'});
  expect(forgotten.state).toBe(game.state);expect(forgotten.memory).toBe(game.memory);
  expect(forgotten.previousAction).toBe(game.previousAction);expect(forgotten.recurrent!.observations).toBe(1);
  expect(forgotten.recurrent!.values).not.toEqual(game.recurrent!.values);
  expect(forgotten.running).toBe(false);
  expect(updateGame(startGame(),{type:'forget'}).recurrent).toBeNull();
});

test('v4 rejects malformed identity, shapes and every nonnumeric or unbounded parameter',()=>{
  const mutations:((m:ReturnType<typeof sample>)=>void)[]=[
    m=>{m.format='odymaze-policy/v3';},m=>{m.lesson='local-map-v1';},m=>{m.generator='other';},
    m=>{m.encoding='grid-known-walls-goal-visited-agent-direction/v1';},m=>{m.maze.max_steps=100;},
    m=>{m.gru.hidden_size=31;},m=>{m.gru.weight_ih.pop();},m=>{m.gru.weight_hh[0].pop();},
    m=>{m.gru.bias_ih.pop();},m=>{m.gru.bias_hh[0]=Infinity;},m=>{m.layers[0].weight[0].pop();},
    m=>{m.layers.push(m.layers[0]);},m=>{m.actions.reverse();},
  ];
  for(const invalid of [NaN,Infinity,101,'1',true,null]) mutations.push(m=>{m.gru.weight_ih[0][0]=invalid as number;});
  for(const mutate of mutations){const m=sample();mutate(m);expect(()=>validateModel(m,m.maze)).toThrow();}
});
