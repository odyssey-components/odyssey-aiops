import { expect, test } from 'bun:test';
import { initialState, probabilities, validateModel, type GameState } from '../core';
import { startGame, updateGame } from '../src/utilities/game';
import * as local from '../src/lessons/local_map_v1';
import * as random from '../src/lessons/random_map_v1';

function sample() {
  return {format:'odymaze-policy/v3',lesson:'local-map-v1',generator:'lcg-corridor-7/v1',
    maze:local.generateMaze(7),encoding:'grid-known-walls-goal-visited-agent-direction/v1',actions:['left','right','forward'],
    layers:[{activation:'tanh',weight:Array.from({length:64},()=>Array<number>(249).fill(0)),bias:Array<number>(64).fill(0)},
      {activation:'linear',weight:Array.from({length:3},()=>Array<number>(64).fill(0)),bias:[0,0,0]}],
    metadata:{epochs:0,seed:7,samples:0,accuracy:0,success:false}};
}

test('local memory reveals diagonal cells through walls without mutating its predecessor',()=>{
  const maze=local.generateMaze(7), state=initialState(maze);
  const blank={known:Array<string>(7).fill('???????'),visited:Array<number>(49).fill(0)};
  const memory=local.observe(maze,state,blank);
  expect(blank.known).toEqual(Array(7).fill('???????'));
  expect(blank.visited.every(n=>n===0)).toBe(true);
  expect(memory.known[2][2]).toBe(maze.rows[2][2]);
  expect(memory.known[3]).toBe('???????');
  expect(memory.visited.reduce((a,b)=>a+b,0)).toBe(1);
  const edge=local.observe(maze,{...state,x:0,y:0},blank);
  expect(edge.known[0].slice(0,3)).toBe('##?');
  expect(edge.known[1][1]).toBe('.');
});

test('249 observation uses only memory and agent state',()=>{
  const maze=local.generateMaze(7), state=initialState(maze), memory=local.initialMemory(maze,state);
  const bits=local.encode(memory,state);
  expect(bits.length).toBe(249);
  expect(bits.slice(0,49).reduce((a,b)=>a+b,0)).toBe(9);
  expect(bits.slice(98,147).every(n=>n===0)).toBe(true);
  expect(bits[147+8]).toBe(1);
  expect(bits[196+8]).toBe(1);
  expect(bits.slice(245)).toEqual([1,0,0,0]);
  const goalState={...state,x:4,y:4};
  const revealed=local.observe(maze,goalState,memory);
  expect(local.encode(revealed,goalState)[98+40]).toBe(1);
  expect(memory.known[5][5]).toBe('?');
  const source=sample();
  source.layers[0].weight[0]=Array.from({length:249},(_,i)=>(i%7)/7);
  source.layers[1].weight[2][0]=2;
  const model=validateModel(source,maze);
  expect(()=>probabilities(model,maze,state)).toThrow(/记忆/);
  const alternate=Array.from({length:100},(_,seed)=>local.generateMaze(seed)).find(candidate=>
    JSON.stringify(local.initialMemory(candidate,state))===JSON.stringify(memory) && JSON.stringify(candidate.rows)!==JSON.stringify(maze.rows))!;
  expect(alternate).toBeDefined();
  expect(probabilities(model,alternate,state,memory)).toEqual(probabilities(model,maze,state,memory));
});

test('memory expert prefers known goal over nearest frontier and never traverses unknown cells',()=>{
  const state:GameState={x:1,y:1,direction:0,steps:0,status:'ready'};
  const memory={known:['#######','#...G?#','#.????#','???????','???????','???????','???????'],visited:Array<number>(49).fill(0)};
  memory.visited[8]=1;
  expect(local.expertAction(memory,state)).toBe(2);
  memory.known[1]='#.??G?#';
  expect(()=>local.expertAction(memory,state)).toThrow();
  memory.known[1]='#.????#';
  expect(local.expertAction(memory,state)).toBe(1);
  const onGoal={...memory,known:memory.known.map((row,y)=>y===1 ? '#G????#' : row)};
  expect(()=>local.expertAction(onGoal,{...state,status:'running'})).toThrow();
  expect(()=>local.expertAction(memory,{...state,status:'timeout'})).toThrow();
  expect(()=>local.expertAction({known:Array(7).fill('???????'),visited:memory.visited},state)).toThrow();
});

test('manual steps and expert ticks update memory; reset and map/course/model changes clear it',()=>{
  let game=startGame('local-map-v1');
  const initial=game.memory;
  game=updateGame(game,{type:'step',action:2});
  expect(game.memory).not.toBe(initial);
  expect(initial).toEqual(local.initialMemory(game.maze,initialState(game.maze)));
  expect(game.memory).toEqual(local.observe(game.maze,game.state,initial!));
  game=updateGame(game,{type:'mode',mode:'expert'});
  game=updateGame(game,{type:'toggle'});
  game=updateGame(game,{type:'tick'});
  expect(game.state.steps).toBe(2);
  const modelGame=updateGame(game,{type:'loaded',model:validateModel(sample(),game.maze)});
  const modelStep=updateGame(modelGame,{type:'step'});
  expect(modelStep.memory).toEqual(local.observe(modelStep.maze,modelStep.state,modelGame.memory!));
  expect(game.memory!.visited[game.state.y*7+game.state.x]).toBe(1);
  for (const command of [{type:'reset'},{type:'maze',seed:42},{type:'loaded',model:validateModel(sample(),game.maze)}] as const) {
    const reset=updateGame(game,command);
    expect(reset.state.steps).toBe(0);
    expect(reset.memory).toEqual(local.initialMemory(reset.maze,reset.state));
  }
  expect(updateGame(game,{type:'course',lesson:'fixed-map-v1'}).memory).toBeNull();
});

test('local model validation preserves separate legacy contracts and rejects malformed data',()=>{
  expect(local.validMaze(local.generateMaze(7))).toBe(true);
  expect(random.validMaze(local.generateMaze(7))).toBe(false);
  expect(local.validMaze(random.generateMaze(7))).toBe(false);
  for (const mutate of [
    (m:ReturnType<typeof sample>)=>{m.maze.max_steps=100;},
    (m:ReturnType<typeof sample>)=>{m.generator='unknown';},
    (m:ReturnType<typeof sample>)=>{m.layers[0].weight[0].pop();},
    (m:ReturnType<typeof sample>)=>{m.layers[0].bias[0]=NaN;},
    (m:ReturnType<typeof sample>)=>{m.lesson='random-map-v1';},
    (m:ReturnType<typeof sample>)=>{m.format='odymaze-policy/v2';},
  ]) {const model=sample();mutate(model);expect(()=>validateModel(model,model.maze)).toThrow();}
});

test('memory expert reaches exits using bounded local observations',()=>{
  for(let seed=0;seed<50;seed++) {
    let game=updateGame(startGame('local-map-v1',seed),{type:'mode',mode:'expert'});
    while(game.state.status!=='success' && game.state.status!=='timeout') game=updateGame(game,{type:'step'});
    expect(game.state.status).toBe('success');
  }
});
