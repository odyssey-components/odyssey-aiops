import { expect, test } from 'bun:test';
import { initialState, probabilities, validateModel } from '../core';
import fixed from '../maze.json';
import { startGame, updateGame } from '../src/utilities/game';
import * as random from '../src/lessons/random_map_v1';

test('explicit unknown lesson cannot masquerade as legacy model', () => {
  expect(() => validateModel({format:'odymaze-policy/v1',lesson:'unknown',maze:fixed}, fixed)).toThrow(/课程/);
});
test('course switching owns current map and resets playback', () => {
  const next = updateGame({...startGame(), running:true}, {type:'course',lesson:'random-map-v1'});
  expect(next.lesson).toBe('random-map-v1');
  expect(next.maze.rows[5][5]).toBe('G');
  expect(next.running).toBe(false);
  const regenerated=updateGame(next,{type:'maze',seed:42});
  expect(regenerated.maze).toEqual(random.generateMaze(42));
  expect(regenerated.state.steps).toBe(0);
});
test('random channel-major observation and deterministic reachable maps', () => {
  expect(typeof random.generateMaze).toBe('function');
  const maze=random.generateMaze(7);
  const bits=random.encode(maze,initialState(maze));
  expect(bits.length).toBe(151);
  expect(bits[0]).toBe(1);
  expect(bits[49+40]).toBe(1);
  expect(bits[98+8]).toBe(1);
  expect(bits.slice(147)).toEqual([1,0,0,0]);
  expect(random.generateMaze(7)).toEqual(maze);
});

function randomModel() {
  return {
    format:'odymaze-policy/v2', lesson:'random-map-v1', generator:'lcg-corridor-7/v1',
    maze:random.generateMaze(7),encoding:'grid-walls-goal-agent-direction/v1',actions:['left','right','forward'],
    layers:[
      {activation:'tanh',weight:Array.from({length:128},()=>Array(151).fill(0)),bias:Array(128).fill(0)},
      {activation:'linear',weight:Array.from({length:3},()=>Array(128).fill(0)),bias:[0,0,0]},
    ],metadata:{epochs:0,seed:7,samples:0,accuracy:0,success:false},
  };
}
test('random model evaluates current map rather than export sample',()=>{
  const source=randomModel();
  const current=random.generateMaze(42);
  const different=source.maze.rows.flatMap((row,y)=>[...row].map((cell,x)=>cell!==current.rows[y][x] ? y*7+x : -1)).find(index=>index>=0)!;
  source.layers[0].weight[0][different]=1;
  source.layers[1].weight[2][0]=2;
  const model=validateModel(source,current);
  const actual=probabilities(model,current,initialState(current));
  const sample=probabilities(model,source.maze,initialState(source.maze));
  expect(actual).not.toEqual(sample);
  const wall=current.rows[Math.floor(different/7)][different%7]==='#' ? 1 : 0;
  const logit=2*Math.tanh(wall);
  expect(actual[2]).toBeCloseTo(Math.exp(logit)/(2+Math.exp(logit)),12);
});

test('random model sample rejects broken borders, goal, generator and dimensions',()=>{
  for(const mutate of [
    (m:ReturnType<typeof randomModel>)=>{m.maze.rows[0]='.######';},
    (m:ReturnType<typeof randomModel>)=>{m.maze.rows[5]='#######';},
    (m:ReturnType<typeof randomModel>)=>{m.generator='unknown';},
    (m:ReturnType<typeof randomModel>)=>{m.layers[0].weight[0].pop();},
    (m:ReturnType<typeof randomModel>)=>{m.lesson='fixed-map-v1';},
  ]) {const model=randomModel();mutate(model);expect(()=>validateModel(model,fixed)).toThrow();}
});

test('regeneration preserves random model but switching cannot run it in fixed course',()=>{
  const model=validateModel(randomModel(),fixed);
  let game=updateGame(startGame(),{type:'loaded',model});
  expect(game.lesson).toBe('random-map-v1');
  expect(game.mode).toBe('model');
  game=updateGame(game,{type:'maze',seed:42});
  expect(game.model).toBe(model);
  expect(game.maze).toEqual(random.generateMaze(42));
  game=updateGame(game,{type:'course',lesson:'fixed-map-v1'});
  expect(game.model).toBeNull();
  expect(updateGame(game,{type:'mode',mode:'model'}).mode).toBe('manual');
  expect(()=>probabilities(model,fixed,initialState(fixed))).toThrow();
});

test('seed domain is unsigned 32-bit and every generated maze is expert reachable',()=>{
  for (const seed of [-1,NaN,Infinity,1.5,4294967296]) expect(()=>random.generateMaze(seed)).toThrow();
  for(let seed=0;seed<20;seed++) {
    let game=updateGame(startGame(),{type:'course',lesson:'random-map-v1'});
    game=updateGame(game,{type:'maze',seed});
    game=updateGame(game,{type:'mode',mode:'expert'});
    while(game.state.status!=='success' && game.state.status!=='timeout') game=updateGame(game,{type:'step'});
    expect(game.state.status).toBe('success');
  }
});

test('independent test rate is optional and bounded; training seed retains CLI bounds',()=>{
  for(const rate of [NaN,Infinity,-.1,1.1,'1']) {
    const model={...randomModel(),metadata:{...randomModel().metadata,test_success_rate:rate}};
    expect(()=>validateModel(model,fixed)).toThrow();
  }
  const model=randomModel(); model.metadata.seed=2147483648;
  expect(()=>validateModel(model,fixed)).toThrow();
});

test('URL configuration preserves episode for controls and model across same-course maps',()=>{
  const model=validateModel(randomModel(),fixed);
  let game=updateGame(startGame(),{type:'loaded',model});
  game=updateGame(game,{type:'mode',mode:'expert'});
  game=updateGame(game,{type:'step'});
  const configured=updateGame({...game,running:true},{type:'configure',lesson:'random-map-v1',seed:7,mode:'expert',speed:40});
  expect(configured.state).toEqual(game.state);
  expect(configured.running).toBe(false);
  const mapped=updateGame(configured,{type:'configure',lesson:'random-map-v1',seed:42,mode:'model',speed:40});
  expect(mapped.model).toBe(model);
  expect(mapped.state.steps).toBe(0);
  expect(mapped.mode).toBe('model');
});

test('URL model replacement clears old runnable model before failed restore',()=>{
  const model=validateModel(randomModel(),fixed);
  const game=updateGame(startGame(),{type:'loaded',model});
  const restored=updateGame({...game,running:true},{type:'configure',lesson:'random-map-v1',seed:7,mode:'model',speed:40,clearModel:true});
  const failed=updateGame(updateGame(restored,{type:'loading'}),{type:'failed'});
  expect(failed.model).toBeNull();
  expect(failed.mode).toBe('manual');
  expect(failed.running).toBe(false);
  expect(updateGame(failed,{type:'mode',mode:'model'}).mode).toBe('manual');
});

test('failed manual model reads preserve the current valid model',()=>{
  const model=validateModel(randomModel(),fixed);
  const game=updateGame(startGame(),{type:'loaded',model});
  const failed=updateGame(updateGame(game,{type:'loading'}),{type:'failed'});
  expect(failed.model).toBe(model);
  expect(failed.mode).toBe('model');
  expect(failed.loading).toBe(false);
});
