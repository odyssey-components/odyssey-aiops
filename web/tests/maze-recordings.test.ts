import {expect,test} from 'bun:test';
import {startGame,updateGame} from '../src/utilities/game';
import {compareRecording,recordingFrames,simulateMaze,type MazeRecording} from '../src/utilities/recordings';
import type {OnnxRuntimeModel} from '../src/utilities/onnx-runtime';
import {validateMazeSearch} from '../src/utilities/search';

test('recording keeps accepted actions beyond the live log and resets the episode identity',()=>{
 const initial=startGame();
 let game=initial;
 for(let index=0;index<12;index++)game=updateGame(game,{type:'step',action:1});
 expect(game.log).toHaveLength(8);
 expect(game.actions).toEqual(Array(12).fill(1));
 const record:MazeRecording={format:'maze-recording/v1',id:'test',createdAt:new Date().toISOString(),lesson:game.lesson,seed:game.seed,maze:game.maze,actions:game.actions};
 expect(recordingFrames(record).at(-1)?.state).toEqual(game.state);
 const reset=updateGame(game,{type:'reset'});
 expect(reset.actions).toEqual([]);
 expect(reset.episode).not.toBe(initial.episode);
 expect(validateMazeSearch({tab:'recordings'}).tab).toBe('recordings');
});

test('re-injection uses the original trajectory and advances candidate GRU state in order',async()=>{
 const game=startGame('memory-map-v1');
 const record:MazeRecording={format:'maze-recording/v1',id:'test',createdAt:new Date().toISOString(),lesson:game.lesson,seed:game.seed,maze:game.maze,actions:[1,2,0]};
 const observations:number[][]=[],hiddenInputs:number[][]=[];
 const candidate={
  policy:{lesson:game.lesson},predict:async(input:number[],hidden:number[])=>{
   observations.push(input);hiddenInputs.push(hidden);
   return {values:[1,0,0],hidden:Array(32).fill(hiddenInputs.length)};
  },
 } as unknown as OnnxRuntimeModel;
 const report=await compareRecording(record,candidate);
 expect(report.map(row=>row.action)).toEqual([0,0,0]);
 expect(report.map(row=>row.observation)).toEqual(observations);
 expect(report.map(row=>row.hidden)).toEqual(hiddenInputs);
 expect(report[0].values).toEqual([1,0,0]);
 expect(hiddenInputs.map(value=>value[0])).toEqual([0,1,2]);
 expect(observations[1]).not.toEqual(observations[0]);
 expect(recordingFrames(record).map(frame=>frame.previousAction)).toEqual([-1,1,2,0]);
});

test('closed-loop simulation advances by model actions until the maze ends',async()=>{
 const game=startGame();
 const candidate={policy:{lesson:game.lesson},predict:async()=>({values:[1,0,0],hidden:null})} as unknown as OnnxRuntimeModel;
 const result=await simulateMaze(game.lesson,game.maze,candidate);
 expect(result.actions).toEqual(Array(game.maze.max_steps).fill(0));
 expect(result.state).toMatchObject({x:game.state.x,y:game.state.y,status:'timeout'});
});
