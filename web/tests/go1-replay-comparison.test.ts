import {expect,test} from 'bun:test';
import type {Go1Frame} from '../src/modules/go1/contract';
import {compareTrajectories,summarizeTrajectory} from '../src/modules/go1/replay/comparison';

function pose(time:number,x:number,y=0,roll=0):Go1Frame {
 return {time,position_m:[x,y,.3],orientation_wxyz:[Math.cos(roll/2),Math.sin(roll/2),0,0],joint_position_rad:Array(12).fill(0),episode_end:false};
}

test('paired error uses only the overlap and endpoint difference uses both final frames',()=>{
 const result=compareTrajectories([pose(0,0),pose(1,1),pose(3,10)],[pose(.5,.5,1),pose(1,1,1)]);
 expect(result.endpoint).toBeCloseTo(Math.sqrt(82));
 expect(result.overlap).toEqual({start:.5,end:1,mean:1,peak:1});
 expect(compareTrajectories([pose(0,0)],[pose(1,1)]).overlap).toBeUndefined();
});

test('path length skips episode resets and tilt is reported in degrees',()=>{
 const frames=[pose(0,0),{...pose(1,3,4,Math.PI/6),episode_end:true},pose(2,100),pose(3,101)];
 const metrics=summarizeTrajectory(frames);
 expect(metrics.length).toBe(6);
 expect(metrics.roll).toBeCloseTo(30);
 expect(metrics.pitch).toBe(0);
});
