import {robotAssets} from './go1-robot-assets';
import {expect,test} from 'bun:test';
import type {Go1Frame} from '../src/modules/go1/contract';
import {frameAtTime,parseFrames,samplePose,trajectorySegments} from '../src/modules/go1/replay/trajectory';
const pose=(time:number,x:number,end=false):Go1Frame=>({time,position_m:[x,0,.3],orientation_wxyz:[1,0,0,0],joint_position_rad:Array(12).fill(0),episode_end:end});
test('time lookup and interpolation use recorded timestamps and stop at reset boundaries',()=>{
 const frames=[pose(.02,0),pose(.04,2,true),pose(.06,20)];
 expect(frameAtTime(frames,0)).toBe(0);expect(frameAtTime(frames,1)).toBe(2);
 expect(samplePose(frames,.03).position_m[0]).toBeCloseTo(1);
 expect(samplePose(frames,.05).position_m[0]).toBe(2);
 expect(samplePose(frames,.06).position_m[0]).toBe(20);
 expect(trajectorySegments(frames)).toEqual([0,0,.012,2,0,.012]);
});
test('quaternion sampling follows the shortest rotation and accepts antipodal representations',()=>{
 const a=pose(0,0),b=pose(1,0);b.orientation_wxyz=[-1,0,0,0];
 expect(samplePose([a,b],.5).orientation_wxyz).toEqual([1,0,0,0]);
 b.orientation_wxyz=[0,0,0,1];const result=samplePose([a,b],.5);
 expect(result.orientation_wxyz[0]).toBeCloseTo(Math.SQRT1_2);expect(result.orientation_wxyz[3]).toBeCloseTo(Math.SQRT1_2);
 expect(samplePose([a],8)).toEqual(a);
});
test('bad or unordered replay data fails before reaching WebGL',()=>{
 for(const frames of [[],[pose(1,0),pose(0,0)],[{...pose(0,0),joint_position_rad:[0]}],[{...pose(0,0),orientation_wxyz:[0,0,0,0]}],[pose(0,NaN)]])expect(()=>parseFrames(frames)).toThrow();
 expect(parseFrames([pose(0,0)])).toHaveLength(1);
});

test('robot applies z-up body position, wxyz orientation and all twelve joint axes',async()=>{
 const {createRobot,applyPose,disposeObjects}=await import('../src/modules/go1/visualization/robot');
 const robot=createRobot(await robotAssets()),frame=pose(0,3);frame.orientation_wxyz=[Math.SQRT1_2,0,0,Math.SQRT1_2];frame.joint_position_rad=Array.from({length:12},(_,i)=>i/10);
 applyPose(robot,frame);expect(robot.root.position.toArray()).toEqual([3,0,.3]);expect(robot.root.quaternion.z).toBeCloseTo(Math.SQRT1_2);
 expect(robot.legs.map(leg=>[leg.hip.rotation.x,leg.thigh.rotation.y,leg.calf.rotation.y]).flat()).toEqual(frame.joint_position_rad);
 expect(robot.legs.map(leg=>leg.hip.name)).toEqual(['FR','FL','RR','RL']);disposeObjects(robot.root);
});

test('downsampling never bridges a reset hidden inside skipped frames',()=>{
 const frames=Array.from({length:12},(_,i)=>pose(i,i<5?i:100+i,i===4));
 expect(trajectorySegments(frames,1)).toEqual([0,0,.012,4,0,.012,105,0,.012,111,0,.012]);
});
