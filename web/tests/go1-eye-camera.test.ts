import {expect,test} from 'bun:test';
import type {Go1Frame} from '../src/modules/go1/contract';
import {eyeCameraPose} from '../src/modules/go1/replay/camera';
const frame=(orientation_wxyz=[1,0,0,0]):Go1Frame=>({time:0,position_m:[10,20,.5],orientation_wxyz,joint_position_rad:Array(12).fill(0),episode_end:false});
test('robot eyes use the body-mounted camera offset and 15-degree downward optical axis',()=>{
 const pose=eyeCameraPose(frame());
 expect(pose.position.toArray()).toEqual([10.3,20,.53]);
 const direction=pose.target.clone().sub(pose.position);
 expect(direction.x).toBeCloseTo(Math.cos(Math.PI/12));expect(direction.y).toBeCloseTo(0);expect(direction.z).toBeCloseTo(-Math.sin(Math.PI/12));
 expect(pose.up.dot(direction)).toBeCloseTo(0);
});
test('camera position, direction and up rotate with recorded yaw and roll, not world-up',()=>{
 const yaw=eyeCameraPose(frame([Math.SQRT1_2,0,0,Math.SQRT1_2]));
 expect(yaw.position.x).toBeCloseTo(10);expect(yaw.position.y).toBeCloseTo(20.3);
 expect(yaw.target.clone().sub(yaw.position).y).toBeCloseTo(Math.cos(Math.PI/12));
 const roll=eyeCameraPose(frame([Math.SQRT1_2,Math.SQRT1_2,0,0]));
 expect(roll.position.y).toBeCloseTo(19.97);expect(roll.position.z).toBeCloseTo(.5);
 expect(roll.up.y).toBeCloseTo(-Math.cos(Math.PI/12));expect(roll.up.z).toBeCloseTo(0);
});
