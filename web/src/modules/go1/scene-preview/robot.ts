import type {RobotAssets} from '../visualization/assets';
import {initialPose} from './geometry';
import {createRobot,applyPose} from '../visualization/robot';

export function initialRobot(assets:RobotAssets){
 const robot=createRobot(assets);
 applyPose(robot,{time:0,position_m:initialPose.position,orientation_wxyz:[1,0,0,0],joint_position_rad:initialPose.joints,episode_end:false});
 return robot.root;
}
