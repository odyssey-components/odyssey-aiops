import {Quaternion,Vector3} from 'three';
import type {Go1Frame} from '../contract';
export const EYE_FOV=65;
// Same body attachment and optical axes as the go1_depth MuJoCo camera.
export function eyeCameraPose(frame:Go1Frame) {
 const [w,x,y,z]=frame.orientation_wxyz,rotation=new Quaternion(x,y,z,w).normalize();
 const position=new Vector3(.30,0,.03).applyQuaternion(rotation).add(new Vector3().fromArray(frame.position_m));
 const pitch=Math.PI/12;
 const forward=new Vector3(Math.cos(pitch),0,-Math.sin(pitch)).applyQuaternion(rotation);
 const up=new Vector3(Math.sin(pitch),0,Math.cos(pitch)).applyQuaternion(rotation);
 return {position,target:position.clone().add(forward),up};
}
