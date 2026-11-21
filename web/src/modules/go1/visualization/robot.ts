import * as THREE from 'three';
import {STLLoader} from 'three/addons/loaders/STLLoader.js';
import {robotMeshNames,type RobotAssets,type RobotMeshName} from './assets';
import type {Go1Frame} from '../contract';

type Leg={hip:THREE.Group;thigh:THREE.Group;calf:THREE.Group};
export interface Go1Robot {root:THREE.Group;legs:Leg[];}

// Original Go1 STL vertices in metres; transforms follow pinned go1_torque.xml.
export function createRobot(assets:RobotAssets):Go1Robot {
 const root=new THREE.Group();root.name='go1';
 const loader=new STLLoader(),geometry={} as Record<RobotMeshName,THREE.BufferGeometry>;
 try{
  for(const name of robotMeshNames){
   const mesh=loader.parse(assets[name]);geometry[name]=mesh;mesh.computeBoundingBox();
   if(!mesh.boundingBox||![...mesh.boundingBox.min.toArray(),...mesh.boundingBox.max.toArray()].every(Number.isFinite))throw Error(`Go1 网格 ${name} 坐标无效`);
  }
 }catch(error){Object.values(geometry).forEach(value=>value.dispose());throw error;}
 const dark=new THREE.MeshStandardMaterial({color:new THREE.Color(.1,.1,.1),roughness:.65});
 function mesh(parent:THREE.Group,name:RobotMeshName){const object=new THREE.Mesh(geometry[name],dark);object.name=name;parent.add(object);return object;}
 mesh(root,'trunk');
 const footGeometry=new THREE.SphereGeometry(.023,16,12),legs:Leg[]=[];
 for(const [index,[x,side]] of [[.1881,-1],[.1881,1],[-.1881,-1],[-.1881,1]].entries()){
  const hip=new THREE.Group();hip.name=['FR','FL','RR','RL'][index];hip.position.set(x,side*.04675,0);root.add(hip);
  const visual=mesh(hip,'hip');
  // Visual-only rotations must not rotate the joint frame or descendant leg links.
  if(index===2)visual.quaternion.set(0,0,-1,0); // MJCF wxyz: 0 0 0 -1
  if(index===3)visual.quaternion.set(0,1,0,0);  // MJCF wxyz: 0 0 1 0
  const thigh=new THREE.Group();thigh.position.y=side*.08;hip.add(thigh);
  mesh(thigh,side<0?'thigh_mirror':'thigh');
  const calf=new THREE.Group();calf.position.z=-.213;thigh.add(calf);mesh(calf,'calf');
  const foot=new THREE.Mesh(footGeometry,dark);foot.name='foot';foot.position.z=-.213;calf.add(foot);
  legs.push({hip,thigh,calf});
 }
 root.userData.model='unitree-go1-stl';
 return {root,legs};
}
export function applyPose(robot:Go1Robot,frame:Go1Frame):void {
 robot.root.position.fromArray(frame.position_m);
 const [w,x,y,z]=frame.orientation_wxyz;robot.root.quaternion.set(x,y,z,w).normalize();
 robot.legs.forEach((leg,index)=>{leg.hip.rotation.x=frame.joint_position_rad[index*3];leg.thigh.rotation.y=frame.joint_position_rad[index*3+1];leg.calf.rotation.y=frame.joint_position_rad[index*3+2];});
 robot.root.updateMatrixWorld(true);
}
export function disposeObjects(root:THREE.Object3D):void {
 const geometries=new Set<THREE.BufferGeometry>(),materials=new Set<THREE.Material>();
 root.traverse(object=>{if(object instanceof THREE.Mesh||object instanceof THREE.Line){geometries.add(object.geometry);for(const material of Array.isArray(object.material)?object.material:[object.material])materials.add(material);}});
 geometries.forEach(value=>value.dispose());materials.forEach(value=>value.dispose());
}
