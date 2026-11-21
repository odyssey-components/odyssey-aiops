import {robotAssets} from './go1-robot-assets';
import {expect,test} from 'bun:test';
import * as THREE from 'three';
import {initialRobot} from '../src/modules/go1/scene-preview/robot';
import {initialPose} from '../src/modules/go1/scene-preview/geometry';
import {createRobot,applyPose,disposeObjects} from '../src/modules/go1/visualization/robot';
function appearance(root:THREE.Object3D){
 root.updateMatrixWorld(true);
 const meshes:unknown[]=[];
 root.traverse(object=>{if(object instanceof THREE.Mesh){
  const material=object.material as THREE.MeshStandardMaterial,position=object.geometry.getAttribute('position');
  object.geometry.computeBoundingBox();
  meshes.push({name:object.name,type:object.geometry.type,vertices:position.count,bounds:[object.geometry.boundingBox?.min.toArray(),object.geometry.boundingBox?.max.toArray()],color:material.color.getHex(),roughness:material.roughness,matrix:object.matrixWorld.toArray()});
 }});
 return meshes;
}
test('scene initial robot and replay at initial pose share dimensions, appearance and world transforms',async()=>{
 const assets=await robotAssets(),preview=initialRobot(assets),replay=createRobot(assets);
 applyPose(replay,{time:0,position_m:[...initialPose.position],orientation_wxyz:[1,0,0,0],joint_position_rad:[...initialPose.joints],episode_end:false});
 expect(appearance(preview)).toEqual(appearance(replay.root));
 const original=appearance(preview);
 applyPose(replay,{time:1,position_m:[2,1,.4],orientation_wxyz:[Math.SQRT1_2,0,0,Math.SQRT1_2],joint_position_rad:Array(12).fill(.2),episode_end:false});
 expect(appearance(preview)).toEqual(original);expect(appearance(replay.root)).not.toEqual(original);
 let previewDisposals=0;preview.traverse(object=>{if(object instanceof THREE.Mesh)object.geometry.addEventListener('dispose',()=>previewDisposals++);});
 disposeObjects(replay.root);expect(previewDisposals).toBe(0);disposeObjects(preview);expect(previewDisposals).toBeGreaterThan(0);
});

test('robot uses original detailed trunk/limb meshes instead of schematic boxes',async()=>{
 const robot=createRobot(await robotAssets());let detailed=0;
 robot.root.traverse(object=>{if(object instanceof THREE.Mesh&&object.geometry.getAttribute('position').count>1000)detailed++;});
 expect(detailed).toBe(13);disposeObjects(robot.root);
});

test('MJCF left/right thigh meshes and rear hip visual transforms preserve joint frames',async()=>{
 const robot=createRobot(await robotAssets());
 expect(robot.legs.map(leg=>leg.thigh.children[0].name)).toEqual(['thigh_mirror','thigh','thigh_mirror','thigh']);
 expect(robot.legs[2].hip.children[0].quaternion.toArray()).toEqual([0,0,-1,0]);
 expect(robot.legs[3].hip.children[0].quaternion.toArray()).toEqual([0,1,0,0]);
 expect(robot.legs.map(leg=>leg.hip.quaternion.toArray())).toEqual(Array(4).fill([0,0,0,1]));
 expect(robot.root.children[0].name).toBe('trunk');disposeObjects(robot.root);
});
