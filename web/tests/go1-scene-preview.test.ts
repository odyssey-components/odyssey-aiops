import {expect,test} from 'bun:test';
import {sceneGeometry,initialPose} from '../src/modules/go1/scene-preview/geometry';
import {validateMazeSearch} from '../src/utilities/search';
import type {Go1Scene} from '../src/modules/go1/scenes';
const id='12345678-1234-1234-1234-123456789abc';
function scene(kind:Go1Scene['kind'],parameters:Record<string,number>):Go1Scene{return {id,kind,parameters,format:'go1-scene/v1',title:'合成场景',createdAt:''};}
test('turn and uturn preview walls match training dimensions and orientation',()=>{
 const turn=sceneGeometry(scene('turn',{distance_m:1.2,angle_deg:90,road_width_m:.8}));
 expect(turn).toHaveLength(4);expect(turn[0].size[0]).toBeCloseTo(2.425);expect(turn[0].size.slice(1)).toEqual([.05,.4]);
 expect(turn[1].position[0]).toBeCloseTo(1.6);expect(turn[1].rotation[2]).toBeCloseTo(Math.PI/2);
 const uturn=sceneGeometry(scene('uturn',{turn_at_m:3,road_width_m:2}));
 expect(uturn).toHaveLength(3);expect(uturn[2].position).toEqual([3.6,0,.2]);
 expect(uturn[2].size[0]).toBe(2);
});
test('slope rise, plateau and initial robot state match training keyframe',()=>{
 const angle=8*Math.PI/180,length=2;
 const [ramp,plateau]=sceneGeometry(scene('slope',{angle_deg:8,length_m:length}));
 expect(ramp.rotation[1]).toBe(-angle);expect(ramp.size).toEqual([2,1.2,.05]);
 expect(plateau.position[0]).toBeCloseTo(.8+Math.cos(angle)*length+1);
 expect(plateau.size[2]).toBeCloseTo(Math.sin(angle)*length);
 expect(initialPose.position).toEqual([0,0,.3]);expect(initialPose.joints).toEqual(Array(4).fill([0,.9,-1.8]).flat());
});
test('scene detail IDs are URL state only in Go1 scenes',()=>{
 expect(validateMazeSearch({module:'go1',go1Tab:'scenes',go1Scene:id}).go1Scene).toBe(id);
 for(const raw of [{module:'go1',go1Tab:'scenes',go1Scene:'../bad'},{module:'go1',go1Tab:'models',go1Scene:id},{go1Scene:id}]) expect(validateMazeSearch(raw).go1Scene).toBeUndefined();
});

test('both turn walls cover the exposed corner, including right turns and wide roads',()=>{
 for(const angle_deg of [-135,-90,-45,45,90,135])for(const road_width_m of [.8,100]){
  const walls=sceneGeometry(scene('turn',{distance_m:1.2,angle_deg,road_width_m}));
  for(let side=0;side<2;side++){
   const approach=walls[side*2],exit=walls[side*2+1],angle=exit.rotation[2];
   const faceOffset=-Math.sign(angle)*approach.size[1]/2;
   const incomingTip=[
    approach.position[0]+approach.size[0]/2,
    approach.position[1]+faceOffset,
   ];
   const outgoingTip=[
    exit.position[0]-Math.cos(angle)*exit.size[0]/2-Math.sin(angle)*faceOffset,
    exit.position[1]-Math.sin(angle)*exit.size[0]/2+Math.cos(angle)*faceOffset,
   ];
   expect(Math.hypot(incomingTip[0]-outgoingTip[0],incomingTip[1]-outgoingTip[1])).toBeLessThan(1e-8);
   expect(approach.size[0]).toBeGreaterThan(0);expect(exit.size[0]).toBeGreaterThan(0);
  }
 }
});

test('beam preview distinguishes raised clearance from a grounded jumping obstacle',()=>{
 const [beam,...posts]=sceneGeometry(scene('duck',{distance_m:1.2,height_m:.28}));
 expect(beam.position[2]-beam.size[2]/2).toBeCloseTo(.28);
 expect(posts).toHaveLength(2);
 const jump=sceneGeometry(scene('jump',{distance_m:1.2,height_m:.2}));
 expect(jump).toHaveLength(1);
 expect(jump[0].position[2]+jump[0].size[2]/2).toBeCloseTo(.2);
});
