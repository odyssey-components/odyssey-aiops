import type {Go1Scene} from '../scenes';
type Vec3=[number,number,number];
export interface SceneBox {name:string;position:Vec3;size:Vec3;rotation:Vec3;}
// Matches vision/scene.py and the pinned upstream Go1 home keyframe. Units: metres, z-up.
export const initialPose={position:[0,0,.30] as Vec3,joints:Array.from({length:4},()=>[0,.9,-1.8]).flat()};
export function sceneGeometry(scene:Go1Scene):SceneBox[]{
 if(scene.kind==='flat')return [];
 const p=scene.parameters,boxes:SceneBox[]=[];
 const box=(name:string,position:Vec3,size:Vec3,rotation:Vec3=[0,0,0])=>boxes.push({name,position,size,rotation});
 const wallThickness=.05;
 const wall=(name:string,x:number,y:number,length:number,angle=0)=>box(name,[x,y,.2],[length,wallThickness,.4],[0,0,angle]);
 if(scene.kind==='duck'||scene.kind==='jump'){
  const duck=scene.kind==='duck',height=p.height_m,distance=p.distance_m;
  box('横梁',[distance,0,duck?height+.04:height/2],[.16,1.2,duck?.08:height]);
  if(duck)for(const side of [-1,1])box('立柱',[distance,side*.56,height/2],[.16,.08,height]);
 }else if(scene.kind==='slope'){
  const angle=p.angle_deg*Math.PI/180,length=p.length_m,rise=Math.sin(angle)*length;
  box('坡面',[.8+Math.cos(angle)*length/2,0,rise/2],[length,1.2,.05],[0,-angle,0]);
  box('坡顶平台',[.8+Math.cos(angle)*length+1,0,rise/2],[2,1.2,rise]);
 }else{
  const distance=scene.kind==='turn'?p.distance_m:p.turn_at_m,half=p.road_width_m/2;
  if(scene.kind==='uturn'){
   for(const side of [-1,1])wall(`侧墙 ${side}`,(distance-.2)/2,side*half,distance+1);
   wall('端墙',distance+.6,0,p.road_width_m,Math.PI/2);
  }else{
   const angle=p.angle_deg*Math.PI/180,bend=half*Math.tan(angle/2);
   // Extend to the exposed face intersection so the box ends do not leave a notch.
   const overlap=wallThickness/2*Math.abs(Math.tan(angle/2));
   const entrance=Math.min(-.8,distance-Math.abs(bend)-.3),end=Math.max(1.3,Math.abs(bend)+.3);
   for(const side of [-1,1]){
    // Intersect each incoming boundary with its outgoing boundary.
    const corner=distance-side*bend;
    const approach=corner+overlap-entrance;
    const start=side*bend-overlap,exit=end-start,offset=(start+end)/2;
    wall(`入口墙 ${side}`,entrance+approach/2,side*half,approach);
    wall(`出口墙 ${side}`,distance+Math.cos(angle)*offset-Math.sin(angle)*side*half,Math.sin(angle)*offset+Math.cos(angle)*side*half,exit,angle);
   }
  }
 }
 return boxes;
}
