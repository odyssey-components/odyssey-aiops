import type {Go1Frame} from '../contract';
import {samplePose} from './trajectory';

function distance(a: readonly number[], b: readonly number[]) {
 return Math.hypot(a[0]-b[0],a[1]-b[1]);
}

export function summarizeTrajectory(frames:readonly Go1Frame[]) {
 let length=0,roll=0,pitch=0;
 for(let i=0;i<frames.length;i++){
  const frame=frames[i];
  if(i>0&&!frames[i-1].episode_end)length+=distance(frames[i-1].position_m,frame.position_m);
  const [w,x,y,z]=frame.orientation_wxyz;
  const norm=Math.hypot(w,x,y,z);
  const sinPitch=2*(w*y-z*x)/(norm*norm);
  roll=Math.max(roll,Math.abs(Math.atan2(2*(w*x+y*z),norm*norm-2*(x*x+y*y)))*180/Math.PI);
  pitch=Math.max(pitch,Math.abs(Math.asin(Math.max(-1,Math.min(1,sinPitch))))*180/Math.PI);
 }
 return {length,roll,pitch};
}

export function compareTrajectories(original:readonly Go1Frame[],candidate:readonly Go1Frame[]) {
 const start=Math.max(original[0].time,candidate[0].time);
 const end=Math.min(original.at(-1)!.time,candidate.at(-1)!.time);
 const endpoint=distance(original.at(-1)!.position_m,candidate.at(-1)!.position_m);
 if(end<start)return {endpoint,overlap:undefined};
 const times=[...new Set([start,end,...original.map(f=>f.time),...candidate.map(f=>f.time)])]
  .filter(time=>time>=start&&time<=end).sort((a,b)=>a-b);
 let sum=0,peak=0;
 for(const time of times){
  const error=distance(samplePose(original,time).position_m,samplePose(candidate,time).position_m);
  peak=Math.max(peak,error);
  sum+=error;
 }
 return {endpoint,overlap:{start,end,mean:sum/times.length,peak}};
}
