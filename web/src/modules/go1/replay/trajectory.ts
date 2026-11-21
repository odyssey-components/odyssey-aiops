import type {Go1Frame} from '../contract';

const vector=(value:unknown,size:number):value is number[]=>Array.isArray(value)&&value.length===size&&Array.from(value).every(n=>typeof n==='number'&&Number.isFinite(n));
export function parseFrames(value:unknown):Go1Frame[] {
 if(!Array.isArray(value)||!value.length||value.length>250000)throw Error('回放没有有效的姿态数据');
 let previous=-1;
 for(const item of value){
  if(!item||typeof item!=='object'||!Number.isFinite(item.time)||item.time<0||item.time<=previous||!vector(item.position_m,3)||!vector(item.orientation_wxyz,4)||!vector(item.joint_position_rad,12)||typeof item.episode_end!=='boolean')throw Error('回放姿态或时间戳无效');
  const norm=Math.hypot(...item.orientation_wxyz);if(!Number.isFinite(norm)||norm<.99||norm>1.01)throw Error('回放机身朝向无效');
  previous=item.time;
 }
 return value as Go1Frame[];
}
export function frameAtTime(frames:readonly Go1Frame[],time:number):number {
 let low=0,high=frames.length-1;
 while(low<high){const middle=Math.ceil((low+high)/2);if(frames[middle].time<=time)low=middle;else high=middle-1;}
 return low;
}
function rotation(a:readonly number[],b:readonly number[],fraction:number):number[] {
 let dot=a.reduce((sum,value,index)=>sum+value*b[index],0);
 const target=dot<0?b.map(value=>-value):b;dot=Math.min(1,Math.max(-1,Math.abs(dot)));
 let result:number[];
 if(dot>.9995)result=a.map((value,index)=>value+(target[index]-value)*fraction);
 else {const angle=Math.acos(dot),denominator=Math.sin(angle);result=a.map((value,index)=>(Math.sin((1-fraction)*angle)*value+Math.sin(fraction*angle)*target[index])/denominator);}
 const norm=Math.hypot(...result);return result.map(value=>value/norm);
}
export function samplePose(frames:readonly Go1Frame[],time:number):Go1Frame {
 const index=frameAtTime(frames,time),a=frames[index],b=frames[index+1];
 if(!a)throw Error('回放没有有效的姿态数据');
 if(!b||a.episode_end||time<=a.time)return a;
 const fraction=Math.min(1,(time-a.time)/(b.time-a.time));
 const blend=(left:readonly number[],right:readonly number[])=>left.map((value,i)=>value+(right[i]-value)*fraction);
 return {time,position_m:blend(a.position_m,b.position_m),orientation_wxyz:rotation(a.orientation_wxyz,b.orientation_wxyz,fraction),joint_position_rad:blend(a.joint_position_rad,b.joint_position_rad),episode_end:false};
}
export function trajectorySegments(frames:readonly Go1Frame[],limit=3000):number[] {
 const points:number[]=[],stride=Math.max(1,Math.ceil(frames.length/limit));let anchor=0;
 for(let index=1;index<frames.length;index++){
  if(frames[index-1].episode_end){anchor=index;continue;}
  if(index-anchor<stride&&index!==frames.length-1&&!frames[index].episode_end)continue;
  const a=frames[anchor].position_m,b=frames[index].position_m;
  points.push(a[0],a[1],.012,b[0],b[1],.012);anchor=index;
 }
 return points;
}
