import type {ReactNode} from 'react';
import {useOdysseyTheme} from '@odyssey/ui/provider';
import {useEffect,useRef,useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {Alert} from '@odyssey/ui/primitives/alert';
import {Button} from '@odyssey/ui/primitives/button';
import type {Go1Scene} from '../scenes';
import {createSceneObjects} from '../scene-preview/objects';
import type {Go1Frame} from '../contract';
import {applyPose,createRobot,disposeObjects,type Go1Robot} from '../visualization/robot';
import {useRobotAssets} from '../visualization/useRobotAssets';
import {trajectorySegments} from './trajectory';
import {EYE_FOV,eyeCameraPose} from './camera';

type View={robot:Go1Robot;camera:THREE.PerspectiveCamera;eyes:THREE.PerspectiveCamera;controls:OrbitControls;render:()=>void;reset:(fit?:boolean)=>void};
export default function ReplayViewport({frame,frames,originalFrames,follow,onFit,showEyes=true,injectedVision,hasDepth,depthView,scene:taskScene}:{scene?:Go1Scene;frame:Go1Frame;frames:readonly Go1Frame[];originalFrames?:readonly Go1Frame[];showEyes?:boolean;injectedVision?:boolean;depthView?:ReactNode;hasDepth?:boolean;follow:boolean;onFit:()=>void}) {
 const {resolvedColorScheme}=useOdysseyTheme();
 const applyTheme=useRef<(dark:boolean)=>void>(()=>{});

 const {assets,error:assetError,retry}=useRobotAssets();
 const eyesVisible=useRef(showEyes);eyesVisible.current=showEyes;
 const inset=useRef<HTMLDivElement>(null);
 const sceneKey=JSON.stringify(taskScene?[taskScene.id,taskScene.kind,taskScene.parameters]:null);
 const host=useRef<HTMLDivElement>(null),view=useRef<View|null>(null);
 const [error,setError]=useState(''),[generation,setGeneration]=useState(0);
 useEffect(()=>{
  const element=host.current;if(!element||!assets)return;setError('');
  let renderer:THREE.WebGLRenderer;
  try{renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'low-power'});}catch{setError('无法创建三维视图，请检查浏览器 WebGL 支持。');return;}
  renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
  const canvas=renderer.domElement;canvas.setAttribute('aria-hidden','true');element.appendChild(canvas);
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(42,1,.01,100);
  camera.up.set(0,0,1);
  const eyes=new THREE.PerspectiveCamera(EYE_FOV,1,.01,100);
  const controls=new OrbitControls(camera,canvas);controls.minDistance=.25;controls.maxPolarAngle=Math.PI*.49;
  const bounds=new THREE.Box3();for(const pose of frames)bounds.expandByPoint(new THREE.Vector3().fromArray(pose.position_m));
  for(const pose of originalFrames??[])bounds.expandByPoint(new THREE.Vector3().fromArray(pose.position_m));
  if(taskScene){const structure=createSceneObjects(taskScene);scene.add(structure);bounds.union(new THREE.Box3().setFromObject(structure));canvas.dataset.sceneId=taskScene.id;canvas.dataset.sceneKind=taskScene.kind;canvas.dataset.sceneBoxes=String(structure.children.length);}
  const centre=bounds.getCenter(new THREE.Vector3()),radius=Math.max(1,bounds.getSize(new THREE.Vector3()).length()/2);
  camera.far=eyes.far=Math.max(100,radius*20);controls.maxDistance=Math.max(30,radius*10);
  scene.add(new THREE.HemisphereLight(0xd7eee7,0x4b4b48,2));
  const light=new THREE.DirectionalLight(0xffffff,2.5);light.position.set(2,-3,5);scene.add(light);
  const floorSize=Math.max(8,radius*4+4);
  const floor=new THREE.Mesh(new THREE.PlaneGeometry(floorSize,floorSize),new THREE.MeshStandardMaterial({color:0x152326,roughness:1}));
  floor.position.set(centre.x,centre.y,-.003);scene.add(floor);
  const grid=new THREE.GridHelper(floorSize,Math.min(100,Math.ceil(floorSize*5)),0x4e706c,0x263b3b);grid.rotation.x=Math.PI/2;grid.position.set(centre.x,centre.y,0);scene.add(grid);
  let robot:Go1Robot;
  try{robot=createRobot(assets);}catch(reason){setError(reason instanceof Error?reason.message:'Go1 模型创建失败');controls.dispose();disposeObjects(scene);renderer.dispose();renderer.forceContextLoss();canvas.remove();return;}
  scene.add(robot.root);applyPose(robot,frame);
  const eyePose=eyeCameraPose(frame);eyes.position.copy(eyePose.position);eyes.up.copy(eyePose.up);eyes.lookAt(eyePose.target);
  function addTrack(poses:readonly Go1Frame[],color:number,dashed=false){
   const segments=trajectorySegments(poses);
   if(segments.length){
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.Float32BufferAttribute(segments,3));
    const material=dashed?new THREE.LineDashedMaterial({color,dashSize:.08,gapSize:.04}):new THREE.LineBasicMaterial({color});
    const line=new THREE.LineSegments(geometry,material);
    line.computeLineDistances();
    if(dashed)line.position.z=.002;
    scene.add(line);
   }
   return segments.length/6;
  }
  if(originalFrames)canvas.dataset.originalPathSegments=String(addTrack(originalFrames,0xef4444));
  canvas.dataset.pathSegments=String(addTrack(frames,originalFrames?0x22c55e:0x72d4aa,Boolean(originalFrames)));

  let lost=false,disposed=false;
  const render=()=>{
   if(lost||disposed)return;
   renderer.setViewport(0,0,element.clientWidth,element.clientHeight);
   renderer.render(scene,camera);
   canvas.dataset.renderCalls=String(renderer.info.render.calls);
   canvas.dataset.triangles=String(renderer.info.render.triangles);
   canvas.dataset.cameraMode='orbit';
   canvas.dataset.cameraPosition=JSON.stringify(camera.position.toArray());
   canvas.dataset.cameraTarget=JSON.stringify(controls.target.toArray());
   const panel=inset.current;
   if(!eyesVisible.current||!panel?.offsetWidth){delete canvas.dataset.eyesViewport;return;}
   const width=panel.offsetWidth,height=panel.offsetHeight;
   const x=panel.offsetLeft,y=element.clientHeight-panel.offsetTop-height;
   eyes.aspect=width/height;
   eyes.updateProjectionMatrix();
   renderer.setViewport(x,y,width,height);
   renderer.setScissor(x,y,width,height);
   renderer.setScissorTest(true);
   renderer.render(scene,eyes);
   renderer.setScissorTest(false);
   canvas.dataset.eyesViewport=JSON.stringify([x,y,width,height]);
   canvas.dataset.eyesPosition=JSON.stringify(eyes.position.toArray());
   canvas.dataset.eyesTarget=JSON.stringify(eyes.position.clone().add(eyes.getWorldDirection(new THREE.Vector3())).toArray());
   canvas.dataset.eyesUp=JSON.stringify(eyes.up.toArray());
   canvas.dataset.eyesFov=String(eyes.fov);
  };
  applyTheme.current=dark=>{
   renderer.setClearColor(dark?0x111a1c:0xe8edf3);
   floor.material.color.setHex(dark?0x263438:0xcbd5e1);
   render();
  };
  const reset=(fit=false)=>{
   const target=fit?centre:robot.root.position;controls.target.copy(target);
   const distance=fit?radius*2.8:1.0;camera.position.copy(target).add(new THREE.Vector3(distance,-distance,distance*.75));controls.update();render();
  };
  const resize=()=>{if(disposed)return;const width=element.clientWidth,height=element.clientHeight;if(!width||!height)return;camera.aspect=width/height;camera.updateProjectionMatrix();renderer.setSize(width,height,false);render();};
  const observer=new ResizeObserver(resize);observer.observe(element);controls.addEventListener('change',render);
  const contextLost=(event:Event)=>{event.preventDefault();lost=true;setError('三维图形上下文已中断，请重新创建视图。');};canvas.addEventListener('webglcontextlost',contextLost);
  view.current={robot,camera,eyes,controls,render,reset};reset();resize();
  return ()=>{disposed=true;applyTheme.current=()=>{};view.current=null;observer.disconnect();canvas.removeEventListener('webglcontextlost',contextLost);controls.removeEventListener('change',render);controls.dispose();disposeObjects(scene);renderer.dispose();renderer.forceContextLoss();canvas.remove();};
 },[frames,originalFrames,generation,sceneKey,assets]);
 useEffect(()=>{
  const current=view.current;if(!current)return;applyPose(current.robot,frame);
  const pose=eyeCameraPose(frame);current.eyes.position.copy(pose.position);current.eyes.up.copy(pose.up);current.eyes.lookAt(pose.target);
  if(follow){const delta=current.robot.root.position.clone().sub(current.controls.target);current.camera.position.add(delta);current.controls.target.copy(current.robot.root.position);current.controls.update();}
  const canvas=host.current?.querySelector<HTMLCanvasElement>(':scope > canvas');if(canvas){canvas.dataset.bodyPosition=JSON.stringify(current.robot.root.position.toArray());canvas.dataset.bodyQuaternion=JSON.stringify(current.robot.root.quaternion.toArray());canvas.dataset.joints=JSON.stringify(frame.joint_position_rad);canvas.dataset.frameTime=String(frame.time);}
  current.render();
 },[frame,follow,showEyes,hasDepth,frames,originalFrames,generation,sceneKey,assets]);
 useEffect(()=>{applyTheme.current(resolvedColorScheme==='dark');},[resolvedColorScheme,frames,originalFrames,generation,sceneKey,assets]);

 return <div className="go1-viewport-wrap">
  <div ref={host} className={hasDepth?"go1-viewport go1-viewport-depth-pair":"go1-viewport"} role="img" aria-label="Go1 机器人三维回放" aria-description={showEyes?"右侧显示机器人视角":undefined}>
   <div ref={inset} className="go1-eyes-inset" hidden={!showEyes||!assets} aria-hidden="true"><span>{injectedVision?'机器人视角 · 仅参考':'机器人眼睛'}</span></div>
   {depthView}
  </div>
  {!assets&&!assetError&&<p role="status">正在加载 Go1 三维模型…</p>}{assetError&&<Alert role="alert" tone="danger">{assetError}<Button onClick={retry}>重试加载模型</Button></Alert>}
  {error&&<Alert role="alert" tone="danger">{error}<Button onClick={()=>setGeneration(value=>value+1)}>重新创建视图</Button></Alert>}
  <div className="training-actions go1-camera-controls"><Button onClick={()=>view.current?.reset()}>重置视角</Button><Button onClick={()=>{onFit();view.current?.reset(true);}}>适配轨迹</Button></div>
 </div>;
}
