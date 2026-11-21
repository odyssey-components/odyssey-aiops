import {useOdysseyTheme} from '@odyssey/ui/provider';
import {useEffect,useRef,useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls.js';
import {Button} from '@odyssey/ui/primitives/button';
import {Alert} from '@odyssey/ui/primitives/alert';
import type {Go1Scene} from '../scenes';
import {sceneGeometry} from './geometry';
import {createSceneObjects} from './objects';
import {useRobotAssets} from '../visualization/useRobotAssets';
import {initialRobot} from './robot';
export default function SceneViewport({scene}:{scene:Go1Scene}){
 const {resolvedColorScheme}=useOdysseyTheme();
 const applyTheme=useRef<(dark:boolean)=>void>(()=>{});

 const {assets,error:assetError,retry}=useRobotAssets();
 const host=useRef<HTMLDivElement>(null),reset=useRef<()=>void>(()=>{}),[error,setError]=useState('');
 // Snapshot polling replaces object identity; only a geometry change recreates WebGL.
 const geometryKey=JSON.stringify([scene.kind,scene.parameters]);
 useEffect(()=>{
  if(!assets)return;
  const container=host.current!;setError('');
  let renderer:THREE.WebGLRenderer|undefined,controls:OrbitControls|undefined,observer:ResizeObserver|undefined;
  const world=new THREE.Scene();
  const background=new THREE.Color();world.background=background;
  let disposed=false;
  const render=()=>{if(!disposed&&renderer){renderer.render(world,camera);renderer.domElement.dataset.triangles=String(renderer.info.render.triangles);}};
  const camera=new THREE.PerspectiveCamera(45,1,.01,1000);camera.up.set(0,0,1);
  const lost=(event:Event)=>{event.preventDefault();setError('三维显示上下文已丢失，请重新打开场景查看');};
  try{
   renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));
   renderer.domElement.setAttribute('aria-label','Go1 场景三维结构');renderer.domElement.setAttribute('role','img');renderer.domElement.style.touchAction='none';
   renderer.domElement.addEventListener('webglcontextlost',lost);container.appendChild(renderer.domElement);
   world.add(new THREE.HemisphereLight(0xffffff,0x64748b,2.5));
   const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(3,-4,6);world.add(light);
   const content=new THREE.Group();world.add(content);
   content.add(createSceneObjects(scene));
   content.add(initialRobot(assets));
   const bounds=new THREE.Box3().setFromObject(content),center=bounds.getCenter(new THREE.Vector3()),size=bounds.getSize(new THREE.Vector3());
   const span=Math.max(2,size.length());camera.far=Math.max(100,span*20);camera.near=Math.min(.01,span/1000);
   const floor=new THREE.Mesh(new THREE.PlaneGeometry(Math.max(4,size.x+2),Math.max(3,size.y+2)),new THREE.MeshStandardMaterial({color:0xcbd5e1}));floor.position.set(center.x,center.y,-.028);world.add(floor);
   const origin=new THREE.AxesHelper(.45);origin.position.z=.01;world.add(origin);
   controls=new OrbitControls(camera,renderer.domElement);controls.maxDistance=span*10;controls.minDistance=.1;
   applyTheme.current=dark=>{
    background.setHex(dark?0x111a1c:0xe8edf3);
    floor.material.color.setHex(dark?0x263438:0xcbd5e1);
    render();
   };
   const resize=()=>{
    if(disposed)return;
    const width=container.clientWidth,height=container.clientHeight;if(!width||!height)return;
    renderer!.setSize(width,height);camera.aspect=width/height;camera.updateProjectionMatrix();render();
   };
   reset.current=()=>{
    const distance=span/(2*Math.tan(THREE.MathUtils.degToRad(camera.fov/2)))*Math.max(1,1/camera.aspect)*1.15;
    controls!.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(-.65,-1,.85).normalize().multiplyScalar(distance));controls!.update();render();
   };
   controls.addEventListener('change',render);observer=new ResizeObserver(resize);observer.observe(container);resize();reset.current();
   container.dataset.ready='true';container.dataset.boxCount=String(sceneGeometry(scene).length);container.dataset.robotPosition='0,0,0.3';
  }catch(cause){setError(`无法加载三维场景：${cause instanceof Error?cause.message:String(cause)}`);}
  return ()=>{
   disposed=true;applyTheme.current=()=>{};reset.current=()=>{};observer?.disconnect();controls?.dispose();
   const geometries=new Set<THREE.BufferGeometry>(),materials=new Set<THREE.Material>();
   world.traverse(object=>{if(object instanceof THREE.Mesh || object instanceof THREE.LineSegments){geometries.add(object.geometry);for(const material of Array.isArray(object.material)?object.material:[object.material])materials.add(material);}});
   geometries.forEach(value=>value.dispose());materials.forEach(value=>value.dispose());
   if(renderer){renderer.domElement.removeEventListener('webglcontextlost',lost);renderer.dispose();renderer.domElement.remove();renderer.forceContextLoss();}
   delete container.dataset.ready;
  };
 },[geometryKey,assets]);
 useEffect(()=>{applyTheme.current(resolvedColorScheme==='dark');},[resolvedColorScheme,geometryKey,assets]);

 return <div className="go1-scene-preview">{!assets&&!assetError&&<p role="status">正在加载 Go1 三维模型…</p>}{assetError&&<Alert role="alert" tone="danger">{assetError}<Button size="sm" onClick={retry}>重试加载模型</Button></Alert>}{error&&<Alert role="alert" tone="danger">{error}</Alert>}<div className="go1-scene-preview-surface"><div ref={host} className="go1-scene-canvas" data-testid="go1-scene-viewport"/><div className="go1-scene-viewport-actions"><Button size="sm" disabled={!assets||Boolean(error)} onClick={()=>reset.current()}>重置视角</Button></div></div></div>;
}
