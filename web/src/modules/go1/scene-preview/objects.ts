import * as THREE from 'three';
import type {Go1Scene} from '../scenes';
import {sceneGeometry} from './geometry';
/** Shared static scene geometry for detail previews and saved simulation replay. */
export function createSceneObjects(scene:Go1Scene){
 const group=new THREE.Group();group.name=scene.title;
 const material=new THREE.MeshStandardMaterial({color:scene.kind==='slope'?0x967958:0x64748b});
 for(const box of sceneGeometry(scene)){
  const mesh=new THREE.Mesh(new THREE.BoxGeometry(...box.size),material);
  mesh.name=box.name;mesh.position.fromArray(box.position);mesh.rotation.set(...box.rotation);group.add(mesh);
 }
 return group;
}
