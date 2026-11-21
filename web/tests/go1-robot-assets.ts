import {readFile} from 'node:fs/promises';
import {robotMeshNames,type RobotAssets} from '../src/modules/go1/visualization/assets';
// Test the actual shipped visual assets, with synthetic poses and no policy/model inference.
export async function robotAssets():Promise<RobotAssets>{
 return Object.fromEntries(await Promise.all(robotMeshNames.map(async name=>{
  const bytes=await readFile(new URL(`../public/models/go1/assets/${name}.stl`,import.meta.url));
  return [name,bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)] as const;
 }))) as RobotAssets;
}
