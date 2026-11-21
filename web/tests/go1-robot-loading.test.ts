import {test,expect} from 'bun:test';
import {createHash} from 'node:crypto';
import {createRobotAssetLoader,robotMeshNames} from '../src/modules/go1/visualization/assets';
import {robotAssets} from './go1-robot-assets';
const hashes={trunk:'051a57995dc6f892fc9bc53cd94e9a894f3dc209fd646e94d55d764150013970',hip:'b6599b9dd2751ebd87449203223f5683a6c235268b76fcc443e00e0d974a06c0',thigh:'bbf7c0a441323088c59276dbec8a32e708a1c61f42d30a6f6775176366ea23f8',thigh_mirror:'1a90b89de47a57cb12a2d703f6b008a6ace8a1890a381c27723296fd0d00b9d8',calf:'ad3685e132454fd64e4ca3e5089f0d812fb53958e95c14913a22e3193e9e9ed8'};
test('shipped STL bytes match the pinned Go1 source assets',async()=>{
 const assets=await robotAssets();
 for(const name of robotMeshNames)expect(createHash('sha256').update(new Uint8Array(assets[name])).digest('hex')).toBe(hashes[name]);
});
test('two viewports share one immutable byte load, not disposable geometry',async()=>{
 const assets=await robotAssets();let requests=0;
 const load=createRobotAssetLoader((async input=>{requests++;const name=String(input).split('/').pop()!.replace('.stl','') as keyof typeof assets;return new Response(assets[name]);}) as typeof fetch);
 const first=load(),second=load();expect(first).toBe(second);
 const [a,b]=await Promise.all([first,second]);expect(a).toBe(b);expect(requests).toBe(5);
 await load();expect(requests).toBe(5);
});
test('failed asset requests can retry and invalid binary/NaN input is rejected',async()=>{
 const assets=await robotAssets();let fail=true;
 const load=createRobotAssetLoader((async input=>{const name=String(input).split('/').pop()!.replace('.stl','') as keyof typeof assets;return fail?new Response('',{status:404}):new Response(assets[name]);}) as typeof fetch);
 await expect(load()).rejects.toThrow('加载失败');fail=false;expect(Object.keys(await load())).toHaveLength(5);
 const nan=assets.trunk.slice(0);new DataView(nan).setFloat32(96,NaN,true);
 for(const [bad,message] of [[new ArrayBuffer(10),'不完整'],[new ArrayBuffer(100),'无效'],[nan,'坐标无效']] as const){
  const invalid=createRobotAssetLoader((async()=>new Response(bad)) as typeof fetch);await expect(invalid()).rejects.toThrow(message);
 }
});
