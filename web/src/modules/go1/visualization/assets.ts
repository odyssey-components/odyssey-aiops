export const robotMeshNames=['trunk','hip','thigh','thigh_mirror','calf'] as const;
export type RobotMeshName=typeof robotMeshNames[number];
export type RobotAssets=Readonly<Record<RobotMeshName,ArrayBuffer>>;
const base=import.meta.env?.BASE_URL ?? '/';
export function createRobotAssetLoader(fetcher:typeof fetch=fetch){
 let pending:Promise<RobotAssets>|undefined;
 return function load():Promise<RobotAssets>{
  // Cache immutable source bytes, never disposable Three geometry or mutable robot instances.
  if(!pending)pending=Promise.all(robotMeshNames.map(async name=>{
   const response=await fetcher(`${base}models/go1/assets/${name}.stl`);
   if(!response.ok)throw Error(`Go1 网格 ${name} 加载失败（${response.status}）`);
   const bytes=await response.arrayBuffer();
   if(bytes.byteLength<84)throw Error(`Go1 网格 ${name} 文件不完整`);
   const data=new DataView(bytes),triangles=data.getUint32(80,true);
   if(!triangles||bytes.byteLength!==84+triangles*50)throw Error(`Go1 网格 ${name} 数据无效`);
   for(let triangle=0;triangle<triangles;triangle++)for(let value=0;value<12;value++){
    if(!Number.isFinite(data.getFloat32(84+triangle*50+value*4,true)))throw Error(`Go1 网格 ${name} 坐标无效`);
   }
   return [name,bytes] as const;
  })).then(entries=>Object.freeze(Object.fromEntries(entries)) as RobotAssets).catch(error=>{pending=undefined;throw error;});
  return pending;
 };
}
export const loadRobotAssets=createRobotAssetLoader();
