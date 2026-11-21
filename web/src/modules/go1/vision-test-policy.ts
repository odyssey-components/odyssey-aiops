import type {Go1Snapshot} from './contract';

export const visionTestModelName='Go1 500万步 · 视觉';
const policySHA256='86dac1920695f035b35998ec49479b4479d74741b44ac5b115af69ceed36da62';

export function visionTestModel(snapshot:Go1Snapshot){
 return snapshot.models.filter(model=>model.manifest.format==='go1-depth-residual/v2'&&model.manifest.base_policy_sha256===policySHA256)
  .sort((a,b)=>a.createdAt===b.createdAt?a.run.localeCompare(b.run):b.createdAt.localeCompare(a.createdAt))[0];
}
