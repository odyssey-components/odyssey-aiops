import type {PageInfo} from '../../utilities/list-pagination';
import type {Go1Scene} from './scenes';
export type Go1Device='cpu'|'cuda'|'rocm';
export const go1Devices=[{value:'cpu',label:'CPU'},{value:'cuda',label:'NVIDIA GPU · CUDA'},{value:'rocm',label:'AMD GPU · ROCm'}];
export type Go1State='queued'|'running'|'cancelling'|'finalizing'|'succeeded'|'failed'|'cancelled';
export const activeGo1=(state:Go1State)=>['queued','running','cancelling','finalizing'].includes(state);
export const go1States:Record<Go1State,string>={queued:'排队中',running:'运行中',cancelling:'正在取消',finalizing:'正在保存',succeeded:'已完成',failed:'失败',cancelled:'已取消'};
export interface Go1Config {iterations:number;rollout_steps:number;seed:number;target_speed_m_s:number;learning_rate:number;eval_seed:number;eval_steps:number;checkpoint_interval:number;}
export const go1Defaults:Go1Config={iterations:20,rollout_steps:64,seed:42,target_speed_m_s:.3,learning_rate:.0003,eval_seed:10001,eval_steps:100,checkpoint_interval:10};
export const validGo1Id=(id:unknown):id is string=>typeof id==='string' && /^[a-f0-9]{8}(-[a-f0-9]{4}){3}-[a-f0-9]{12}$/.test(id);
export function parseGo1Config(value:unknown):Go1Config {
  if(!value || typeof value!=='object' || Array.isArray(value))throw new Error('Go1 参数无效');
  const raw=value as Record<string,unknown>;
  if(Object.keys(raw).some(key=>!Object.hasOwn(go1Defaults,key)))throw new Error('未知训练参数');
  const config={...go1Defaults,...raw} as Go1Config;
  for(const [key,low,high] of [['iterations',1,100000],['rollout_steps',8,4096],['seed',0,2147483647],['eval_seed',0,2147483647],['eval_steps',2,500],['checkpoint_interval',1,100000]] as const) {
    if(!Number.isInteger(config[key]) || config[key]<low || config[key]>high)throw new Error(`参数 ${key} 超出范围`);
  }
  if(!Number.isFinite(config.target_speed_m_s) || Math.abs(config.target_speed_m_s)>2 || !Number.isFinite(config.learning_rate) || config.learning_rate<=0 || config.learning_rate>.01 || config.seed===config.eval_seed || config.iterations*config.rollout_steps>1000000)throw new Error('学习率、速度、独立评估种子或总步数（最多100万）无效');
  return config;
}
export type Go1Command={action:'train';title:string;config:Go1Config;scene?:string;base?:string}|{action:'cancel';id:string};
export function parseGo1Command(value:unknown):Go1Command {
  if(!value || typeof value!=='object' || Array.isArray(value))throw new Error('参数无效');
  const data=value as Record<string,unknown>;
  if(data.action==='cancel' && validGo1Id(data.id) && Object.keys(data).every(k=>['action','id'].includes(k)))return {action:'cancel',id:data.id};
  if(data.action!=='train' || Object.keys(data).some(k=>!['action','title','config','scene','base'].includes(k)) || typeof data.title!=='string' || !data.title.trim() || data.title.length>80)throw new Error('请填写任务名称和有效训练参数');
  if(data.base!==undefined && !validGo1Id(data.base))throw Error('基础模型引用无效');
  if(data.scene!==undefined && !validGo1Id(data.scene))throw Error('场景引用无效');
  return {action:'train',...(data.base?{base:String(data.base)}:{}),title:data.title.trim(),config:parseGo1Config(data.config),...(data.scene?{scene:data.scene as string}:{})};
}
export interface Go1Metric {iteration:number;timesteps:number;total_timesteps:number;mean_reward:number;loss:number;mean_height_m:number;episode_ends:number;elapsed_seconds:number;}
export interface Go1Base {id:string;title:string;policySha256:string;manifestSha256:string;}
export interface Go1Job {base?:Go1Base;device?:Go1Device;format:'go1-local-job/v1';id:string;title:string;state:Go1State;startedAt:string;finishedAt?:string;config:Go1Config;scene?:Go1Scene;logs:string;metrics:Go1Metric[];error?:string;run?:string;}
export interface Go1Environment {device?:Go1Device;ready:boolean;errors:string[];devices:string[];source_commit:string;}
export interface Go1Model {scene?:Go1Scene;run:string;task:string;title:string;createdAt:string;manifest:Record<string,unknown>;}
export interface Go1SceneSet {id:string;title:string;scenes:string[];createdAt:string;}
export interface Go1TestCaseSet {id:string;title:string;cases:string[];createdAt:string;}
export interface Go1RunMetrics {completed:number;failed:number;cancelled:number;sampleCount:number;p95:number|null;}
export interface Go1Snapshot {runMetrics?:Go1RunMetrics;pagination?:PageInfo;taskDetail?:Go1Job|null;simulationDetails?:Go1Simulation[];caseOptions?:Go1Simulation[];selectedModel?:Go1Model;selectedBaseModel?:Go1Base;testCaseSets?:Go1TestCaseSet[];baseModels?:Go1Base[];simulations?:Go1Simulation[];scenes?:Go1Scene[];tasks:Go1Job[];models:Go1Model[];environment:Go1Environment|null;activity:{active:{id:string;module:string}|null;queued:{id:string;module:string}[]};}
export function go1Metric(value:unknown):Go1Metric|null {
  if(!value || typeof value!=='object')return null;const item=value as Go1Metric;
  const names=['iteration','timesteps','total_timesteps','mean_reward','loss','mean_height_m','episode_ends','elapsed_seconds'] as const;
  if(names.some(name=>typeof item[name]!=='number' || !Number.isFinite(item[name])) || !Number.isInteger(item.iteration) || item.iteration<1 || item.timesteps<1 || item.total_timesteps<item.timesteps || item.elapsed_seconds<0)return null;
  return Object.fromEntries(names.map(name=>[name,item[name]])) as unknown as Go1Metric;
}

export interface Go1Frame {time:number;position_m:number[];orientation_wxyz:number[];joint_position_rad:number[];episode_end:boolean;}

export interface Go1Simulation {

 id:string;title:string;state:Go1State;startedAt:string;finishedAt?:string;model:string;modelManifest:Record<string,unknown>;modelFiles:Record<string,string>;
 visionInjected?:boolean;testCaseSet?:Go1TestCaseSet;recording?:string;recorded?:boolean;sceneSet?:Go1SceneSet;cases?:Go1Scene[];
 scene:Go1Scene;seed:number;steps:number;progress:number;logs:string;error?:string;result?:Record<string,unknown>;
}
export type Go1SimulationCommand = {action:'simulate';title:string;model:string;seed:number;steps:number} & ({scene:string;sceneSet?:never}|{sceneSet:string;scene?:never}) | {action:'regress'|'inject-vision';title:string;model:string;recording:string} | {action:'inject-vision';title:string;model:string;testCaseSet:string};
export interface Go1RegressionStep {step:number;time_seconds:number;baseline_action:number[];candidate_action:number[];action_delta:number[];max_abs_delta:number;candidate_state?:number[];has_depth?:boolean;}
export interface Go1RegressionDepth {width:number;height:number;f32le:string;}
