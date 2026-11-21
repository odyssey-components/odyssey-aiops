import {expect,test} from 'bun:test';
import {parseGo1Command,parseGo1Config,go1Metric} from '../src/modules/go1/contract';
test('local Go1 only accepts bounded fixed parameter commands',()=>{
 expect(parseGo1Config({}).iterations).toBe(20);
 for(const config of [{seed:true},{iterations:0},{iterations:100000,rollout_steps:4096},{command:'x'},{seed:42,eval_seed:42},{learning_rate:NaN}])expect(()=>parseGo1Config(config)).toThrow();
 expect(()=>parseGo1Command({action:'train',title:'test',config:{},source:'/tmp'})).toThrow();
 expect(()=>parseGo1Command({action:'cancel',id:'../other'})).toThrow();
 expect(go1Metric({iteration:1,mean_reward:NaN})).toBeNull();
});

test('Go1 training commands leave device selection to the server',()=>{
 for(const device of ['cpu','cuda','rocm'] as const)expect(()=>parseGo1Command({action:'train',title:'device test',device,config:{}})).toThrow();
 expect(()=>parseGo1Command({action:'train',title:'bad',device:'auto',config:{}})).toThrow();
});
