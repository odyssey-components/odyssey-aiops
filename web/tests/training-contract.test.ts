import {expect,test} from 'bun:test';
import {parseTrainingCommand} from '../src/utilities/training-contract';

test('training accepts only bounded lesson commands, never argv or paths',()=>{
 const good={action:'prepare',course:'random-map-v1',seed:7,maps:16};
 expect(parseTrainingCommand(good)).toEqual(good);
 for(const bad of [{...good,course:'future'},{...good,maps:513},{...good,seed:true},{...good,output:'/tmp/elsewhere'},{...good,title:'x'.repeat(81)},{...good,description:{}},
  {action:'train',course:'random-map-v1',seed:7,epochs:1,dataset:'../file'},
  {action:'train',course:'random-map-v1',seed:7,epochs:1001,dataset:'00000000-0000-0000-0000-000000000000'}]) {
  expect(()=>parseTrainingCommand(bad)).toThrow();
 }
});
