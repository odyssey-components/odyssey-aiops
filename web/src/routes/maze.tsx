import {createFileRoute,redirect} from '@tanstack/react-router';
import {mazeVisible} from '../maze-visibility';

export const Route=createFileRoute('/maze')({
 beforeLoad:()=>{
  if(!mazeVisible) throw redirect({to:'/go1/training',replace:true});
 },
});
