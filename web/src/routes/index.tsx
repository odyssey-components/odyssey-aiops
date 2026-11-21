import {createFileRoute,redirect} from '@tanstack/react-router';
import {mazeVisible} from '../maze-visibility';
import {legacyPageLocation} from '../utilities/page-url';

export const Route=createFileRoute('/')({
 validateSearch:(search:Record<string,unknown>)=>search,
 beforeLoad:({search,location})=>{
  const destination=legacyPageLocation(search);
  if(!mazeVisible && destination.to.startsWith('/maze')) {
   throw redirect({to:'/go1/training',replace:true});
  }
  throw redirect({...destination,hash:location.hash,replace:true});
 },
});
