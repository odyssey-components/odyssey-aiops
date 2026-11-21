import {createFileRoute,redirect} from '@tanstack/react-router';
import {legacyPageLocation} from '../utilities/page-url';

export const Route=createFileRoute('/go1/')({
 validateSearch:(search:Record<string,unknown>)=>search,
 beforeLoad:({search,location})=>{
  throw redirect({...legacyPageLocation({...search,module:'go1'}),hash:location.hash,replace:true});
 },
});
