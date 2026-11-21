import type {MazeSearch} from '../../utilities/search';
import {activeGo1,type Go1Snapshot} from './contract';

export function go1NeedsPolling(snapshot:Go1Snapshot|null,section:MazeSearch['go1Tab']):boolean {
 if(section!==undefined && section!=='simulation' && section!=='recording' && section!=='vision-injection')return false;
 return Boolean(snapshot?.activity.active||snapshot?.activity.queued.length||snapshot?.tasks.some(task=>activeGo1(task.state)) || snapshot?.simulations?.some(task=>activeGo1(task.state)));
}

function reuse<T>(previous:T,next:T):T {
 if(Object.is(previous,next))return previous;
 if(!previous || !next || typeof previous!=='object' || typeof next!=='object' || Array.isArray(previous)!==Array.isArray(next))return next;
 const before=previous as Record<string,unknown>,after=next as Record<string,unknown>;
 const keys=Object.keys(after);
 let unchanged=keys.length===Object.keys(before).length;
 const result:Record<string,unknown>=Array.isArray(next)?[] as unknown as Record<string,unknown>:{};
 for(const key of keys){result[key]=reuse(before[key],after[key]);if(!Object.hasOwn(before,key)||result[key]!==before[key])unchanged=false;}
 return unchanged?previous:result as T;
}
export function reuseGo1Snapshot(previous:Go1Snapshot|null,next:Go1Snapshot):Go1Snapshot {
 return previous?reuse(previous,next):next;
}
