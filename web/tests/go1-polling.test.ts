import {expect,test} from 'bun:test';
import {go1NeedsPolling,reuseGo1Snapshot} from '../src/modules/go1/snapshot';
import type {Go1Snapshot} from '../src/modules/go1/contract';
const snapshot:Go1Snapshot={tasks:[],models:[],scenes:[],environment:null,activity:{active:null,queued:[]}};
test('idle and static pages do not poll; active training/simulation retains updates',()=>{
 expect(go1NeedsPolling(snapshot,undefined)).toBe(false);
 const live={...snapshot,tasks:[{state:'running'} as Go1Snapshot['tasks'][number]]};
 expect(go1NeedsPolling(live,undefined)).toBe(true);
 expect(go1NeedsPolling(live,'scenes')).toBe(false);
 expect(go1NeedsPolling(live,'models')).toBe(false);
 expect(go1NeedsPolling(live,'replay')).toBe(false);
 expect(go1NeedsPolling({...snapshot,simulations:[{state:'queued'} as NonNullable<Go1Snapshot['simulations']>[number]]},'simulation')).toBe(true);
});
test('unchanged polling data keeps identity including scene when task status changes',()=>{
 const next=structuredClone(snapshot);
 expect(reuseGo1Snapshot(snapshot,next)).toBe(snapshot);
 next.activity.active={id:'job',module:'go1'};
 const updated=reuseGo1Snapshot(snapshot,next);
 expect(updated).not.toBe(snapshot);
 expect(updated.scenes).toBe(snapshot.scenes);
 expect(updated.models).toBe(snapshot.models);
 expect(updated.activity.active?.id).toBe('job');
});
