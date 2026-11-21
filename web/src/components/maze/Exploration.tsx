import type { GameState } from '../../../core';
import type { Memory } from '../../lessons/local_map_v1';

export function Exploration({memory,state}: {memory:Memory;state:GameState}) {
  const known=memory.known.join('').split('').filter(cell=>cell!=='?').length;
  const visited=memory.visited.reduce((sum,value)=>sum+value,0);
  const discovered=memory.known.some(row=>row.includes('G'));
  return <section className="exploration" aria-label="探索记录">
    <div><h3>当前视野 · 3 × 3</h3><div id="local-view" className="local-view" aria-label="角色周围的九个格子">
      {Array.from({length:9},(_,index)=>{
        const x=state.x+index%3-1,y=state.y+Math.floor(index/3)-1;
        const cell=memory.known[y]?.[x] ?? '#';
        return <span key={index} data-cell={cell} className={`sensor-cell ${cell==='#'?'sensor-wall':''}`}>{index===4?'我':cell==='#'?'墙':cell==='G'?'出口':cell==='?'?'?':'路'}</span>;
      })}
    </div></div>
    <div className="exploration-summary" role="status">
      <p id="explored-count">已观察 {known} / 49 格</p>
      <p id="visited-count">已访问 {visited} 格</p>
      <p id="goal-discovered">{discovered?'已发现出口':'出口尚未发现'}</p>
    </div>
  </section>;
}
