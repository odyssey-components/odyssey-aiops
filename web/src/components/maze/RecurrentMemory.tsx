import {Card} from '@odyssey/ui/primitives/card';
import {Button} from '@odyssey/ui/primitives/button';

export function RecurrentMemory({state,disabled,onForget}: {
  state:{hidden:number[];observations:number}|null;disabled:boolean;onForget:()=>void;
}) {
  return <Card className="card recurrent-card">
    <div className="card-heading"><h2>模型的记忆 · GRU</h2><span className="tiny-label">32 个隐藏数值</span></div>
    <div id="gru-hidden" className="hidden-state" aria-label="GRU隐藏状态">
      {state?.hidden.map((value,index)=><span key={index} className="hidden-unit" data-value={value}
        title={`隐藏单元 ${index+1}：${value.toFixed(6)}`}>
        <span className="hidden-bar" style={{height:`${Math.abs(value)*100}%`}} />
        <span>{value.toFixed(2)}</span>
      </span>)}
    </div>
    <p id="gru-observations" role="status">{state?`已处理 ${state.observations} 次观察`:'加载本课模型后显示隐藏状态。'}</p>
    <Button id="forget-memory" size="sm" disabled={disabled || !state} onClick={onForget}>清空模型记忆</Button>
  </Card>;
}
