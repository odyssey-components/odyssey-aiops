import {Dialog} from '@odyssey/ui/primitives/dialog';
import type {Go1Model} from '../contract';
import {Go1Replay} from './Replay';

export function EvaluationReplayDialog({model,onClose}:{model?:Go1Model;onClose:()=>void}) {
 if(!model)return null;
 return <Dialog open onOpenChange={open=>{if(!open)onClose();}} className="go1-simulation-replay-dialog"
  title={`训练评估记录 · ${model.title}`}
  content={<Go1Replay key={model.run} embedded models={[model]} selected={model.run} onSelect={()=>{}}/>} cancel={{label:'关闭回放'}}/>;
}
