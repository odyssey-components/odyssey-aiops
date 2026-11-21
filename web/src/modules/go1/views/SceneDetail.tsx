import {Button} from '@odyssey/ui/primitives/button';
import {sceneKinds,sceneSummary,sceneSuccess,type Go1Scene} from '../scenes';
import '../scene-preview/scene-preview.css';
export function Go1SceneDetail({scene,onTrain}:{scene:Go1Scene;onTrain:()=>void}){
 return <section className="training-workspace go1-scene-detail" aria-label="Go1 场景详情">
 <div className="dataset-heading"><h2>{scene.title}</h2><div className="training-actions">{scene.kind!=='flat'&&<Button size="sm" variant="solid" onClick={onTrain}>创建训练任务</Button>}</div></div>
 <dl className="go1-scene-facts"><div><dt>目标</dt><dd>{sceneKinds[scene.kind]}</dd></div><div className="go1-scene-parameters"><dt>参数</dt><dd>{sceneSummary(scene)}</dd></div><div className="go1-scene-success"><dt>成功条件</dt><dd>{sceneSuccess(scene)}</dd></div><div><dt>初始位置（m）</dt><dd>0 / 0 / 0.30</dd></div><div><dt>初始朝向</dt><dd>+X</dd></div></dl>
 </section>;
}
