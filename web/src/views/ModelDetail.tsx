import {useEffect,useState} from 'react';
import {Alert} from '@odyssey/ui/primitives/alert';
import {Button} from '@odyssey/ui/primitives/button';
import {Card} from '@odyssey/ui/primitives/card';
import {OnnxStructure} from '../components/models/OnnxStructure';
import type {OnnxInspection} from '../utilities/onnx-inspection';
import type {LessonId} from '../../core';
import {fetchModelArtifact} from '../apis/models';
import {validateOnnxManifest,type OnnxManifest} from '../utilities/onnx-contract';

export function ModelDetail({course,run,onBack}:{course:LessonId;run:string;onBack:()=>void}) {
  const [manifest,setManifest]=useState<OnnxManifest|null>(null),[error,setError]=useState(''),[retry,setRetry]=useState(0);
  const [inspection,setInspection]=useState<OnnxInspection|null>(null);
  useEffect(()=>{
    let disposed=false;setManifest(null);setInspection(null);setError('');
    void (async()=>{
      try {
        const bytes=await fetchModelArtifact(run,'manifest',64*1024);
        const value=validateOnnxManifest(JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes)));
        if(value.lesson!==course) throw new Error('模型不属于当前课时');
        if(disposed) return;
        setManifest(value);
        const binary=await fetchModelArtifact(run,'onnx',1024*1024);
        const {inspectVerifiedOnnx}=await import('../utilities/onnx-inspection');
        const structure=await inspectVerifiedOnnx(binary,value.sha256);
        if(!disposed) setInspection(structure);
      } catch(error) {if(!disposed) setError(error instanceof Error?error.message:'无法读取模型详情');}
    })();
    return ()=>{disposed=true;};
  },[course,run,retry]);
  return <section className="training-workspace" aria-label="模型详情">
    <div className="dataset-heading"><h2>模型详情</h2><Button onClick={onBack}>返回模型列表</Button></div>
    {error && <Alert tone="danger" role="alert">{error}<Button size="sm" onClick={()=>setRetry(value=>value+1)}>重试</Button></Alert>}
    {!error && !manifest && <p role="status">正在读取模型详情…</p>}
    {manifest && <Card className="card dataset-detail"><h3>{run}/model.onnx</h3><dl>
      <dt>课时</dt><dd>{manifest.lesson}</dd>
      {manifest.dataset && <><dt>地图组 ID</dt><dd>{manifest.dataset.id}</dd><dt>地图组 SHA-256</dt><dd>{manifest.dataset.sha256}</dd></>}
      <dt>模型格式</dt><dd>{manifest.policy_format}</dd>
      <dt>清单格式</dt><dd>{manifest.schema}</dd>
      <dt>ONNX Opset</dt><dd>{manifest.opset}</dd>
      <dt>输入编码</dt><dd>{manifest.encoding}</dd>
      <dt>输入张量</dt><dd>{manifest.inputs.map(item=>`${item.name} · ${item.dtype} · [${item.shape.join(', ')}]`).join('\n')}</dd>
      <dt>输出张量</dt><dd>{manifest.outputs.map(item=>`${item.name} · ${item.dtype} · [${item.shape.join(', ')}]`).join('\n')}</dd>
      <dt>动作顺序</dt><dd>{manifest.actions.join(' / ')}</dd>
      <dt>训练轮数</dt><dd>{manifest.metadata.epochs}</dd>
      <dt>随机种子</dt><dd>{manifest.metadata.seed}</dd>
      <dt>训练样本数</dt><dd>{manifest.metadata.samples}</dd>
      <dt>训练动作正确率</dt><dd>{(manifest.metadata.accuracy*100).toFixed(2)}%</dd>
      <dt>独立测试通关率</dt><dd>{manifest.metadata.test_success_rate===undefined?'未记录':`${(manifest.metadata.test_success_rate*100).toFixed(2)}%`}</dd>
      {manifest.generator && <><dt>地图生成器</dt><dd>{manifest.generator}</dd></>}
      {manifest.recurrent && <><dt>循环记忆</dt><dd>{manifest.recurrent.initial_hidden.length} 维 · 每张地图重置 · 每步更新一次</dd></>}
      <dt>SHA-256</dt><dd>{manifest.sha256}</dd>
    </dl></Card>}
    {manifest && !inspection && !error && <p role="status">正在读取 ONNX 文件结构…</p>}
    {inspection && <OnnxStructure info={inspection}/>}
  </section>;
}
