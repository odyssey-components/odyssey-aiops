import {useLocalTablePagination} from '../../utilities/list-pagination';
import {Card} from '@odyssey/ui/primitives/card';
import {Table,TableHeader,TableBody,TableRow,TableHead,TableCell} from '@odyssey/ui/primitives/table';
import type {OnnxInspection,TensorSummary} from '../../utilities/onnx-inspection';

function Tensors({title,items}:{title:string;items:TensorSummary[]}) {
  const pagination=useLocalTablePagination(items.length);
  return <><h4>{title}</h4><Table pagination={pagination.options} aria-label={title} containerClassName="dataset-list">
    <TableHeader><TableRow><TableHead scope="col">名称</TableHead><TableHead scope="col">类型</TableHead><TableHead scope="col">形状</TableHead></TableRow></TableHeader>
    <TableBody>{pagination.slice(items).map((item,index)=><TableRow key={index}><TableHead scope="row">{item.name || '未命名'}</TableHead><TableCell>{item.dtype}</TableCell><TableCell>[{item.shape.join(', ')}]</TableCell></TableRow>)}</TableBody>
  </Table></>;
}
export function OnnxStructure({info}:{info:OnnxInspection}) {
  const operators=useLocalTablePagination(info.operators.length);
  const weights=useLocalTablePagination(info.weights.length);
  const nodes=useLocalTablePagination(info.nodes.length);
  return <Card className="card dataset-detail" id="onnx-structure">
    <h3>ONNX 文件信息</h3><dl>
      <dt>生产工具</dt><dd>{info.producer || '未记录'} {info.producerVersion}</dd>
      <dt>IR 版本</dt><dd>{info.irVersion}</dd><dt>模型版本</dt><dd>{info.modelVersion}</dd>
      <dt>Domain</dt><dd>{info.domain || '未记录'}</dd><dt>计算图</dt><dd>{info.graph || '未命名'}</dd>
      <dt>说明</dt><dd>{info.description || '未记录'}</dd>
      <dt>Opset</dt><dd>{info.opsets.map(([domain,version])=>`${domain}: ${version}`).join('\n') || '未记录'}</dd>
      <dt>主图节点数</dt><dd>{info.nodes.length}</dd><dt>权重张量数</dt><dd>{info.weights.length}</dd>
      <dt>Initializer 参数总数</dt><dd>{info.parameters.toLocaleString('zh-CN')}</dd>
    </dl>
    <Tensors title="ONNX 输入" items={info.inputs}/><Tensors title="ONNX 输出" items={info.outputs}/>
    <h4>自定义元信息</h4>{info.metadata.length?<dl>{info.metadata.map(([key,value],index)=><div key={index}><dt>{key}</dt><dd>{value}</dd></div>)}</dl>:<p>文件未记录自定义元信息</p>}
    <h4>主图算子统计</h4><Table pagination={operators.options} aria-label="主图算子统计" containerClassName="dataset-list">
      <TableHeader><TableRow><TableHead scope="col">算子</TableHead><TableHead scope="col">数量</TableHead></TableRow></TableHeader>
      <TableBody>{operators.slice(info.operators).map(([name,count])=><TableRow key={name}><TableHead scope="row">{name}</TableHead><TableCell>{count}</TableCell></TableRow>)}</TableBody>
    </Table>
    <details><summary>权重张量（{info.weights.length}）</summary><Table pagination={weights.options} aria-label="ONNX 权重张量" containerClassName="dataset-list">
      <TableHeader><TableRow>{['名称','类型','形状','参数数量','存储'].map(label=><TableHead scope="col" key={label}>{label}</TableHead>)}</TableRow></TableHeader>
      <TableBody>{weights.slice(info.weights).map((item,index)=><TableRow key={index}><TableHead scope="row">{item.name || '未命名'}</TableHead><TableCell>{item.dtype}</TableCell><TableCell>[{item.shape.join(', ')}]</TableCell><TableCell>{item.parameters.toLocaleString('zh-CN')}</TableCell><TableCell>{item.sparse?'稀疏':'稠密'}</TableCell></TableRow>)}</TableBody>
    </Table></details>
    <details><summary>主图节点与连接（{info.nodes.length}）</summary><Table pagination={nodes.options} aria-label="ONNX 节点连接" containerClassName="dataset-list">
      <TableHeader><TableRow>{['节点','算子','输入','输出'].map(label=><TableHead scope="col" key={label}>{label}</TableHead>)}</TableRow></TableHeader>
      <TableBody>{nodes.slice(info.nodes).map((item,index)=><TableRow key={index}><TableHead scope="row">{item.name}</TableHead><TableCell>{item.operator}</TableCell><TableCell>{item.inputs.join('\n')}</TableCell><TableCell>{item.outputs.join('\n')}</TableCell></TableRow>)}</TableBody>
    </Table></details>
  </Card>;
}
