import {onnx} from 'odymaze-onnx-proto';

export interface TensorSummary {name:string;dtype:string;shape:(number|string)[];}
export interface WeightSummary extends TensorSummary {parameters:number;sparse:boolean;}
export interface OnnxInspection {
  irVersion:number;producer:string;producerVersion:string;domain:string;modelVersion:number;description:string;graph:string;
  opsets:[string,number][];metadata:[string,string][];inputs:TensorSummary[];outputs:TensorSummary[];
  nodes:{name:string;operator:string;inputs:string[];outputs:string[]}[];
  weights:WeightSummary[];parameters:number;operators:[string,number][];
}
function integer(value:number|{toString():string}|null|undefined) {
  const result=Number(value?.toString() ?? 0);
  if(!Number.isSafeInteger(result) || result<0) throw new Error('ONNX 数值超出可显示范围');
  return result;
}
const dtype=(value:number|null|undefined)=>onnx.TensorProto.DataType[value ?? 0] ?? `类型 ${value}`;
function tensor(value:onnx.IValueInfoProto):TensorSummary {
  const type=value.type?.tensorType;
  return {name:value.name ?? '',dtype:type?dtype(type.elemType):'非张量或未声明',
    shape:type?.shape?.dim?.map(dim=>dim.dimParam || (dim.dimValue!=null?integer(dim.dimValue):'?')) ?? []};
}
function weight(value:onnx.ITensorProto,shape=value.dims ?? [],sparse=false):WeightSummary {
  const dims=shape.map(integer);
  const parameters=dims.reduce((total,size)=>integer(total*size),1);
  return {name:value.name ?? '',dtype:dtype(value.dataType),shape:dims,parameters,sparse};
}
export function inspectOnnx(bytes:Uint8Array):OnnxInspection {
  if(!bytes.byteLength || bytes.byteLength>1024*1024) throw new Error('ONNX 文件为空或超过 1 MiB');
  let model:onnx.ModelProto;
  try {model=onnx.ModelProto.decode(bytes);} catch {throw new Error('无法解析 ONNX 文件');}
  const graph=model.graph;
  if(!graph) throw new Error('ONNX 缺少计算图');
  const bounded=<T>(items:readonly T[]|null|undefined):readonly T[]=>{
    if((items?.length ?? 0)>10000) throw new Error('ONNX 结构条目过多');return items ?? [];
  };
  const weights=[...bounded(graph.initializer).map(item=>weight(item)),
    ...bounded(graph.sparseInitializer).map(item=>weight(item.values ?? {},item.dims ?? [],true))];
  const operators=new Map<string,number>();
  const nodes=bounded(graph.node).map((node,index)=>{
    const operator=`${node.domain || 'ai.onnx'}::${node.opType || '?'}`;
    operators.set(operator,(operators.get(operator) ?? 0)+1);
    return {name:node.name || `节点 ${index+1}`,operator,inputs:[...bounded(node.input)],outputs:[...bounded(node.output)]};
  });
  return {irVersion:integer(model.irVersion),producer:model.producerName,producerVersion:model.producerVersion,
    domain:model.domain,modelVersion:integer(model.modelVersion),description:model.docString,graph:graph.name ?? '',
    opsets:bounded(model.opsetImport).map(item=>[item.domain || 'ai.onnx',integer(item.version)]),
    metadata:bounded(model.metadataProps).map(item=>[item.key ?? '',item.value ?? '']),
    inputs:bounded(graph.input).map(tensor),outputs:bounded(graph.output).map(tensor),
    nodes,weights,parameters:weights.reduce((sum,item)=>integer(sum+item.parameters),0),operators:[...operators.entries()]};
}

export async function inspectVerifiedOnnx(bytes:Uint8Array,sha256:string) {
  const digest=await crypto.subtle.digest('SHA-256',Uint8Array.from(bytes));
  const actual=Array.from(new Uint8Array(digest),value=>value.toString(16).padStart(2,'0')).join('');
  if(actual!==sha256) throw new Error('ONNX 文件校验失败，无法显示结构');
  return inspectOnnx(bytes);
}
