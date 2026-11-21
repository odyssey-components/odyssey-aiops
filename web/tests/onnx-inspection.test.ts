import {expect,test} from 'bun:test';
import {onnx} from '../node_modules/onnxruntime-web/lib/onnxjs/ort-schema/protobuf/onnx';
import {inspectOnnx} from '../src/utilities/onnx-inspection';
const bytes=()=>onnx.ModelProto.encode(onnx.ModelProto.create({irVersion:8,producerName:'synthetic',producerVersion:'1',opsetImport:[{domain:'',version:17}],metadataProps:[{key:'note',value:'hello'}],graph:{name:'test',input:[{name:'x',type:{tensorType:{elemType:1,shape:{dim:[{dimParam:'batch'},{dimValue:2}]}}}}],output:[{name:'y'}],node:[{name:'dense',opType:'Gemm',input:['x','weight'],output:['y']}],initializer:[{name:'weight',dataType:1,dims:[2,3],floatData:[1,2,3,4,5,6]}]}})).finish();
test('reads actual protobuf metadata, tensors, connections and parameter counts',()=>{
 const info=inspectOnnx(bytes());expect(info.producer).toBe('synthetic');expect(info.parameters).toBe(6);
 expect(info.inputs[0]).toEqual({name:'x',dtype:'FLOAT',shape:['batch',2]});
 expect(info.nodes[0].inputs).toEqual(['x','weight']);expect(info.operators).toEqual([['ai.onnx::Gemm',1]]);
 expect(info.metadata).toEqual([['note','hello']]);expect(info.weights[0].shape).toEqual([2,3]);
});
test('rejects empty, truncated, oversized and unsafe-dimension graphs',()=>{
 for(const data of [new Uint8Array(),new Uint8Array([255]),new Uint8Array(1024*1024+1)]) expect(()=>inspectOnnx(data)).toThrow();
 const bad=onnx.ModelProto.encode({graph:{initializer:[{dims:[Number.MAX_SAFE_INTEGER,2]}]}}).finish();expect(()=>inspectOnnx(bad)).toThrow();
});

test('digest mismatch rejects inspection before parsing',async()=>{
 const {inspectVerifiedOnnx}=await import('../src/utilities/onnx-inspection');
 await expect(inspectVerifiedOnnx(bytes(),'0'.repeat(64))).rejects.toThrow('校验失败');
 const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',Uint8Array.from(bytes()))),v=>v.toString(16).padStart(2,'0')).join('');
 expect((await inspectVerifiedOnnx(bytes(),hash)).parameters).toBe(6);
});
