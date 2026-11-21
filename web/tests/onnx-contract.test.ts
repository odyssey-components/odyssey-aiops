import { expect, test } from 'bun:test';
import fixedMaze from '../maze.json';
import { generateMaze } from '../src/lessons/local_map_v1';
import { policyFromManifest, validateOnnxManifest } from '../src/utilities/onnx-contract';

const contracts = [
  ['fixed-map-v1', 'odymaze-policy/v1', 'one-hot-xy-direction/v1', 18],
  ['random-map-v1', 'odymaze-policy/v2', 'grid-walls-goal-agent-direction/v1', 151],
  ['local-map-v1', 'odymaze-policy/v3', 'grid-known-walls-goal-visited-agent-direction/v1', 249],
  ['memory-map-v1', 'odymaze-policy/v4', 'local-view-xy-direction-previous-action/v1', 40],
] as const;
for (const [lesson, format, encoding, width] of contracts) {
  test(`strict ONNX manifest contract: ${lesson}`, () => {
    const recurrent = lesson === 'memory-map-v1';
    const tensor = (name: string, size: number) => ({name,dtype:'float32',shape:['batch',size]});
    const manifest = {
      schema:'odymaze-onnx/v1',filename:'model.onnx',sha256:'a'.repeat(64),opset:17,
      lesson, policy_format:format, encoding, maze:lesson === 'fixed-map-v1' ? fixedMaze : {...generateMaze(7),max_steps:lesson === 'random-map-v1'?100:200},
      ...(lesson === 'fixed-map-v1'?{}:{generator:'lcg-corridor-7/v1'}),
      actions:['left','right','forward'],metadata:{seed:7,epochs:0,samples:0,accuracy:0,success:false},
      outputs_are_logits:true, inputs:[tensor('observations',width),...(recurrent?[tensor('hidden',32)]:[])],
      outputs:[tensor('logits',3),...(recurrent?[tensor('next_hidden',32)]:[])],
      ...(recurrent?{recurrent:{initial_hidden:Array(32).fill(0),reset:'per-maze',update:'once-per-action'}}:{}),
    };
    expect(policyFromManifest(validateOnnxManifest(manifest)).layers).toEqual([]);
    const reference={...manifest,maze:undefined,map_ref:{id:manifest.maze.id},dataset:{id:'group-id',sha256:'b'.repeat(64)},
      inputs:manifest.inputs.map(item=>({...item,shape:{batch:'batch',width:item.shape[1]}})),
      outputs:manifest.outputs.map(item=>({...item,shape:{batch:'batch',width:item.shape[1]}}))};
    expect(validateOnnxManifest(reference).maze).toEqual(manifest.maze);
    expect(()=>validateOnnxManifest({...reference,map_ref:{id:'../../maps'}})).toThrow();
    expect(()=>validateOnnxManifest({...reference,dataset:{id:'group-id',sha256:'invalid'}})).toThrow();

    for (const patch of [
      {schema:'other'}, {filename:'../model.onnx'}, {sha256:'z'.repeat(64)}, {opset:18},
      {lesson:'future'}, {policy_format:'other'}, {encoding:'other'}, {actions:['forward','left','right']},
      {maze:{}}, {metadata:{...manifest.metadata,samples:Infinity}}, {outputs_are_logits:false},
      {inputs:[tensor('observations',width+1)]}, {outputs:[tensor('logits',4)]},
      {recurrent:{initial_hidden:[1],reset:'per-maze',update:'once-per-action'}},
    ]) expect(() => validateOnnxManifest({...manifest,...patch})).toThrow();
  });
}
