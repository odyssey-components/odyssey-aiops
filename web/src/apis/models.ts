import type {PageInfo} from '../utilities/list-pagination';
import {apiPath} from './path';
import type { LessonId } from '../../core';

export type ModelEntry = { run: string; lesson: LessonId };
export type ModelSource = { name: string; run: string; read: () => Promise<string> };

export function defaultModel(run: string): ModelSource {
  return { run, name: `${run}/model.onnx`, async read() {
    const response = await fetch(apiPath(`/api/models/${encodeURIComponent(run)}?artifact=manifest`));
    if (response.status === 413) throw new Error("模型文件超过 1 MiB，请重新训练后刷新列表");
    if (!response.ok) throw new Error("无法读取该模型，请刷新列表后重试");
    return response.text();
  }};
}


export async function listModels():Promise<ModelEntry[]> {
  const response=await fetch(apiPath('/api/models?catalog=1'));
  if(!response.ok) throw new Error('无法读取模型列表');
  const entries:unknown=await response.json();
  if(!Array.isArray(entries) || !entries.every(entry=>entry && typeof entry.run==='string'
    && ['fixed-map-v1','random-map-v1','local-map-v1','memory-map-v1'].includes(entry.lesson))) throw new Error('模型列表格式无效');
  return entries;
}

export async function fetchModelArtifact(run: string, artifact: string, limit: number) {
  const response = await fetch(apiPath(`/api/models/${encodeURIComponent(run)}?artifact=${artifact}`), { credentials: "same-origin" });
  if (!response.ok) throw new Error(`无法读取 ONNX ${artifact}（${response.status}），请重新选择模型`);
  const declared = response.headers.get("content-length");
  if (declared && Number(declared) > limit) throw new Error("ONNX 产物超过大小限制");
  if (!response.body) throw new Error("ONNX 产物为空");
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const result = await reader.read();
      if (result.done) break;
      size += result.value.length;
      if (size > limit) throw new Error("ONNX 产物超过大小限制");
      chunks.push(result.value);
    }
  } finally {
    await reader.cancel();
    reader.releaseLock();
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes;
}


export async function listModelPage(course:LessonId,page:number,signal:AbortSignal,pageSize=20):Promise<{items:ModelEntry[];pagination:PageInfo}> {
 const response=await fetch(apiPath('/api/models?'+new URLSearchParams({catalog:'1',course,page:String(page),pageSize:String(pageSize)})),{signal});
 if(!response.ok)throw Error('无法读取模型列表');
 return response.json();
}
