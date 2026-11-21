import {defaultModel,listModels,type ModelEntry,type ModelSource} from '../../apis/models';
import { Alert } from "@odyssey/ui/primitives/alert";
import { Button } from "@odyssey/ui/primitives/button";
import { Card } from "@odyssey/ui/primitives/card";
import { useEffect, useRef, useState } from "react";
import type { LessonId } from "../../../core";
import { Choice } from "../common/Choice";

interface ModelPickerProps {
  initialRuns: ModelEntry[] | null;
  course: LessonId;
  selectedRun?: string;
  loading?: boolean;
  onCatalogChange: (catalog:ModelEntry[]) => void;
  onLoad: (source:ModelSource) => void;
  info: string;
  message: { text: string; kind: "neutral" | "error" | "success" };
}

export function ModelPicker({ initialRuns, course, selectedRun, loading, onCatalogChange, onLoad, message }: ModelPickerProps) {
  const [catalog, setCatalog] = useState(initialRuns);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(false);
  const listVersion = useRef(0);
  useEffect(() => { setCatalog(initialRuns); }, [initialRuns]);
  useEffect(() => () => { ++listVersion.current; }, []);
  const models = catalog?.filter(entry => entry.lesson === course) ?? [];

  async function refresh() {
    const version = ++listVersion.current;
    setRefreshing(true);
    setError(false);
    try {
      const entries=await listModels();
      if (version === listVersion.current) {setCatalog(entries);onCatalogChange(entries);}
    } catch {
      if (version === listVersion.current) setError(true);
    } finally {
      if (version === listVersion.current) setRefreshing(false);
    }
  }

  return <Card className="card model-card">
    <h2>本课训练模型</h2>
    <div className="model-directory-controls">
      <div id="current-model" title={selectedRun ? `${selectedRun}/model.onnx` : undefined}><Choice id="model-list" label="当前模型" value={selectedRun ?? ""}
        disabled={loading || !models.length}
        placeholder={loading ? "正在加载本课模型…" : selectedRun ? `${selectedRun}/model.onnx` : "本课暂无已加载模型"}
        options={models.map(entry=>({value:entry.run,label:`${entry.run}/model.onnx`}))}
        onChange={run=>{if(run!==selectedRun) onLoad(defaultModel(run));}} /></div>
      <Button size="sm" id="refresh-models" disabled={refreshing || loading} onClick={refresh}>刷新模型列表</Button>
    </div>
    {(refreshing || error || catalog===null) && <p id="model-directory-status" className="model-info" role="status">{refreshing ? "正在读取训练目录…" : "无法读取固定训练目录，请刷新列表重试。"}</p>}
    {message.kind==='error' && message.text && <Alert id="import-message" role="alert" tone="danger" data-kind="error">{message.text}</Alert>}
  </Card>;
}
