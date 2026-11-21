import {Alert} from '@odyssey/ui/primitives/alert';
import {Button} from '@odyssey/ui/primitives/button';
import {Card} from '@odyssey/ui/primitives/card';
import {EmptyState,EmptyStateTitle} from '@odyssey/ui/primitives/empty-state';
import type {DatasetInfo} from '../utilities/training-contract';

const courses={'fixed-map-v1':'第一课：固定地图','random-map-v1':'第二课：随机地图','local-map-v1':'第三课：局部探索','memory-map-v1':'第四课：模型记忆'};
export function DatasetDetail({dataset,loading,error,onBack}:{dataset?:DatasetInfo;loading:boolean;error:string;onBack:()=>void}) {
  return <section className="training-workspace" aria-label="地图组详情">
    <div className="dataset-heading"><h2>地图组详情</h2><Button onClick={onBack}>返回地图组</Button></div>
    {error && <Alert tone="danger" role="alert">{error}</Alert>}
    {loading && !error && <p role="status">正在读取地图组…</p>}
    {!loading && !dataset && <EmptyState><EmptyStateTitle>未找到本课地图组</EmptyStateTitle></EmptyState>}
    {dataset && <Card className="card dataset-detail">
      <h3>{dataset.title}</h3>
      <dl>
        <dt>说明</dt><dd className="dataset-description">{dataset.description || '—'}</dd>
        <dt>课时</dt><dd>{courses[dataset.course]}</dd>
        <dt>训练地图</dt><dd>{dataset.maps} 张</dd>
        <dt>验证地图</dt><dd>{dataset.validation} 张</dd>
        <dt>测试地图</dt><dd>{dataset.test} 张</dd>
        <dt>随机种子</dt><dd>{dataset.seed}</dd>
        <dt>创建时间</dt><dd><time dateTime={dataset.createdAt}>{new Date(dataset.createdAt).toLocaleString('zh-CN',{
          timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23',
        })}</time></dd>
        <dt>地图组 ID</dt><dd>{dataset.id}</dd>
        <dt>地图文件</dt><dd>{dataset.id}.jsonl</dd>
        <dt>元信息文件</dt><dd>{dataset.id}.toml</dd>
        <dt>SHA-256</dt><dd>{dataset.sha256}</dd>
      </dl>
    </Card>}
  </section>;
}
