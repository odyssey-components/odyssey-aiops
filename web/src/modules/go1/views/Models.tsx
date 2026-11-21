import type {TableProps} from '@odyssey/ui/primitives/table';
import {Drawer,DrawerContent,DrawerBody} from '@odyssey/ui/primitives/drawer';
import {Button} from '@odyssey/ui/primitives/button';
import {Card} from '@odyssey/ui/primitives/card';
import {Table,TableHeader,TableBody,TableRow,TableHead,TableCell} from '@odyssey/ui/primitives/table';
import type {Go1Snapshot} from '../contract';
function Facts({value}:{value:unknown}) {return <dl>{value && typeof value==='object'?Object.entries(value).map(([key,item])=><div key={key}><dt>{key}</dt><dd>{typeof item==='object'?JSON.stringify(item):String(item ?? '未记录')}</dd></div>):<dd>未记录</dd>}</dl>;}
export function Go1Models({pagination,snapshot,selected,onSelect,onSimulate}:{pagination?:TableProps['pagination'];snapshot:Go1Snapshot|null;selected?:string;onSelect:(id?:string)=>void;onSimulate:(id:string)=>void}) {
 const model=snapshot?.selectedModel?.run===selected?snapshot?.selectedModel:snapshot?.models.find(model=>model.run===selected);
 const base=snapshot?.selectedBaseModel?.id===selected?snapshot?.selectedBaseModel:snapshot?.baseModels?.find(model=>model.id===selected);
 const rows=[
  ...(snapshot?.models ?? []).map(item=>({id:item.run,title:item.title,format:String(item.manifest.format ?? item.manifest.kind ?? 'NumPy NPZ'),createdAt:item.createdAt})),
  ...(snapshot?.baseModels ?? []).filter(item=>!snapshot?.models.some(model=>model.run===item.id)).map(item=>({id:item.id,title:item.title,format:'上游 SB3 · 45 状态 / 12 动作',createdAt:undefined})),
 ];
 return <section className="training-workspace" aria-label="Go1 模型"><div className="dataset-heading"><h1>Go1 模型</h1></div>
 <Table pagination={pagination} aria-label="Go1 模型列表" containerClassName="dataset-list"><TableHeader><TableRow>{['名称','格式','导入时间','操作'].map(label=><TableHead key={label} scope="col">{label}</TableHead>)}</TableRow></TableHeader>
 <TableBody>{rows.map(item=><TableRow key={item.id}><TableHead scope="row">{item.title}</TableHead><TableCell>{item.format}</TableCell><TableCell>{item.createdAt?new Date(item.createdAt).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false}):'—'}</TableCell><TableCell><div className="training-actions"><Button size="sm" onClick={()=>onSelect(item.id)}>详情</Button><Button size="sm" onClick={()=>onSimulate(item.id)}>创建仿真任务</Button></div></TableCell></TableRow>)}</TableBody></Table>
 {snapshot&&!rows.length&&<p>暂无可用 Go1 模型</p>}
 <Drawer open={Boolean(selected)} onOpenChange={open=>{if(!open)onSelect();}}>
 {selected&&<DrawerContent title="Go1 模型详情" closeLabel="关闭模型详情"><DrawerBody>
 {selected && snapshot && !model && !base && <p>未找到有效 Go1 模型</p>}
 {base && !model && <Card className="card dataset-detail"><h3>{base.title}</h3><dl><dt>模型 ID</dt><dd>{base.id}</dd><dt>格式</dt><dd>上游 SB3 默认 MLP · NumPy NPZ</dd><dt>Policy SHA-256</dt><dd>{base.policySha256}</dd><dt>Manifest SHA-256</dt><dd>{base.manifestSha256}</dd></dl></Card>}
 {model&&<Card className="card dataset-detail"><h3>{model.title}</h3><p>{model.manifest.format==='go1-robot-parkour/v1'?'闭环仿真与视觉回灌 · 低横杆及跳跃尚未通过验证':model.manifest.simulation_only?'闭环仿真 · 观测历史与关节位置控制':'闭环仿真与视觉回灌'}</p><h4>权重来源</h4><Facts value={model.manifest}/></Card>}
 </DrawerBody></DrawerContent>}
 </Drawer>
 </section>;
}
