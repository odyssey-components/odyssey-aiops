import type {TableProps} from '@odyssey/ui/primitives/table';
import {Drawer,DrawerBody,DrawerContent} from '@odyssey/ui/primitives/drawer';
import {useState} from 'react';
import {Go1SceneViewDialog} from './SceneViewDialog';
import {Button} from '@odyssey/ui/primitives/button';
import {Table,TableHeader,TableBody,TableRow,TableHead,TableCell} from '@odyssey/ui/primitives/table';
import {Alert} from '@odyssey/ui/primitives/alert';
import {Go1SceneDetail} from './SceneDetail';
import {sceneKinds,sceneSummary,sceneSuccess,type Go1Scene} from '../scenes';
export function Go1Scenes({pagination,scenes,onTrain,selected,onSelect,loading}:{pagination?:TableProps['pagination'];loading?:boolean;selected?:string;onSelect?:(id?:string)=>void;scenes:Go1Scene[];onTrain:(id:string)=>void}){
 const [viewing,setViewing]=useState<Go1Scene>();
 const selectedScene=scenes.find(scene=>scene.id===selected);
 return <>
 <section className="training-workspace" aria-label="Go1 场景"><div className="dataset-heading"><h1>场景</h1></div>
 <Table pagination={pagination} aria-label="Go1 场景列表" containerClassName="dataset-list"><TableHeader><TableRow>{['名称','目标','参数','成功条件','操作'].map(label=><TableHead scope="col" key={label}>{label}</TableHead>)}</TableRow></TableHeader><TableBody>{scenes.map(scene=><TableRow key={scene.id}><TableHead scope="row">{scene.title}</TableHead><TableCell>{sceneKinds[scene.kind]}</TableCell><TableCell>{sceneSummary(scene)}</TableCell><TableCell>{sceneSuccess(scene)}</TableCell><TableCell><Button size="sm" onClick={()=>setViewing(scene)}>查看</Button><Button size="sm" onClick={()=>onSelect?.(scene.id)}>详情</Button>{scene.kind!=='flat'&&<Button size="sm" onClick={()=>onTrain(scene.id)}>创建训练任务</Button>}</TableCell></TableRow>)}</TableBody></Table>
 {!scenes.length&&<p>暂无场景</p>}
 </section>
 <Drawer open={Boolean(selected)} onOpenChange={open=>{if(!open)onSelect?.();}}>
  {selected&&<DrawerContent title="Go1 场景详情" closeLabel="关闭场景详情">
   <DrawerBody>
    {loading?<p role="status">正在加载场景…</p>:selectedScene
     ?<Go1SceneDetail scene={selectedScene} onTrain={()=>onTrain(selectedScene.id)}/>
     :<Alert role="alert" tone="danger">场景不存在</Alert>}
   </DrawerBody>
  </DrawerContent>}
 </Drawer>
 <Go1SceneViewDialog scene={viewing} onClose={()=>setViewing(undefined)}/>
 </>;
}
