import {lazy,Suspense,Component,type ReactNode} from 'react';
import {Alert} from '@odyssey/ui/primitives/alert';
import {Dialog} from '@odyssey/ui/primitives/dialog';
import type {Go1Scene} from '../scenes';
import '../scene-preview/scene-preview.css';
const Viewport=lazy(()=>import('../scene-preview/Viewport'));
class PreviewBoundary extends Component<{children:ReactNode},{failed:boolean}>{
 state={failed:false};
 static getDerivedStateFromError(){return {failed:true};}
 render(){return this.state.failed?<Alert role="alert" tone="danger">三维组件加载失败，请刷新页面重试</Alert>:this.props.children;}
}
export function Go1SceneViewDialog({scene,onClose}:{scene?:Go1Scene;onClose:()=>void}) {
 if(!scene)return null;
 return <Dialog open onOpenChange={open=>{if(!open)onClose();}} className="go1-scene-view-dialog"
  title={`查看场景 · ${scene.title}`}
  description="场景三维结构与机器狗初始位置" descriptionClassName="sr-only"
  content={<PreviewBoundary key={scene.id}><Suspense fallback={<p role="status">正在加载三维场景…</p>}><Viewport scene={scene}/></Suspense></PreviewBoundary>} cancel={{label:'关闭查看'}}/>;
}
