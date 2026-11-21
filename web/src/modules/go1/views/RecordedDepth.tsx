import {useEffect,useRef} from 'react';

export function paintDepth(canvas:HTMLCanvasElement,width:number,height:number,data:DataView) {
 const context=canvas.getContext('2d');
 if(!context)return;
 if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
 const image=context.createImageData(width,height);
 for(let i=0;i<width*height;i++){
  const shade=255*Math.max(0,Math.min(1,data.getFloat32(i*4,true)));
  const offset=i*4;
  image.data[offset]=image.data[offset+1]=image.data[offset+2]=shade;
  image.data[offset+3]=255;
 }
 context.putImageData(image,0,0);
}

export function RecordedDepth({width,height,f32le}:{width:number;height:number;f32le:string}) {
 const canvas=useRef<HTMLCanvasElement>(null);
 useEffect(()=>{
  if(!canvas.current||!Number.isInteger(width)||!Number.isInteger(height)||width<16||width>128||height<16||height>128)return;
  const bytes=Uint8Array.from(atob(f32le),character=>character.charCodeAt(0));
  if(bytes.length===width*height*4)paintDepth(canvas.current,width,height,new DataView(bytes.buffer));
 },[width,height,f32le]);
 return <canvas ref={canvas} role="img" aria-label="录制深度帧" style={{width:256,maxWidth:'100%',imageRendering:'pixelated'}}/>;
}
