import {expect,test} from 'bun:test';
import {renderToStaticMarkup} from 'react-dom/server';
import {ModelView,type ModelViewProps} from '../src/views/ModelView';
const props:ModelViewProps={course:'random-map-v1',catalog:[{run:'random-model',lesson:'random-map-v1'},{run:'fixed-model',lesson:'fixed-map-v1'}],selectedRun:'random-model',loading:false,onRefresh:async()=>{},onDetail:()=>{}};
test('models table scopes rows to current course and marks actual selection',()=>{
 const html=renderToStaticMarkup(<ModelView {...props}/>);
 expect(html).toContain('data-od-slot="table"');expect(html).toContain('random-model/model.onnx');
 expect(html).not.toContain('fixed-model');expect(html).toContain('当前使用');expect(html).toContain('详情');expect(html).not.toContain('使用中');
});
test('models table has honest empty, unavailable and loading states',()=>{
 expect(renderToStaticMarkup(<ModelView {...props} catalog={[]}/>)).toContain('本课暂无有效模型');
 const unavailable=renderToStaticMarkup(<ModelView {...props} catalog={null}/>);
 expect(unavailable).toContain('无法读取模型列表');expect(unavailable).not.toContain('本课暂无有效模型');

});
