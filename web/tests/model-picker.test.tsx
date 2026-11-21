import { expect, test } from 'bun:test';
import { renderToStaticMarkup } from 'react-dom/server';
import { ModelPicker } from '../src/components/models/ModelPicker';

test('model picker offers only this course and displays the active model without load or file controls', () => {
  const html = renderToStaticMarkup(<ModelPicker
    course="random-map-v1" initialRuns={[
      {run:'fixed-only',lesson:'fixed-map-v1'},
      {run:'random-active',lesson:'random-map-v1'},
    ]} selectedRun="random-active" onCatalogChange={()=>{}} onLoad={()=>{}} info="model details"
    message={{kind:'neutral',text:''}} />);
  expect(html).toContain('random-active/model.onnx');
  expect(html).not.toContain('fixed-only');
  expect(html).not.toContain('加载所选模型');
  expect(html).not.toContain('type="file"');
  expect(html).toContain('role="combobox"');
  expect(html).toContain('id="model-list"');
  expect(html).not.toContain('请选择模型');
  expect(html).not.toContain('默认使用当前课时列表');
  expect(html).not.toContain('本课有 1 个模型。');
  expect(html).not.toContain('model details');
});

test('refreshing an empty catalog still displays the model currently in memory', () => {
  const html = renderToStaticMarkup(<ModelPicker course="fixed-map-v1" initialRuns={[]}
    selectedRun="current-model" onCatalogChange={()=>{}} onLoad={()=>{}} info="current details" message={{kind:'neutral',text:''}} />);
  expect(html).toContain('current-model/model.onnx');
  expect(html).not.toContain('current details');
});

test('automatic model uses first valid catalog entry of current course', async()=>{
  const {firstCourseModel}=await import('../src/utilities/models');
  const catalog=[{run:'random-new',lesson:'random-map-v1' as const},{run:'fixed-new',lesson:'fixed-map-v1' as const},{run:'fixed-old',lesson:'fixed-map-v1' as const}];
  expect(firstCourseModel(catalog,'fixed-map-v1')?.run).toBe('fixed-new');
  expect(firstCourseModel(catalog,'random-map-v1')?.run).toBe('random-new');
  expect(firstCourseModel([],'fixed-map-v1')).toBeUndefined();
});

test('model picker hides successful descriptions but retains actionable failures',()=>{
  const props={course:'fixed-map-v1' as const,initialRuns:[],onCatalogChange:()=>{},onLoad:()=>{},info:'训练摘要'};
  const success=renderToStaticMarkup(<ModelPicker {...props} message={{kind:'success',text:'已加载 ONNX 模型'}}/>);
  expect(success).not.toContain('已加载 ONNX 模型');expect(success).not.toContain('训练摘要');
  const failure=renderToStaticMarkup(<ModelPicker {...props} message={{kind:'error',text:'模型校验失败'}}/>);
  expect(failure).toContain('role="alert"');expect(failure).toContain('模型校验失败');
});
